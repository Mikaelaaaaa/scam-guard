"""165 涉詐網址黑名單的**查詢** —— 讀本機快照，無網路、無外部格式知識。

此模組只認識一個**我們自己定義**的格式（`entries.jsonl` 加 `manifest.json`）。
data.gov.tw 的下載網址、欄位名稱、編碼、民國紀年全部只出現在
`tools/fetch_blocklist.py`。

**界線畫在「誰知道外部格式」，不是「誰上網」。** 理由不是乾淨，是資料集改版時
壞在哪裡：若本模組知道欄位叫什麼，改版會在一個線上請求裡拋 `KeyError`，
而 `check.py` 要求檢查自行捕捉例外回傳空陣列 —— 於是黑名單會**安靜地變成空的**。
一個看起來正常運作、但硬證據永遠不命中的系統，是最怕的失敗模式。

**新鮮度以 `data_through` 判定，不以 `fetched_at` 判定。** 這條是查證之後改掉的：
160055（假投資博弈網站）的詮釋資料在 2026-07-29 還更新過，下載下來
`fetched_at` 永遠是今天，而檔案內最新的一筆停在 2025-12-31。
只看 `fetched_at`，一份內容停滯九個月的清單看起來永遠新鮮。

**`retired` 的 source 跳過新鮮度檢查。** 160055 的資料集描述明文指出它已被
176455 取代，不會再更新。仍然納入是因為黑名單過期意味著**召回下降**而不是
**答案變錯** —— 一個 2024 年被通報的假投資網站，今天仍然是一個假投資網站。
但它不能擋住整個系統啟動。

**本層不判定類型與強度。** 查詢回傳**紀錄**而不是布林值，每筆帶自己的 `source`，
因為三個資料集說的不是同一件事。映射到 `ScamType` 與 `hard` 是 `url_check` 的事。
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from scam_guard.url import PublicSuffixList, parse_iso_date, read_manifest, utc_today

ENTRIES_FILENAME = "entries.jsonl"
MANIFEST_FILENAME = "manifest.json"
FETCH_COMMAND = "python -m tools.fetch_blocklist"

ENTRY_FIELDS = ("host", "url", "source", "first_seen", "last_seen")
SOURCE_MANIFEST_FIELDS = (
    "title",
    "agency",
    "source_url",
    "record_count",
    "unique_host_count",
    "data_through",
    "data_through_granularity",
    "retired",
    "retired_reason",
)
MANIFEST_FIELDS = ("entries_file", "entries_sha256", "fetched_at", "license", "sources")


@dataclass(frozen=True, slots=True)
class Entry:
    """快照中的一筆紀錄。

    `slots=True` 不是微調：`project.md` 的執行環境是 HuggingFace Spaces
    免費層（2 vCPU / 16 GB），而 Gradio 與 Gemma 3 1B 在**同一個程序**。
    三份資料合計十餘萬筆，每個實例多帶一個 `__dict__` 就是數十 MB，
    而它和模型權重擠同一塊記憶體。

    `nature`（網站性質）**只有 176455 有**，且是**不透明字串**。
    它是被冒用的**產業**（金融保險 47,347、電子商務 23,352…）不是 165 案類，
    而且是帶錯字的自由文字（實測有「金融保健」66 筆、「金融保線」2 筆）。
    一個帶錯字的自由文字欄位不可能當 Enum 的鍵，所以它只用於依據文案。

    `site_created_on`（詐騙網站創建日期）**只有 165027 有**。
    它是本機就有的網域年齡資料，那 1,574 個主機不需要任何對外查詢。
    """

    host: str
    url: str
    source: str
    first_seen: str
    last_seen: str
    nature: str | None = None
    site_created_on: str | None = None


class BlocklistStore:
    """本機快照的查詢介面。載入為顯式動作，模組層級不做任何檔案讀取。

    載入後以雜湊表常駐記憶體：約十三萬筆、十餘萬個唯一主機，
    查詢是一次字典查表，遠優於「單次查詢 < 1 毫秒」的驗收條件。
    不用 SQLite —— 它的優勢在跨行程共享與部分載入，而這裡是唯讀、全量、
    每次請求查數次。
    """

    def __init__(
        self,
        entries: tuple[Entry, ...],
        manifest: dict[str, object],
        psl: PublicSuffixList,
    ) -> None:
        self._manifest = manifest
        by_host: dict[str, list[Entry]] = {}
        by_domain: dict[str, list[Entry]] = {}
        for entry in entries:
            by_host.setdefault(entry.host, []).append(entry)
            # 可註冊網域在**載入時**以傳入的 PSL 算，不讀快照裡預算好的值 ——
            # 寫進快照就同時綁住了黑名單版本與 PSL 版本，而兩者更新週期不同。
            # PSL 新增一條 PRIVATE 後綴之後，舊快照裡的可註冊網域就是錯的，
            # 而沒有任何地方會發現。
            domain = psl.registrable_domain(entry.host)
            if domain is not None:
                by_domain.setdefault(domain, []).append(entry)
        self._by_host = {host: tuple(items) for host, items in by_host.items()}
        self._by_domain = {domain: tuple(items) for domain, items in by_domain.items()}

    @property
    def manifest(self) -> dict[str, object]:
        """快照的中繼資料。`url_check` 由此取得資料集的正式名稱寫進依據文案。"""
        return self._manifest

    @property
    def entry_count(self) -> int:
        return sum(len(items) for items in self._by_host.values())

    @classmethod
    def load(
        cls,
        path: str | Path,
        psl: PublicSuffixList,
        *,
        max_age_days: int,
    ) -> "BlocklistStore":
        """自本機快照目錄載入。`max_age_days` 為必填的僅限關鍵字參數。

        **沒有預設值**，這是「禁止臆測性 fallback」在資料新鮮度上的直接套用：
        三個資料集的宣告更新頻率都是「不定期更新」，所以沒有任何一個數字
        可以從公告推出來 —— 預設值會假裝我們知道一件我們不知道的事。

        **超過門檻拋例外，不是印警告。** 一份過期的黑名單不是「稍微差一點的
        黑名單」，它對最近數月的新網域**系統性地全部漏掉**。要降級的話，
        正確做法是組裝層捕捉這個例外後**不註冊**黑名單檢查（系統明確地少一個
        訊號），而不是讓 store 自己決定繼續用舊資料。

        一致性檢查全部在此完成，MUST NOT 於每次查詢時執行。
        """
        directory = Path(path)
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"黑名單 manifest 不存在：{manifest_path}。請先執行 `{FETCH_COMMAND}` 取得"
            )
        manifest = read_manifest(manifest_path)
        for field in MANIFEST_FIELDS:
            if field not in manifest:
                raise ValueError(
                    f"黑名單 manifest 缺少欄位 {field!r}：{manifest_path}，"
                    f"實際欄位為 {sorted(manifest)}"
                )
        entries_path = directory / str(manifest["entries_file"])
        if not entries_path.is_file():
            raise FileNotFoundError(
                f"黑名單快照不存在：{entries_path}。請先執行 `{FETCH_COMMAND}` 取得"
            )
        raw = entries_path.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != manifest["entries_sha256"]:
            raise ValueError(
                f"黑名單快照的 sha256 與 manifest 不符：檔案為 {digest}、"
                f"manifest 為 {manifest['entries_sha256']}"
            )
        _check_freshness(manifest["sources"], max_age_days)
        entries = _parse_entries(raw.decode("utf-8"), entries_path)
        if not entries:
            raise ValueError(f"黑名單快照的總筆數為 0：{entries_path}")
        return cls(entries, manifest, psl)

    def by_host(self, host: str) -> tuple[Entry, ...]:
        """以主機精確比對。未命中回傳空序列。

        比對只用主機，不用路徑：釣魚站的路徑常帶一次性參數，通報時記下的
        那一個與受害者收到的那一個不會一樣；而不帶 scheme 的寫法其 query
        不保證完整（見 `url.py`），用一個不保證完整的東西當主鍵是錯的。
        """
        return self._by_host.get(host, ())

    def by_registrable_domain(self, domain: str) -> tuple[Entry, ...]:
        """以可註冊網域比對，回傳該網域下全部被通報的主機。

        必要性來自實測：176455 的 34,770 個唯一可註冊網域中，有 1,141 個
        （3.28%）底下不只一個被通報的主機，涉及 42,013 個主機（55.54%）。
        前幾名是 `word1018.shop`（3,006 個子網域）這種一個詐騙者把數字子網域
        窮舉開出來的形態 —— 通報的是其中一個，受害者收到的是另一個。
        只做精確比對就會漏掉。

        **與 `by_host()` 分開回答**，兩者強度不同，而 store 不該替 `url_check`
        決定強度。
        """
        return self._by_domain.get(domain, ())


def _check_freshness(sources: object, max_age_days: int) -> None:
    """逐 source 以 `data_through` 判定新鮮度，`retired` 者跳過。

    錯誤訊息必須**指名道姓** —— 「黑名單過期」這種訊息會讓人重跑取得程式
    然後發現沒用（那個 source 根本不會再更新了）。
    """
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"黑名單 manifest 的 sources 必須為非空物件，實為 {sources!r}")
    today = utc_today()
    for dataset_id, meta in sources.items():
        if not isinstance(meta, dict):
            raise ValueError(f"黑名單 manifest 的 sources[{dataset_id!r}] 必須為物件")
        for field in SOURCE_MANIFEST_FIELDS:
            if field not in meta:
                raise ValueError(
                    f"黑名單 manifest 的 sources[{dataset_id!r}] 缺少欄位 {field!r}，"
                    f"實際欄位為 {sorted(meta)}"
                )
        if meta["retired"]:
            continue
        data_through = parse_iso_date(meta["data_through"], f"sources[{dataset_id!r}].data_through")
        age_days = (today - data_through).days
        if age_days > max_age_days:
            raise ValueError(
                f"黑名單過期：資料集 {dataset_id} 的 data_through 為 "
                f"{data_through.isoformat()}、距今 {age_days} 天、"
                f"max_age_days 為 {max_age_days}。請重新執行 `{FETCH_COMMAND}`"
            )


def _parse_entries(text: str, source_path: Path) -> tuple[Entry, ...]:
    """逐行解析 JSON Lines。缺 `host` 或 `source` 的紀錄拋例外並指出行號。"""
    entries: list[Entry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{source_path} 第 {line_number} 行不是 JSON 物件：{line!r}")
        for field in ENTRY_FIELDS:
            if field not in record:
                raise ValueError(
                    f"{source_path} 第 {line_number} 行缺少欄位 {field!r}，"
                    f"實際欄位為 {sorted(record)}"
                )
        entries.append(
            Entry(
                host=record["host"],
                url=record["url"],
                source=record["source"],
                first_seen=record["first_seen"],
                last_seen=record["last_seen"],
                nature=record["nature"] if "nature" in record else None,
                site_created_on=(
                    record["site_created_on"] if "site_created_on" in record else None
                ),
            )
        )
    return tuple(entries)
