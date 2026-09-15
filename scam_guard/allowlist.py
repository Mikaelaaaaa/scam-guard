"""Tranco 排名白名單的**查詢** —— 讀本機快照，無網路、無外部格式知識。

**它量的是流量，不是品行。** Tranco 把多個提供者的排名平均過去 30 天，
一個網域要進前 1,000 名必須在真實瀏覽器導覽、DNS 查詢量、反向連結上同時
且持續居高位。這讓「前 1,000 名」成為一個攻擊成本的下限（165027 實測的
「網站創建 → 被通報」中位數是 44 天，大約就是那個平均窗口的長度），
但它**不是**「這個網站是良性的」的證明 —— 實測 Tranco 前一百萬名裡直接
含有當天仍在釣魚的 `roblox.com.mu`、`robiox.com.py`、`roblox.com.ml`。

因此本模組只回答一件事：**這個可註冊網域在清單上的第幾名**。
要不要因此收窄某個訊號，是 `url_check` 的事；而它只被允許收窄
`url_blocklist` 一個檢查。

**apex 規則是這份白名單的主要防線，而它在 `url_check` 那邊。**
快照裡只有可註冊網域（取得程式已丟掉本身即為 public suffix 的項目），
所以查詢的鍵必然是一個可註冊網域；`https://sites.google.com/view/xxx` 的
主機是 `sites.google.com`，它永遠不會等於快照裡的 `google.com`。
PSL 的 PRIVATE 區段擋掉了前 1,000 名裡 42 個平台後綴（`blogspot.com`、
`github.io`、`wixsite.com`…），但實測它**沒有**收錄 `godaddysites.com`、
`weebly.com`、`amazonaws.com`、`myqcloud.com`、`sites.google.com`
—— 那 42 個是運氣，剩下的全靠 apex 規則。

**查詢回傳名次而不是布林值。** 依據文案要寫得出「`google.com` 為 Tranco
清單 GQNVK 第 1 名」這個可查證的事實，而一個布林值回答不了它；
`add-ablation` 掃描較低門檻時也要能直接過濾同一份快照。
成本實測為 0（前 1,000 名的 `dict[str, int]` 追蹤到 25 KB）。
"""

import json
from hashlib import sha256
from pathlib import Path

from scam_guard.url import parse_iso_date, read_manifest, utc_today

ENTRIES_FILENAME = "entries.jsonl"
MANIFEST_FILENAME = "manifest.json"
FETCH_COMMAND = "python -m tools.fetch_tranco"

ENTRY_FIELDS = ("domain", "rank")
MANIFEST_FIELDS = (
    "list_id",
    "permalink",
    "source_url",
    "threshold",
    "fetched_at",
    "data_through",
    "data_through_source",
    "raw_row_count",
    "kept_count",
    "dropped_public_suffix_count",
    "entries_file",
    "entries_sha256",
    "attribution",
)


class RankAllowlist:
    """本機快照的查詢介面。載入為顯式動作，模組層級不做任何檔案讀取。

    理由與 `BlocklistStore` 相同：一個沒有跑過取得程式的環境（CI 正是這樣的
    環境）必須仍能執行整個測試套件。

    常駐的只有門檻 N 以內的項目。完整的一百萬筆實測載入後 RSS 增量為
    128.5 MB（集合）至 155.0 MB（帶名次的字典），與黑名單本身（143.6 MB）
    同一量級 —— 一個壓誤判的輔助元件不該與主要證據來源吃掉一樣多的記憶體，
    而執行環境是 HF Spaces 免費層，Gradio 與 Gemma 3 1B 在同一個程序裡。
    """

    def __init__(self, ranks: dict[str, int], manifest: dict[str, object]) -> None:
        self._ranks = ranks
        self._manifest = manifest

    @property
    def manifest(self) -> dict[str, object]:
        """快照的中繼資料。`url_check` 由此取得 `list_id` 寫進依據文案。"""
        return self._manifest

    @property
    def entry_count(self) -> int:
        return len(self._ranks)

    @classmethod
    def load(cls, path: str | Path, *, max_age_days: int) -> "RankAllowlist":
        """自本機快照目錄載入。`max_age_days` 為必填的僅限關鍵字參數。

        **沒有預設值。** Tranco 明文說明每日 0:00 UTC 更新，所以 30 天是一個
        有依據的起點（一份 30 天沒更新的白名單意味著 `tranco-list.eu` 停止
        服務了），但那是部署環境的知識，寫進程式的預設值等於替呼叫端決定。

        **過期時拋例外，不降級。** 一份過期的白名單造成的傷害不是漏抓，
        是繼續替一批可能已經易主的網域背書。要降級的話，正確做法是組裝層
        捕捉這個例外後傳 `allowlist=None` —— 系統明確地少一個修正，
        而不是用舊資料。

        一致性檢查全部在此完成，MUST NOT 於每次查詢時執行。
        """
        directory = Path(path)
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"白名單 manifest 不存在：{manifest_path}。請先執行 `{FETCH_COMMAND}` 取得"
            )
        manifest = read_manifest(manifest_path)
        for field in MANIFEST_FIELDS:
            if field not in manifest:
                raise ValueError(
                    f"白名單 manifest 缺少欄位 {field!r}：{manifest_path}，"
                    f"實際欄位為 {sorted(manifest)}"
                )
        entries_path = directory / str(manifest["entries_file"])
        if not entries_path.is_file():
            raise FileNotFoundError(
                f"白名單快照不存在：{entries_path}。請先執行 `{FETCH_COMMAND}` 取得"
            )
        raw = entries_path.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != manifest["entries_sha256"]:
            raise ValueError(
                f"白名單快照的 sha256 與 manifest 不符：檔案為 {digest}、"
                f"manifest 為 {manifest['entries_sha256']}"
            )
        _check_freshness(manifest, max_age_days)
        ranks = _parse_entries(raw.decode("utf-8"), entries_path, manifest["threshold"])
        if not ranks:
            raise ValueError(f"白名單快照的總筆數為 0：{entries_path}")
        return cls(ranks, manifest)

    def rank(self, domain: str) -> int | None:
        """可註冊網域的名次；不在清單上時回傳 `None`。一次字典查表。"""
        return self._ranks.get(domain)


def _check_freshness(manifest: dict[str, object], max_age_days: int) -> None:
    """以 `data_through` 判定新鮮度。

    `data_through` 取自下載回應的 HTTP `Last-Modified`，這是一個**比黑名單
    （由資料內容解析）弱**的判準 —— manifest 的 `data_through_source`
    因此把兩者區分開，不可混為一談。
    """
    data_through = parse_iso_date(manifest["data_through"], "data_through")
    age_days = (utc_today() - data_through).days
    if age_days > max_age_days:
        raise ValueError(
            f"白名單過期：data_through 為 {data_through.isoformat()}、"
            f"距今 {age_days} 天、max_age_days 為 {max_age_days}。"
            f"請重新執行 `{FETCH_COMMAND}`"
        )


def _parse_entries(text: str, source_path: Path, threshold: object) -> dict[str, int]:
    """逐行解析 JSON Lines。缺欄位、名次非整數或名次超出門檻時拋例外。

    名次超出門檻要擋下來，是因為「常駐項目只含門檻以內的項目」是這個模組對
    記憶體的唯一承諾，而一份被手改過的快照可以安靜地打破它。
    """
    if not isinstance(threshold, int):
        raise ValueError(f"白名單 manifest 的 threshold 必須為整數，實為 {threshold!r}")
    ranks: dict[str, int] = {}
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
        rank = record["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"{source_path} 第 {line_number} 行的 rank 必須為整數，實為 {rank!r}")
        if rank > threshold:
            raise ValueError(
                f"{source_path} 第 {line_number} 行的 rank 為 {rank}，"
                f"超出 manifest 的 threshold {threshold}"
            )
        ranks[record["domain"]] = rank
    return ranks
