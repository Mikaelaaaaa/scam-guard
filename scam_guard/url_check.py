"""URL 層的五個檢查 —— 全部本機比對，無任何網路存取。

`url_blocklist`、`url_shortener`、`url_tld_risk`、`url_host_shape`、`url_brand`
是**五個獨立註冊的檢查**，不是一個。分開的三個理由，任一個都足夠：

1. **`hard` 不同。** 只有 `url_blocklist` 的精確主機命中夠格標 `hard=True`，
   而那會觸發 `detect()` 的短路、決定 LLM 跑不跑。合成一個檢查就只有一個 `hard`。
2. **權重不同。** 黑名單與 TLD 風險的條件機率差一到兩個數量級。
3. **消融要逐項。** `registry.disable("url_tld_risk")` 才能回答
   「TLD 風險到底有沒有貢獻」，而那個問題的答案可能是「沒有」——
   也就是說分開正是為了能把東西刪掉。

**五個檢查皆不輸出 `FAKE_PARCEL` 與 `ORDER_ANOMALY`。**
兩者的判定要件都是連言，其中「冒稱物流通知」「冒稱訂單異常」是一段中文，
在 `doc.sentences` 裡，不在主機名裡。有人會說 `tw-711-parcel.com` 看起來就是包裹，
但同一個網域同樣可以承載「7-11 抽獎中獎通知」（`FAKE_PRIZE`）——
從主機名猜案類，是用一個 0.5 準確度的線索去覆蓋一個文字層 0.9 準確度的判斷。

更嚴重的是它會製造一個**沒有辦法仲裁的衝突**：`add-type-resolve` 的規則是
「規則與 LLM 衝突時規則優先」，但這裡衝突的是**兩個規則層訊號**，沒有優先順序
可用。URL 層只輸出 `PHISHING_LINK` 就沒有這個問題 ——
`PHISHING_LINK` 描述的是**手段**，`FAKE_PARCEL` 描述的是**託辭**，兩者可以並存。

**對照表放 `scam_guard/tables/` 而非 `scam_guard/data/`：** `.gitignore` 有一行
`data/`（無前導斜線，於任何深度生效）與一行 `*.jsonl`，取名 `data` 會讓這三份
必須進版控的檔案被靜默忽略。
"""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Iterable

from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore, Entry
from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document
from scam_guard.types import CheckResult, Coord, Request, ScamType
from scam_guard.url import ExtractedUrl, PublicSuffixList, extract_urls

TABLES_PACKAGE = "scam_guard.tables"

# 黑名單的 `source` → 輸出類型。
# 176455 的「網站性質」MUST NOT 進這張表 —— 它是被冒用的**產業**（金融保險
# 47,347 筆）而不是 165 案類，且是帶錯字的自由文字（「金融保健」66 筆）。
# 165027 對應的 165 案類是「網路購物」，而 `add-scam-type` 明確排除了那一類
# （由交易結果界定），所以它輸出 `PHISHING_LINK`，MUST NOT 輸出「網路購物」。
SOURCE_SCAM_TYPES: dict[str, ScamType] = {
    "176455": ScamType.PHISHING_LINK,
    "160055": ScamType.FAKE_INVESTMENT,
    "165027": ScamType.PHISHING_LINK,
}

# 混合文字系統的判定範圍。完整的 UTS #39 Restriction Level 需要完整的 script
# 資料，而 stdlib 沒有 `unicodedata.script`，因此做簡化版：只判單一標籤內是否
# 混用拉丁、西里爾、希臘三種文字，以字元區間表實作，可逐項測試。
#
# **誠實揭露簡化的代價**：中日韓與拉丁混用（`中国.com` 之類）不會命中，
# 而那在台灣是**合法且常見**的 —— 所以這個簡化剛好對我們有利，
# 但它的理由是實作成本，不是設計正確性。若之後發現 CJK 同形異義字規避
# （例如用「巳」冒充「己」），要換成完整實作。
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "拉丁": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)),
    "西里爾": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "希臘": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
}

# 視覺替換表。不是從某個來源抄的 —— 依據是「在 LINE 與手機瀏覽器使用的
# 無襯線字體下這幾組字形幾乎相同」，可以截圖驗證。
# `rn` → `m` 放在前面，因為它是兩字元對一字元，先做才不會被單字元替換打斷。
VISUAL_MULTI_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (("rn", "m"),)
VISUAL_SUBSTITUTIONS: dict[str, str] = {"0": "o", "1": "l"}

DEFAULT_SHORT_LABEL_LENGTH = 8
DEFAULT_SHORT_LABEL_DISTANCE = 1
DEFAULT_LONG_LABEL_DISTANCE = 2

# 短於此長度的官方主標籤**不做編輯距離判定**（仍可由 `label_tokens` 的子字串
# 比對與完整官方網域字串命中）。
#
# ⚠️ 5 這個值沒有實驗依據，但它有一個實測的動機：把 `lin.ee`（LINE 官方短網址）
# 列為官方網域之後，主標籤 `lin` 只有三個字元，而距離 1 的判定讓
# `ltn.com.tw`（自由時報）在 Cofacts 樣本中誤判 14 次。
# 三個字元差一個字元代表三分之一的字串不同 —— 那不是「近似拼寫」，是巧合。
DEFAULT_MIN_LABEL_LENGTH_FOR_DISTANCE = 5


@dataclass(frozen=True, slots=True)
class Brand:
    """一個被冒用品牌。`official_domains` 為**可註冊網域**清單，不是主機。"""

    chinese_name: str
    english_name: str
    official_domains: tuple[str, ...]
    label_tokens: tuple[str, ...]
    source: str
    verified_on: str


@dataclass(frozen=True, slots=True)
class Shortener:
    domain: str
    service: str
    source: str
    verified_on: str


@dataclass(frozen=True, slots=True)
class TldRisk:
    tld: str
    metric: str
    value: float
    as_of: str
    source: str


@dataclass(frozen=True, slots=True)
class Tables:
    """三份對照表，由 `load_tables()` 顯式載入。"""

    brands: tuple[Brand, ...]
    shorteners: dict[str, Shortener]
    tld_risk: dict[str, TldRisk]
    baseline_tld: str
    baseline_value: float

    @property
    def official_domains(self) -> frozenset[str]:
        return frozenset(domain for brand in self.brands for domain in brand.official_domains)

    @property
    def known_service_domains(self) -> frozenset[str]:
        """短網址服務與品牌官方網域的聯集。TLD 風險對這些網域不產出結果。"""
        return self.official_domains | frozenset(self.shorteners)


def load_tables() -> Tables:
    """以 `importlib.resources` 讀取三份對照表。**顯式動作，不在 import 時發生。**

    用 `importlib.resources` 而非相對路徑拼接：正式安裝（非可編輯安裝）後
    `__file__` 旁邊不一定有那些檔案，而這種錯在 `pip install -e` 下不會出現。
    """
    brands_raw = _read_table("brands.json")
    shorteners_raw = _read_table("shorteners.json")
    tld_raw = _read_table("tld_risk.json")

    brands = []
    for index, item in enumerate(brands_raw["brands"]):
        for field in ("chinese_name", "official_domains", "source", "verified_on"):
            if field not in item:
                raise ValueError(
                    f"brands.json 第 {index} 筆缺少欄位 {field!r}，實際欄位為 {sorted(item)}"
                )
        if not item["official_domains"]:
            raise ValueError(
                f"brands.json 第 {index} 筆（{item['chinese_name']}）的 official_domains 為空"
            )
        brands.append(
            Brand(
                chinese_name=item["chinese_name"],
                english_name=item["english_name"],
                official_domains=tuple(item["official_domains"]),
                label_tokens=tuple(item["label_tokens"]),
                source=item["source"],
                verified_on=item["verified_on"],
            )
        )

    shorteners = {}
    for index, item in enumerate(shorteners_raw["shorteners"]):
        for field in ("domain", "service", "source", "verified_on"):
            if field not in item:
                raise ValueError(
                    f"shorteners.json 第 {index} 筆缺少欄位 {field!r}，實際欄位為 {sorted(item)}"
                )
        shorteners[item["domain"]] = Shortener(
            domain=item["domain"],
            service=item["service"],
            source=item["source"],
            verified_on=item["verified_on"],
        )

    tld_risk = {}
    for index, item in enumerate(tld_raw["tlds"]):
        for field in ("tld", "metric", "value", "as_of", "source"):
            if field not in item:
                raise ValueError(
                    f"tld_risk.json 第 {index} 筆缺少欄位 {field!r}，實際欄位為 {sorted(item)}"
                )
        tld_risk[item["tld"]] = TldRisk(
            tld=item["tld"],
            metric=item["metric"],
            value=item["value"],
            as_of=item["as_of"],
            source=item["source"],
        )

    return Tables(
        brands=tuple(brands),
        shorteners=shorteners,
        tld_risk=tld_risk,
        baseline_tld=tld_raw["baseline"]["tld"],
        baseline_value=tld_raw["baseline"]["value"],
    )


def _read_table(name: str) -> dict:
    return json.loads(resources.files(TABLES_PACKAGE).joinpath(name).read_text(encoding="utf-8"))


def group_by_registrable_domain(
    urls: Iterable[ExtractedUrl],
) -> dict[str, list[ExtractedUrl]]:
    """依可註冊網域分組，順序為首次出現順序。`None` 者略過。

    網域層的訊號（黑名單、TLD 風險、網域年齡）講的都是網域這一件事：
    `evil.com/a`、`evil.com/b`、`evil.com/c` 各產一筆的話，UI 上是三行
    一模一樣的依據，而 `detail` 也完全相同。
    """
    groups: dict[str, list[ExtractedUrl]] = {}
    for url in urls:
        if url.registrable_domain is None:
            continue
        groups.setdefault(url.registrable_domain, []).append(url)
    return groups


def coords_of(urls: Iterable[ExtractedUrl]) -> list[Coord]:
    """一組 URL 出現過的全部座標，去重且保持順序。"""
    seen: dict[Coord, None] = {}
    for url in urls:
        for coord in url.coords:
            seen[coord] = None
    return list(seen)


class UrlBlocklistCheck:
    """165 黑名單比對 —— 全系統唯一可能標 `hard=True` 的 URL 訊號。

    比對分兩個層級。**主機精確命中**代表官方已依法聲請停止解析，是已發生的
    事實，`hard=True`。**可註冊網域命中但主機不同**只標 `hard=False` ——
    PSL 的 PRIVATE 區段不完整（社群自願送交），那個命中可能來自同一個託管平台
    上的鄰居。

    `hard` 的門檻要抓這麼緊，是因為它的後果不只是分數：`detect()` 的短路會讓
    LLM 完全不跑，而 LLM 正是防詐宣導、新聞轉傳這些情形的最後一道判斷。

    **誠實揭露：** 176455 的 83,323 筆全部是**已被停止 DNS 解析**的網域，
    使用者點進去多半已經打不開。這不減損訊號價值（訊息仍是詐騙，依據仍成立），
    但命中代表「這是一波已被處理過的詐騙」，不代表「你現在有立即危險」。

    **選配的 `allowlist`（Tranco 排名白名單）只收窄這一個檢查，分三種情形。**
    它不產生 `CheckResult`、不帶權重、不進入信心值計算 —— 它唯一能做的事
    是讓這個檢查少產一筆結果，或把一筆結果的 `hard` 由 `True` 改為 `False`。
    這是三種可能作法裡唯一**不可能讓分數變高**也**不可能讓其他層的證據被
    抵銷**的一種；一個能扣分的白名單會從防守元件變成攻擊面。
    """

    name = "url_blocklist"
    stage = Stage.LOCAL

    def __init__(
        self,
        store: BlocklistStore,
        psl: PublicSuffixList,
        tables: Tables,
        *,
        allowlist: RankAllowlist | None = None,
    ) -> None:
        self._store = store
        self._psl = psl
        self._known_services = tables.known_service_domains
        sources = store.manifest["sources"]
        self._titles = {dataset_id: meta["title"] for dataset_id, meta in sources.items()}
        self._allowlist = allowlist

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results = []
        for domain, urls in group_by_registrable_domain(extract_urls(doc, self._psl)).items():
            hosts = {url.host for url in urls}
            exact = [entry for host in sorted(hosts) for entry in self._store.by_host(host)]
            rank = self._allowlist.rank(domain) if self._allowlist is not None else None
            if exact:
                # 情形三：被通報的主機**就是**該可註冊網域本身 → 黑名單贏。
                # 黑名單是第一方機關對整個網站的直接指控，而 Tranco 排名量的是
                # 流量；一個有流量的詐騙網站仍然是詐騙網站（實測前一百萬名內
                # 含 `e-visacan.com`、`welove777.com`）。把「哪邊贏」交給排名，
                # 等於用流量推翻法律程序。
                #
                # 情形二：命中的主機全部是白名單網域的**子網域** → 降 `hard`。
                # 黑名單的鍵是主機，而平台上攻擊者的單位是路徑（實測獨立釣魚
                # 樣本 41.7% 的 URL 帶非平凡路徑），「這個主機被通報過」不足以
                # 指認任何一個具體頁面，而 `hard=True` 會短路掉整個 LLM 層。
                # 降 `hard` 不刪結果：訊號仍在、依據仍寫、LLM 照跑。
                apex_hit = any(entry.host == domain for entry in exact)
                downgraded = rank is not None and not apex_hit
                results.append(
                    self._result(
                        domain,
                        urls,
                        tuple(exact),
                        hard=not downgraded,
                        suffix=self._allowlist_suffix(domain, rank) if downgraded else "",
                    )
                )
                continue
            if rank is not None:
                # 第一級：可註冊網域在白名單上時不做網域層比對。
                # 與下面的 `known_service_domains` 完全同形，只是把那份人工
                # 清單換成資料驅動的版本 —— 一個共用平台上的鄰居不能證明
                # 這個連結有問題。
                continue
            if domain in self._known_services:
                # 可註冊網域是已知的共用服務時**不做網域層比對**。
                # Cofacts 樣本實測到的具體傷害：165027 收了 `twamiino.pse.is`，
                # 而 `pse.is` 是短網址服務 —— 沒有這條規則，每一個 `pse.is`
                # 連結都會命中（實測 5 次，全部是非詐騙訊息）。
                # 這與 `url_tld_risk` 的「已知服務不觸發」是同一條理由：
                # 一個共用平台上的鄰居不能證明這個連結有問題。
                continue
            neighbours = self._store.by_registrable_domain(domain)
            if neighbours:
                observed = "、".join(sorted(hosts))
                results.append(
                    self._result(
                        domain,
                        urls,
                        neighbours,
                        hard=False,
                        suffix=(
                            f"。訊息中的主機為 {observed}，"
                            f"與被通報的主機不同，同屬可註冊網域 {domain}"
                        ),
                    )
                )
        return results

    def _allowlist_suffix(self, domain: str, rank: int | None) -> str:
        """白名單造成降級時的依據文案。

        寫的是**可查證的事實**（確切名次與清單 ID），不是「知名網站」
        「可信網域」這類形容詞 —— 後者無法被反駁，也無法在半年後複驗。
        """
        list_id = self._allowlist.manifest["list_id"] if self._allowlist is not None else ""
        return (
            f"。可註冊網域 {domain} 為 Tranco 清單 {list_id} 第 {rank} 名，被通報的主機為其子網域"
        )

    def _result(
        self,
        domain: str,
        urls: list[ExtractedUrl],
        entries: tuple[Entry, ...],
        *,
        hard: bool,
        suffix: str,
    ) -> CheckResult:
        scam_types: list[ScamType] = []
        for entry in entries:
            if entry.source not in SOURCE_SCAM_TYPES:
                raise ValueError(
                    f"黑名單紀錄的 source 不在類型對照表中：source={entry.source!r}、"
                    f"host={entry.host!r}，已知的為 {sorted(SOURCE_SCAM_TYPES)}"
                )
            scam_type = SOURCE_SCAM_TYPES[entry.source]
            if scam_type not in scam_types:
                scam_types.append(scam_type)
        return CheckResult(
            name=self.name,
            hit=True,
            detail=self._detail(entries) + suffix,
            evidence=coords_of(urls),
            scam_types=scam_types,
            hard=hard,
        )

    def _detail(self, entries: tuple[Entry, ...]) -> str:
        """依據是事實陳述，不用「可疑」「危險」這類形容詞。

        比對層級與白名單造成的補充說明由呼叫端以 `suffix` 帶入 ——
        兩者是不同的事實（「主機不同但同網域」與「這個網域是大平台」），
        不能由同一個 `hard` 旗標推導出來。
        """
        parts = []
        for entry in entries:
            title = self._titles[entry.source] if entry.source in self._titles else entry.source
            piece = f"{entry.host} 於 {entry.last_seen} 列入 {title}"
            if entry.nature is not None:
                piece += f"，網站性質：{entry.nature}"
            parts.append(piece)
        return "；".join(parts)


class UrlShortenerCheck:
    """短網址 —— 命中的意義是**證據不足**，不是詐騙訊號。

    這個檢查存在的唯一理由是：**沉默不可以被誤讀成清白。**
    一則用 `reurl.cc/2xY3z` 包起來的詐騙連結，在黑名單、TLD、主機結構、
    品牌四個檢查上全部不命中（`reurl.cc` 是合法服務、`.cc` 不在風險清單、
    主機結構正常、品牌無相似）。若系統就這樣安靜下去，`add-confidence` 會把
    「URL 層沒有訊號」當成一個負面證據，而事實是 URL 層**根本沒有機會**產生證據。

    **不展開短網址**，三個理由每一個都是具體的傷害：

    1. 請求送到的是**詐騙者控制的伺服器**。它會學到我們的 IP、這個連結仍在流通、
       以及有人在檢查它。（這與 `add-domain-age` 的 RDAP 完全不同 ——
       RDAP 送到的是**註冊局**，一個與詐騙者無關的中立第三方。）
    2. **一次性連結會被燒掉。** 我們的展開請求可能就是「那一次」，
       使用者之後點進去反而是死連結 —— 結果對了，但過程是我們替使用者點了它。
    3. **同一個短網址可以對不同來源回不同目的地。** 展開得到的答案不保證是
       使用者會看到的答案，而一個不保證正確的答案比沒有答案糟。

    結果粒度為 **URL** 而非網域：`reurl.cc/a` 與 `reurl.cc/b` 是兩個不同的目的地，
    而這個訊號講的正是「目的地未知」。
    """

    name = "url_shortener"
    stage = Stage.LOCAL

    def __init__(self, tables: Tables, psl: PublicSuffixList) -> None:
        self._tables = tables
        self._psl = psl

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results = []
        for url in extract_urls(doc, self._psl):
            if url.registrable_domain not in self._tables.shorteners:
                continue
            service = self._tables.shorteners[url.registrable_domain]
            results.append(
                CheckResult(
                    name=self.name,
                    hit=True,
                    detail=(
                        f"短網址 {url.url}（{service.service}），目的地未知；"
                        f"黑名單、TLD 風險與網域年齡對此連結不具資訊量"
                    ),
                    evidence=list(url.coords),
                    scam_types=[],
                    hard=False,
                )
            )
        return results


class UrlTldRiskCheck:
    """高風險 gTLD —— 統計訊號，而且在台灣特別不準，所以只收 gTLD。

    入選依據是**每萬網域的釣魚比率**而非惡意網域的絕對數量：以絕對數量排，
    榜首是 `.com`，而 `.com` 顯然不是高風險 TLD，它只是很大。

    ccTLD 全部不收，理由見 `tables/tld_risk.json` 的註記 —— 簡言之是
    「全世界的 .tw 網域裡有多少是壞的」與「台灣使用者收到的 .tw 連結裡
    有多少是壞的」不是同一個問題，而後者我們沒有數字。

    **已知服務不觸發。** 可註冊網域出現在短網址表或品牌官方網域中時不產出結果，
    這樣我們不必因為一個合法服務就放棄整個 TLD。
    """

    name = "url_tld_risk"
    stage = Stage.LOCAL

    def __init__(self, tables: Tables, psl: PublicSuffixList) -> None:
        self._tables = tables
        self._psl = psl

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        known = self._tables.known_service_domains
        results = []
        for domain, urls in group_by_registrable_domain(extract_urls(doc, self._psl)).items():
            if domain in known:
                continue
            tld = domain.rsplit(".", 1)[-1]
            if tld not in self._tables.tld_risk:
                continue
            risk = self._tables.tld_risk[tld]
            multiple = risk.value / self._tables.baseline_value
            results.append(
                CheckResult(
                    name=self.name,
                    hit=True,
                    detail=(
                        f"網域 {domain} 的 TLD .{tld} 每萬網域的釣魚分數為 {risk.value}"
                        f"（{risk.source}，{risk.as_of}），約為 .{self._tables.baseline_tld}"
                        f"（{self._tables.baseline_value}）的 {multiple:.0f} 倍"
                    ),
                    evidence=coords_of(urls),
                    scam_types=[ScamType.PHISHING_LINK],
                    hard=False,
                )
            )
        return results


class UrlHostShapeCheck:
    """主機結構異常 —— 四個幾乎免費的高精確度訊號，讀 `ExtractedUrl` 的欄位即可。

    | 訊號 | 為什麼幾乎不會誤判 |
    |---|---|
    | 主機為 IP 位址字面值 | 台灣沒有銀行或物流用 `http://192.0.2.1/` 發通知 |
    | URL 含使用者資訊 | `https://post.gov.tw@evil.com/` 的唯一用途是騙人看錯 |
    | IDNA 編碼失敗 | 真實網站的主機必然編得出來，否則 DNS 解析不了 |
    | 混合文字系統 | `аpple.com` 的 `а` 是西里爾字母 |

    **即使精確度高，`hard` 仍為 False。** 它們高精確但也**罕見**，
    給 `hard` 換來的短路省下的成本極少，而一次誤判要付出的是整個 LLM 層 ——
    而 LLM 正是引述與宣導的最後一道判斷。這個不對稱決定了保守是對的。

    混合文字系統的判定在 `raw_host`（Unicode 形式）上做，不在 `host`
    （punycode）上做 —— `аpple.com` 轉成 `xn--pple-43d.com` 之後那個資訊就沒了。
    """

    name = "url_host_shape"
    stage = Stage.LOCAL

    def __init__(self, psl: PublicSuffixList) -> None:
        self._psl = psl

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results = []
        for url in extract_urls(doc, self._psl):
            for detail in self._details(url):
                results.append(
                    CheckResult(
                        name=self.name,
                        hit=True,
                        detail=detail,
                        evidence=list(url.coords),
                        scam_types=[ScamType.PHISHING_LINK],
                        hard=False,
                    )
                )
        return results

    def _details(self, url: ExtractedUrl) -> list[str]:
        details = []
        if url.is_ip_literal:
            details.append(f"連結 {url.url} 的主機為 IP 位址 {url.host}，不是網域名稱")
        if url.has_userinfo:
            details.append(
                f"連結 {url.url} 的實際主機為 {url.host}，"
                f"`@` 之前的部分位於使用者資訊欄位而不是主機"
            )
        if url.idna_failed:
            details.append(f"連結 {url.url} 的主機 {url.raw_host} 無法以 IDNA 編碼為 punycode")
        mixed = mixed_scripts(url.raw_host)
        if mixed:
            details.append(f"連結 {url.url} 的主機 {url.raw_host} 混用{'與'.join(mixed)}字母")
        return details


class UrlBrandCheck:
    """品牌冒用 —— 三種做得到的判定，加一種做不到的。

    **做不到的那一種：中文品牌名對拉丁網域。** `tw-mail.com` 冒充中華郵政，
    而「中華郵政」與 `tw-mail` 的字面相似度**是零**，沒有任何編輯距離或
    同形異義字判定能把它們連起來。兩條看似可行的路都是死路：

    - **音譯／英譯 alias 表**（`chunghwapost`、`post`、`mail`、`ems`）。
      有用的 alias 太短而不可用 —— 把 `mail` 當中華郵政的 alias，
      `webmail.company.com.tw` 會命中；而夠長到安全的 alias（`chunghwapost`）
      詐騙者根本不會用，他們用的正是 `tw-mail` 這種短而模糊的組合。
      **alias 表抓得到的，是實務上不會出現的那些。**
    - **列舉已知的假網域**。那就是黑名單，而且是一份我們自己維護、必然落後的。

    **可行的那一條是判定三（跨層）：** 訊息**文字**提到某個品牌的中文名，
    而同一則訊息中的 URL 其可註冊網域**不在該品牌的官方清單中**。
    `tw-mail.com` 在這條規則下完全被抓到，而且不需要任何音譯 ——
    訊息一定會寫「中華郵政」（不然收件人不知道在講什麼）。
    這條規則對**每一個中文品牌一體適用**。

    判定三的誤判來源明確且已知：**新聞轉傳與防詐宣導**
    （「蝦皮購物被冒用，詳見 https://news.example.com/xxx」會命中）。
    因此 `hard=False`、權重低，且 `add-quotation-check` 命中時應抑制此訊號。

    混合文字系統不在此檢查重複判定（已在 `url_host_shape`）。
    """

    name = "url_brand"
    stage = Stage.LOCAL

    def __init__(
        self,
        tables: Tables,
        psl: PublicSuffixList,
        *,
        short_label_length: int = DEFAULT_SHORT_LABEL_LENGTH,
        short_label_distance: int = DEFAULT_SHORT_LABEL_DISTANCE,
        long_label_distance: int = DEFAULT_LONG_LABEL_DISTANCE,
        min_label_length: int = DEFAULT_MIN_LABEL_LENGTH_FOR_DISTANCE,
    ) -> None:
        """⚠️ 四個門檻**沒有實驗依據**，是參數不是常數，由 `add-ablation` 掃描。"""
        self._tables = tables
        self._psl = psl
        self._short_label_length = short_label_length
        self._short_label_distance = short_label_distance
        self._long_label_distance = long_label_distance
        self._min_label_length = min_label_length

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        urls = extract_urls(doc, self._psl)
        results = [
            self._result(detail, evidence)
            for detail, evidence in self._host_findings(urls) + self._mention_findings(doc, urls)
        ]
        return results

    def _result(self, detail: str, evidence: list[Coord]) -> CheckResult:
        return CheckResult(
            name=self.name,
            hit=True,
            detail=detail,
            evidence=evidence,
            scam_types=[ScamType.PHISHING_LINK],
            hard=False,
        )

    def _host_findings(self, urls: list[ExtractedUrl]) -> list[tuple[str, list[Coord]]]:
        """判定一（品牌 token 出現在非官方網域）與判定二（編輯距離）。"""
        findings: list[tuple[str, list[Coord]]] = []
        for url in urls:
            if url.registrable_domain is None:
                continue
            for brand in self._tables.brands:
                if url.registrable_domain in brand.official_domains:
                    continue
                token = self._matching_token(url, brand)
                if token is not None:
                    findings.append(
                        (
                            f"連結 {url.url} 的主機含 {brand.chinese_name} 的 {token}，"
                            f"但其可註冊網域為 {url.registrable_domain}，"
                            f"不屬 {brand.chinese_name} 的官方網域"
                            f"（{'、'.join(brand.official_domains)}）",
                            list(url.coords),
                        )
                    )
                    continue
                near = self._nearest_official_label(url.registrable_domain, brand)
                if near is not None:
                    official_label, distance = near
                    findings.append(
                        (
                            f"連結 {url.url} 的可註冊網域 {url.registrable_domain} 其主標籤與 "
                            f"{brand.chinese_name} 的官方主標籤 {official_label} "
                            f"編輯距離為 {distance}（套用視覺替換後）",
                            list(url.coords),
                        )
                    )
        return findings

    def _matching_token(self, url: ExtractedUrl, brand: Brand) -> str | None:
        for domain in brand.official_domains:
            if domain in url.host:
                return f"官方網域字串 {domain}"
        for token in brand.label_tokens:
            if token in url.host:
                return f"主標籤 {token}"
        return None

    def _nearest_official_label(self, domain: str, brand: Brand) -> tuple[str, int] | None:
        label = visual_normalize(domain.split(".", 1)[0])
        best: tuple[str, int] | None = None
        for official in brand.official_domains:
            official_label = official.split(".", 1)[0]
            if len(official_label) < self._min_label_length:
                continue
            distance = levenshtein(label, visual_normalize(official_label))
            if distance == 0:
                # 距離為 0 但可註冊網域不是官方的 —— 視覺替換後與官方主標籤
                # 完全相同，這正是 `esunb0nk.com` 這類的形態，仍要命中。
                return official_label, 0
            threshold = (
                self._short_label_distance
                if len(official_label) <= self._short_label_length
                else self._long_label_distance
            )
            if distance <= threshold and (best is None or distance < best[1]):
                best = (official_label, distance)
        return best

    def _mention_findings(
        self, doc: Document, urls: list[ExtractedUrl]
    ) -> list[tuple[str, list[Coord]]]:
        """判定三：訊息文字提到品牌，但**同一則訊息**中的連結不屬該品牌。

        必須限制在同一則訊息內 —— 跨訊息會把「前文聊到蝦皮、後文貼一個新聞連結」
        判成冒用，所以用 `doc.message_range()` 限制範圍。
        """
        findings: list[tuple[str, list[Coord]]] = []
        for brand in self._tables.brands:
            for message_index in sorted({coord[0] for coord in doc.coords}):
                indexes = doc.message_range(message_index)
                mentions = [
                    doc.coords[i] for i in indexes if brand.chinese_name in doc.sentences[i]
                ]
                if not mentions:
                    continue
                in_message = [
                    url for url in urls if any(coord[0] == message_index for coord in url.coords)
                ]
                if not in_message:
                    continue
                if any(url.registrable_domain in brand.official_domains for url in in_message):
                    continue
                offenders = "、".join(
                    sorted({url.registrable_domain or url.host for url in in_message})
                )
                findings.append(
                    (
                        f"第 {message_index} 則訊息提到 {brand.chinese_name}，"
                        f"但其中的連結屬 {offenders}，"
                        f"不在 {brand.chinese_name} 的官方網域"
                        f"（{'、'.join(brand.official_domains)}）中",
                        mentions + coords_of(in_message),
                    )
                )
        return findings


def mixed_scripts(host: str) -> list[str]:
    """主機的任一標籤內混用拉丁／西里爾／希臘時，回傳混用到的文字名稱。

    判定在**單一標籤內**做 —— 跨標籤混用（`тест.com`）是合法的 IDN 寫法。
    """
    for label in host.split("."):
        present = [
            name
            for name, ranges in SCRIPT_RANGES.items()
            if any(_in_ranges(character, ranges) for character in label)
        ]
        if len(present) >= 2:
            return present
    return []


def _in_ranges(character: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    code = ord(character)
    return any(low <= code <= high for low, high in ranges)


def visual_normalize(text: str) -> str:
    """套用視覺替換表。依據是無襯線字體下這幾組字形幾乎相同，可以截圖驗證。"""
    result = text
    for source, target in VISUAL_MULTI_SUBSTITUTIONS:
        result = result.replace(source, target)
    return "".join(VISUAL_SUBSTITUTIONS.get(character, character) for character in result)


def levenshtein(left: str, right: str) -> int:
    """編輯距離。自行實作而不用 `difflib.SequenceMatcher` ——

    後者算的是最長共同子序列的比率而不是編輯距離，在短字串上的行為不直觀。
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_character in enumerate(left, start=1):
        current = [i]
        for j, right_character in enumerate(right, start=1):
            cost = 0 if left_character == right_character else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def register_url_checks(
    registry: CheckRegistry,
    psl: PublicSuffixList,
    tables: Tables,
    *,
    store: BlocklistStore | None = None,
    allowlist: RankAllowlist | None = None,
) -> None:
    """把 URL 層的檢查註冊進 registry。

    **未提供 `store` 時不註冊 `url_blocklist`** —— 不註冊一個永遠不命中的空檢查，
    否則「沒有訊號」與「沒有資料」在 `Verdict.checks` 裡看起來一模一樣。

    **`allowlist` 未提供時 `url_blocklist` 的行為與白名單落地前完全相同。**
    白名單不是 `Check`，`CheckRegistry.disable()` 關不掉它 —— 把它包成 `Check`
    只為了能被 `disable()`，會逼它產出一個「這個網域很紅」的 `hit=True` 結果
    進入 `Verdict.checks`，而下游會把它誤用為負面證據。
    `add-ablation` 的對照方式因此是以相同資料註冊兩組檢查，
    一組帶白名單、一組不帶；兩次執行完全獨立，不會有殘留狀態。

    白名單只影響 `url_blocklist`，其餘四個檢查不接收它 —— 逐項理由見
    `add-tranco-allowlist` 的 design，簡言之：`url_shortener` 報的是證據不足
    （抑制它會讓信心值把「沒有訊號」誤讀為「乾淨」，而 `bit.ly` 實測排第 118
    名）、`url_brand` 的跨層比對正是抓「寄生在高排名網站上的釣魚」那條規則、
    `url_host_shape` 判的是主機字串本身的形狀、`url_tld_risk` 已有自己的
    「已知服務不觸發」規則且前 1,000 名內高風險 gTLD 實測為 0。

    檢查不接收權重：權重由 `(name, hard)` 於 `weights.toml` 查得
    （`add-weight-table`），檢查本身不需要知道任何數值。
    """
    if store is not None:
        registry.register(UrlBlocklistCheck(store, psl, tables, allowlist=allowlist))
    registry.register(UrlShortenerCheck(tables, psl))
    registry.register(UrlTldRiskCheck(tables, psl))
    registry.register(UrlHostShapeCheck(psl))
    registry.register(UrlBrandCheck(tables, psl))
