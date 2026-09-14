"""網域年齡檢查 —— 只定義查詢介面與判斷，網路實作在頂層 `net/`。

**為什麼網域年齡補得到黑名單補不到的洞。** 黑名單是事後的：要先有人受害、
報案、經查證、進入 165 的開放資料，我們才下載得到。而釣魚網域的生命週期
通常比這條鏈短 —— 它在註冊後數天內被大量發送，被通報時往往已經停用。
所以黑名單對**當下正在發送**的那一批幾乎必然是空的，而那一批的共同特徵
恰好是「網域很新」。

**本模組不做 I/O，一行都沒有。** `DomainAgeCheck` 從頭到尾只呼叫
`lookup(domain) -> DomainAge`，不知道有 HTTP、不知道有 RDAP、不知道有快取。
實作在 `net/rdap.py`，由組裝層注入 —— 未注入時 `register_domain_age_check()`
不註冊本檢查，系統照常產出 `Verdict`，且**不產生任何對外連線**。
「預設零外流」因此是真的，開啟本檢查是一個要顯式做的決定。

**三種查詢結果不得互相代替，這是本模組最重要的一條。**
最誘人的 fallback 是「查不到就當作沒有訊號」，而它危險的原因是
「查不到年齡」與「年齡很老」會產生**同一個輸出**：兩者都是空陣列。
於是一則帶著全新釣魚網域、但 RDAP 剛好逾時的訊息，與一則帶著十年老網域的
訊息，在下游看起來一模一樣。三種 outcome 在型別上分開，至少讓
`lookup` 的實作與快取層不會把它們弄混。

⚠️ **已知缺口（`Check` 協定表達不出「問不到」）：** `NO_DATA` 與 `UNAVAILABLE`
依協定都回傳空陣列，`pipeline._run()` 補上 `detail="未命中"` 的記錄 ——
於是它與「查到了，網域五年老」在 `Verdict.checks` 裡完全同形。
`pipeline` 自己就在乎這個區別（`_skipped()` 存在的唯一理由就是區分
「跑了沒訊號」與「根本沒跑」），少的是第三種：「跑了，但問不到」。
修法屬 `check-protocol` 的破壞性變更，不在本 PR。在修好之前，
查詢失敗會使 `add-confidence` 的信心值**偏高**，方向是高估，須記入報告限制。
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Iterable, Protocol

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document
from scam_guard.types import CheckResult, Request, ScamType
from scam_guard.url import PublicSuffixList, extract_urls, utc_today
from scam_guard.url_check import coords_of, group_by_registrable_domain

DEFAULT_THRESHOLD_DAYS = 30
"""⚠️ **這個值沒有實驗依據，它是一個起點不是調過的參數。**

本機唯一能拿到的依據是 165027（數位產業署聲請停止解析清單，1,612 筆），
它同時帶「詐騙網站創建日期」與「接獲通報日期」兩個欄位。實測分布：
中位數 44 天、p10 為 5 天、p25 為 13 天；**41.3%（666/1,612）的網站
在被通報時網域年齡未滿 30 天**（未滿 7 天 16.1%、未滿 90 天 65.0%）。

這個數字是**下界**而非命中率：通報落後於發送，使用者收到訊息的當下，
網域比「被通報時」更年輕。但它也只說得出**召回**那一半 —— 合法網域的
年齡分布我們沒有數字，所以它證明不了 30 天的精確度，更證明不了 30 天
比 14 天或 60 天好。因此本值 MUST 為建構參數，由 `add-ablation` 掃描。
"""


class AgeOutcome(Enum):
    """一次年齡查詢的結果類別。三者 MUST NOT 互相代替。

    三者的差別不只是語意，**處置也不同**：

        KNOWN        這是關於**這個網域**的事實    快取 7 天    不需重試
        NO_DATA      這是關於**這個註冊局**的事實  快取 7 天    重試也一樣
        UNAVAILABLE  這是關於**這次請求**的事實    短快取       應該重試

    `NO_DATA` 被單獨列出來的原因是它真實存在：一部分註冊局的 RDAP 回應不含
    建立事件（有的基於當地個資法規遮蔽，有的只給更新日期），而 IANA 的
    RDAP bootstrap 也只涵蓋 1,200 / 1,438 個 TLD。把它歸進 `UNAVAILABLE`
    會讓系統對那些 TLD 無止境地重試一件不會改變的事。
    """

    KNOWN = "known"
    NO_DATA = "no_data"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DomainAge:
    """一次年齡查詢的結果。`registered_on` 僅在 `KNOWN` 時非 `None`。

    型別在此處就把「未知不得帶日期」釘死：`NO_DATA` 或 `UNAVAILABLE` 卻帶著
    一個日期的物件根本建不出來，於是「查不到就填今天」這種 fallback
    不必靠 code review 擋，它在建構時就炸。
    """

    domain: str
    outcome: AgeOutcome
    registered_on: date | None = None

    def __post_init__(self) -> None:
        if self.outcome is AgeOutcome.KNOWN and self.registered_on is None:
            raise ValueError(
                f"outcome 為 KNOWN 時 registered_on 不得為 None：domain={self.domain!r}"
            )
        if self.outcome is not AgeOutcome.KNOWN and self.registered_on is not None:
            raise ValueError(
                f"outcome 為 {self.outcome.value} 時 registered_on MUST 為 None，"
                f"實為 {self.registered_on!r}：domain={self.domain!r}"
            )


class DomainAgeLookup(Protocol):
    """網域年齡的查詢介面。函式與類別實例皆可滿足，不需繼承任何基底類別。

    實作 MUST 自行捕捉網路例外並轉為 `UNAVAILABLE`，且捕捉 MUST 逐一列舉型別
    —— `DomainAgeCheck` 因此不需要 try/except，它收到的永遠是一個合法的
    `DomainAge`。這是把 `check.py`「依賴外部服務的檢查 MUST 自行捕捉例外」
    再往下推一層：責任落在真正知道會發生哪些失敗的那一層。
    """

    def __call__(self, domain: str) -> DomainAge: ...


class DomainAgeCheck:
    """網域註冊日期距今未滿門檻天數即命中。

    **`hard=False`。** 新網域不是「事實不可能」—— 每天都有大量合法網域被註冊，
    一家新公司的官網在第 6 天也是 6 天。`CheckResult.hard` 的定義是「事實不可能」，
    網域年齡不符合，因此它也不觸發短路。

    **結果粒度是網域，不是 URL。** `evil.com/a`、`evil.com/b` 各產一筆的話，
    UI 上是兩行一模一樣的依據 —— 註冊日期講的是網域這一件事。

    **短網址服務的網域 MUST NOT 查詢。** 對「包裹配送失敗 reurl.cc/2xY3z」
    回報「網域註冊於 2,400 天前」是一句**事實正確但完全誤導**的依據：
    它描述的是短網址服務，不是使用者即將連到的地方，而且它看起來像一個
    「這個連結沒問題」的訊號。清單與 `url_shortener` 共用同一份資料
    （`Tables.shorteners`），不各自維護；那個檢查會另外產出
    「這是短網址，目的地未知」的訊號，所以這裡的沉默不會讓資訊消失。

    **被短路時整個檢查不執行，於是最像詐騙的那一批網域反而不會外流。**
    這是設計不是漏洞：黑名單已命中時 `Stage.EXPENSIVE` 讓本檢查根本不跑，
    短路省下的不只是延遲，是一次對外查詢。

    已知的邊界：註冊日期若晚於今天（註冊局時鐘偏移），天數會是負數而
    `detail` 會寫出負的天數。本檢查不替它改寫成 0 —— 那會把一個資料異常
    變成一句看起來正常的話。
    """

    name = "domain_age"
    stage = Stage.EXPENSIVE

    def __init__(
        self,
        lookup: DomainAgeLookup,
        psl: PublicSuffixList,
        shortener_domains: Iterable[str],
        *,
        threshold_days: int = DEFAULT_THRESHOLD_DAYS,
    ) -> None:
        self._lookup = lookup
        self._psl = psl
        self._shortener_domains = frozenset(shortener_domains)
        self._threshold_days = threshold_days

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        today = utc_today()
        results = []
        groups = group_by_registrable_domain(extract_urls(doc, self._psl))
        for domain, urls in groups.items():
            if domain in self._shortener_domains:
                continue
            age = self._lookup(domain)
            if age.outcome is not AgeOutcome.KNOWN:
                # NO_DATA 與 UNAVAILABLE 皆不產出結果，且**皆不得**被表達成
                # 「網域很新」或「網域夠老」。兩者的差別在 lookup 那一層有意義
                # （要不要重試、快多久），在這一層沒有 —— 見模組 docstring
                # 記下的協定缺口。
                continue
            days = (today - age.registered_on).days
            if days >= self._threshold_days:
                continue
            results.append(
                CheckResult(
                    name=self.name,
                    hit=True,
                    detail=(f"網域 {domain} 註冊於 {days} 天前（{age.registered_on.isoformat()}）"),
                    evidence=coords_of(urls),
                    scam_types=[ScamType.PHISHING_LINK],
                    hard=False,
                )
            )
        return results


def register_domain_age_check(
    registry: CheckRegistry,
    psl: PublicSuffixList,
    shortener_domains: Iterable[str],
    *,
    lookup: DomainAgeLookup | None = None,
    threshold_days: int = DEFAULT_THRESHOLD_DAYS,
) -> None:
    """把網域年齡檢查註冊進 registry。**未提供 `lookup` 時不註冊。**

    與 `register_url_checks` 對 `store` 的處理同一個理由：不註冊一個永遠不會
    有答案的檢查，否則「沒有訊號」與「沒有查詢管道」在 `Verdict.checks` 裡
    看起來一模一樣。

    這裡多一層意義：`lookup` 是本系統唯一的對外查詢入口，**預設不注入**，
    所以預設組裝下 `detect()` 全程不連網。注入它是一個顯式的決定，
    而那個決定的後果 MUST 出現在使用者可見的說明中 ——
    啟用後，訊息中連結的網域會被送到該網域的註冊局。

    檢查不接收權重：權重由 `(name, hard)` 於 `weights.toml` 查得
    （`add-weight-table`），檢查本身不需要知道任何數值。
    """
    if lookup is None:
        return
    registry.register(
        DomainAgeCheck(
            lookup,
            psl,
            shortener_domains,
            threshold_days=threshold_days,
        )
    )
