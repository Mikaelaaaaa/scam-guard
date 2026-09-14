"""輸入與輸出契約 —— 系統的資料載體，不含任何判斷邏輯。

此模組被所有 check、pipeline 與介面層 import，自身不 import 專案內任何模組。
契約一旦變更即為 BREAKING，須同步修改所有 client。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    # 唯一的例外，而且只在型別檢查時存在：`Verdict.redacted` 的型別定義在
    # `scam_guard.redact`，而該模組 import `normalize` 與本模組。執行期沒有這條
    # 邊，import 圖仍是單向的 `redact.py` → `normalize.py` → `types.py`。
    # 把 `RedactedText` 搬進本模組可以完全消掉這條邊，但 `add-redact-apply` 的
    # proposal 明文把它放在 `redact.py`，不在實作階段改。
    from scam_guard.redact import RedactedText

Coord: TypeAlias = tuple[int, int]
"""證據座標 `(原始訊息序號, 訊息內句子序號)`，兩者皆從 0 起算。

**訊息序號**是該訊息在**原始 `Request`** 中的位置，不因上下文截斷而位移 ——
丟掉最舊的 3 則之後，保留下來的第一則其訊息序號仍是 3。這讓 `Document` 與
`Request` 的索引直接對齊：座標為 `(m, s)` 時，`req.messages[m]` 就是該句所屬的
訊息，需要 `sent_at` 的軌跡檢查因此不需要另一張對照表。

**句子序號是訊息內的，不是全域的。** 這是最容易被實作反的地方。若它是全域序號，
第一個分量就成了冗餘資訊，兩個座標退化成一個；而且全域序號會因截斷而整體位移
（原本的第 30 句變成第 12 句），同一則訊息在兩次請求中的編號就不同。
LLM 的 prompt 也是逐則編號的，要它回報全域序號等於要它跨訊息做加法。
"""


class ScamType(Enum):
    """詐騙類型的唯一詞彙來源。訊號層、計分層、LLM 層與介面層皆用此列舉表達類型。

    成員名為英文識別字，供 `weights.yaml` 的 key 與程式碼引用；值為 165 打詐儀錶板
    `CaseTitle` 的中文原文，供 LLM prompt 的可選值與呈現層直接顯示 ——
    因此不需要另一張顯示名稱對照表。

    **納入準則：判定要件必須全部在訊息裡。** 一個 165 類別要成為成員，必須存在
    一段可指認的訊息內容，使某個訊號層能夠命中它、並回報指向該內容的 `evidence`
    座標。準則問的不是「這類好不好判」（那是準確率問題），而是「有沒有東西可以指」——
    一個連指都指不到的類別，`Verdict.evidence` 就只能寫形容詞。

    **排除的五類，各自的判定要件都落在訊息之外**（合計 57,387 件，29.90%）：

    - `網路購物`（49,586）—— 由交易結果界定。165 按**通路**分類，話術不拘；
      同一句「先匯訂金 300 保留」在賣家出貨時是正常交易，不出貨時是詐騙，
      字面完全相同。可偵測的部分已由別的成員承接（帶惡意連結是 `PHISHING_LINK`、
      要求賣家去認證是 `FAKE_BUYER`、冒稱訂單異常是 `ORDER_ANOMALY`）。
    - `假廣告`（2,694）—— 與合法行銷沒有訊息層的界線。「限時五折」「庫存最後三組」
      在真實促銷與詐騙廣告中字面相同，判定要件是商品存不存在。
    - `信用卡遭盜刷`（2,331）—— 受害者事後的報案分類，描述的是填完假刷卡頁**之後**
      的結果。「索取卡號與 CVV」是高精確度訊號，但它命中時該輸出的是承載該索取行為
      的類型（`PHISHING_LINK`，或無連結時的 `FAKE_AUTHORITY` / `FAKE_BUYER`）。
    - `假預付型消費`（2,186）—— 由業者是否履約界定，與「網路購物」是同一種錯誤。
    - `其他`（590）—— 受理端的殘差桶，沒有任何訊號指向它。

    **不設「其他」成員。** 三個理由，任一個都足夠：「類型：其他」的資訊量是零，
    而「未判定出類型」已經有表達方式（`Verdict.scam_type = None`），兩種表達同一件事
    會逼呈現層處理兩個 case；一旦存在，它會成為未知類型的 fallback 著陸點，把大聲的
    失敗變成安靜的錯誤答案；它不是一種話術，不可能有規則寫「命中則為其他」。

    **未知值 MUST NOT 被映射到任何成員。** 本模組不提供任何把未知字串轉為成員的函式。
    `ScamType(value)` 對未知值拋 `ValueError`、`ScamType[name]` 對未知名稱拋 `KeyError`，
    兩者皆為預期行為 —— 規則或設定檔裡出現未知值是**程式錯誤**。LLM 回傳未知類型
    是**模型輸出不合法**，由 `add-llm-validate` 依 fail-closed 處理，同樣不映射為成員。
    """

    FAKE_INVESTMENT = "假投資"
    ROMANCE_INVESTMENT = "假交友(投資詐財)"
    SEXUAL_SERVICE = "色情應召"
    FAKE_BUYER = "假買家騙賣家"
    ROMANCE_MARRIAGE = "假交友(徵婚詐財)"
    PHISHING_LINK = "釣魚簡訊/惡意連結"
    FAKE_LOAN = "假借銀行貸款"
    FAKE_AUTHORITY = "假檢警/假冒公務機關"
    BANK_ACCOUNT_HARVEST = "騙取金融帳戶(卡片)"
    GAME_ITEM = "虛擬遊戲"
    FAKE_PRIZE = "假中獎通知"
    FAKE_JOB = "假求職"
    ACCOUNT_TAKEOVER = "盜用通訊軟體帳號"
    GUESS_WHO = "猜猜我是誰"
    INSTALLMENT_CANCEL = "解除分期付款"
    ORDER_ANOMALY = "假消費異常"
    FAKE_CHARITY = "假慈善機關(急難救助)"
    FAKE_PARCEL = "假借包裹招領"


MERGED_CASE_TITLES: dict[ScamType, tuple[str, ...]] = {
    ScamType.INSTALLMENT_CANCEL: ("解除分期付款(騙買家)", "解除分期付款(騙賣家)"),
    ScamType.ORDER_ANOMALY: ("假消費異常詐騙(騙買家)", "假交易異常詐騙(騙賣家)"),
}
"""因受訊者是買家或賣家而在 165 分裂、但在訊息層無法區分的成員。

收訊者的身分不在訊息裡 ——「您的訂單設定錯誤為分期付款，請至 ATM 操作解除」
送給買家與送給賣家時內容可以完全一樣。留兩個成員等於要求 `add-type-resolve`
判斷一件訊息裡沒有的事。合併的代價是這兩個成員的值不再等於任何單一 `CaseTitle`，
因此對照關係由本模組的 `case_titles()` 提供，而非讓每個消費端各建一張表。
"""


def case_titles(scam_type: ScamType) -> tuple[str, ...]:
    """取得成員涵蓋的 165 `CaseTitle`，供 `add-metrics` 與官方分項統計對齊。

    合併成員查 `MERGED_CASE_TITLES`，其餘成員的 `CaseTitle` 即其值本身。
    """
    if scam_type in MERGED_CASE_TITLES:
        return MERGED_CASE_TITLES[scam_type]
    return (scam_type.value,)


@dataclass(frozen=True)
class Message:
    """單一則訊息。

    `sender` 為自由字串而非布林 —— 轉傳群組對話時可能有三方以上，
    布林表達不了。adapter 自行決定識別粒度（匿名化成 "them"/"me"，
    或保留暱稱）。`None` 表示未知，單則轉傳時 adapter 通常不知道發送者。

    `sent_at` 可為 `None` —— 轉傳單則訊息時 LINE 不提供原始時間戳，
    強制必填會逼 adapter 填入無意義的值。

    `__repr__` **不輸出 `text`**，只輸出長度。`repr()` 是原文進 log 最短的一條
    路徑：一行 `logger.info("收到 %s", message)` 就足夠，而寫那行的人不會意識到
    自己在記錄使用者的 LINE 對話。長度與 `sender` 保留，debug 仍然可用。
    """

    text: str = field(repr=False)
    sender: str | None = None
    sent_at: datetime | None = None

    def __repr__(self) -> str:
        return f"Message(len={len(self.text)}, sender={self.sender!r}, sent_at={self.sent_at!r})"


@dataclass(frozen=True)
class Request:
    """一至多則訊息組成的偵測請求。

    最後一則為待判定的訊息（`latest`），其餘為前文脈絡（`context`）。
    多包一層而不直接傳 `list[Message]` 的理由：這個切分邏輯只該有一份。

    `frozen=True` 防止 check 意外修改輸入 —— check 之間的隔離靠這個保證。
    已知缺口：frozen 只保護欄位重新賦值，`messages` 這個 list 的內容仍可被修改。

    `__repr__` 只輸出則數，理由同 `Message`。
    """

    messages: list[Message] = field(repr=False)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("Request 至少需要一則訊息")

    def __repr__(self) -> str:
        return f"Request(messages={len(self.messages)})"

    @property
    def latest(self) -> Message:
        """待判定的訊息，即陣列中的最後一則。"""
        return self.messages[-1]

    @property
    def context(self) -> list[Message]:
        """`latest` 之外的前文，順序與輸入一致。單則時為空陣列。"""
        return self.messages[:-1]

    @classmethod
    def from_text(cls, text: str) -> "Request":
        """從純文字建構單則請求，供單則轉傳情境使用。"""
        return cls(messages=[Message(text=text)])


@dataclass(frozen=True)
class CheckResult:
    """單一檢查的輸出。一個檢查可產出多筆（例：訊息中的三個 URL 各一筆）。

    `detail` 為自由字串而非結構化欄位 —— 不同檢查的「實際數字」形狀差太多
    （相似度是兩個浮點數、網域年齡是天數、黑名單命中根本沒有數字），
    強行結構化會產生大量 `None` 欄位。形式如 `"0.87/0.85"`、
    `"網域註冊於 6 天前"`、`"命中 165 涉詐網站清單"`。

    `evidence` 為**座標**而非文字片段，對應 normalize 後的切句結果。
    存位置則從結構上排除「檢查回傳與原文不符的字串」這個可能 ——
    座標要嘛有效要嘛越界，可驗證。

    座標為 `(原始訊息序號, 訊息內句子序號)`（見 `Coord`），兩者皆從 0 起算。
    訊息序號**不因上下文截斷而位移**；句子序號是**訊息內**的，不是全域序號。
    原本此欄位是單一整數的句子編號，在「一個 `Document` 對應一則訊息」這個
    前提下足以定位；`add-document-type` 把 `Document` 擴及請求中的全部訊息
    之後，單一整數的兩種讀法都壞掉（全域序號取不到訊息序號且會因截斷整體位移，
    訊息內序號則根本不知道是哪一則），因此改為兩個分量。

    空 `evidence` 是合法值：有些訊號沒有句子位置（訊息則數異常、發送時間
    集中度、整體長度），此時 `hit` 仍可為 True。與「未命中」由 `hit` 區分。
    消費端 MUST 以 `Document.index_of()` 解析座標 —— 越界座標是產生它的檢查
    算錯了，拋例外而非安靜略過。

    `scam_types` 為 `ScamType` 詞彙表的成員，不是自由字串 —— 未列於 `ScamType`
    的類型無法被表達。一個訊號可指向多個成員（某條話術同時屬兩種類型），
    收斂由 `add-type-resolve` 負責。

    **`detail` MUST NOT 含訊息原文片段。** 它只得含數值、名稱與描述
    （`"0.87/0.85"`、`"網域註冊於 6 天前"`、`"命中 165 涉詐網站清單"`）。
    理由是 `detail` 會進 log，而 log 的文字只得來自 `RedactedText`。
    一條寫成 `detail=f"命中關鍵字：{sentence}"` 的檢查會繞過整條遮蔽路徑，
    而且它看起來比不含原文的版本更有用 —— 這正是需要把規定寫在型別旁邊的理由。

    空 `scam_types` 是合法值：有些訊號指示可疑但不指向特定類型（規避偵測命中），
    此時 `hit` 仍可為 True。與「未命中」同樣由 `hit` 區分，不由空陣列表達。

    `hard` 標示此訊號是否為**硬證據**：黑名單命中、Tier-A 規則這類
    「事實不可能」的訊號為 True；弱訊號（Tier-B）為 False。
    判定標準由 `add-confidence` 的 spec 明確定義。此旗標供信心值計算與
    pipeline 的短路判斷使用，不是「有多確定」的分數 —— 檢查作者能可靠
    判斷的粒度是「這是硬證據嗎」，不是自陳信心。
    """

    name: str
    hit: bool
    weight: float
    detail: str
    evidence: list[Coord] = field(default_factory=list)
    scam_types: list[ScamType] = field(default_factory=list)
    hard: bool = False


@dataclass(frozen=True)
class Verdict:
    """系統的最終判定，可直接渲染成使用者看得懂的四行。

    `scam_probability` 為 `None` 代表**信心不足，無法判定** ——
    呼叫端 MUST 顯示「無法判定」而非數字。完全無訊號時計分會算出 0.5，
    但那個 0.5 不傳達任何資訊；用 `None` 讓型別系統強制呼叫端處理此情況，
    而不是把「不要顯示 0.5」這條規則散佈到每個 client。

    `scam_probability` 與 `confidence` 是兩個獨立的量：前者答「是不是詐騙」，
    後者答「有沒有足夠依據下判斷」。

    `scam_type` 為 `ScamType` 詞彙表的成員，不是自由字串。`None` 的語意是
    **未判定出類型** —— 有訊號命中但無法收斂到特定類型時即為此。系統 MUST NOT
    以萬用類型值（「其他」）表達同一件事：詞彙表刻意沒有那個成員，
    兩種表達同一件事只會讓呈現層多處理一個 case。

    `checks` 保留所有執行過的檢查，含未命中者，不做摘要或過濾 ——
    UI 顯示依據時需要，消融實驗逐項分析時需要。過濾的責任在呈現層。

    `redacted` 是**可記錄投影**，由 `detect()` 在全部檢查之後填入。
    它是外層取得該投影的**唯一**途徑 —— 不另外回傳、不放全域、不讓呼叫端
    重新正規化一次。選欄位而不選 `(Verdict, RedactedText)` 的理由是失敗模式的
    比較：欄位被忽略的後果是「沒寫 log」，缺資料是吵的；tuple 被
    `verdict, _ = detect(...)` 丟掉的後果是「呼叫端改去 log `req`」，
    而 `req` 是原文 —— 安靜、看起來正常、沒有任何東西會報告。**選會吵的那一個。**

    **`redacted` 不是呈現用的資料。** HTTP 回應與使用者介面 MUST NOT 顯示它，
    呈現一律取 `Document.raw_at()` 的原文片段。

    `__repr__` 不輸出 `evidence` 與 `checks` 的內容，也不輸出 `redacted` ——
    `evidence` 是原文片段，而 `redacted` 雖然可安全記錄，但要記錄它應該是
    一個顯式的動作，不該是 `repr(verdict)` 的副作用。
    """

    scam_probability: float | None
    confidence: float
    scam_type: ScamType | None
    evidence: list[str] = field(repr=False)
    actions: list[str]
    checks: list[CheckResult] = field(repr=False)
    redacted: "RedactedText" = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"Verdict(scam_probability={self.scam_probability!r}, "
            f"confidence={self.confidence!r}, scam_type={self.scam_type!r}, "
            f"evidence={len(self.evidence)}, checks={len(self.checks)})"
        )
