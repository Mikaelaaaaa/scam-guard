"""輸入與輸出契約 —— 系統的資料載體，不含任何判斷邏輯。

此模組被所有 check、pipeline 與介面層 import，自身不 import 專案內任何模組。
契約一旦變更即為 BREAKING，須同步修改所有 client。
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Message:
    """單一則訊息。

    `sender` 為自由字串而非布林 —— 轉傳群組對話時可能有三方以上，
    布林表達不了。adapter 自行決定識別粒度（匿名化成 "them"/"me"，
    或保留暱稱）。`None` 表示未知，單則轉傳時 adapter 通常不知道發送者。

    `sent_at` 可為 `None` —— 轉傳單則訊息時 LINE 不提供原始時間戳，
    強制必填會逼 adapter 填入無意義的值。
    """

    text: str
    sender: str | None = None
    sent_at: datetime | None = None


@dataclass(frozen=True)
class Request:
    """一至多則訊息組成的偵測請求。

    最後一則為待判定的訊息（`latest`），其餘為前文脈絡（`context`）。
    多包一層而不直接傳 `list[Message]` 的理由：這個切分邏輯只該有一份。

    `frozen=True` 防止 check 意外修改輸入 —— check 之間的隔離靠這個保證。
    已知缺口：frozen 只保護欄位重新賦值，`messages` 這個 list 的內容仍可被修改。
    """

    messages: list[Message]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("Request 至少需要一則訊息")

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

    `evidence` 為**句子編號**而非文字片段，對應 normalize 後的切句結果。
    存編號則從結構上排除「檢查回傳與原文不符的字串」這個可能 ——
    編號要嘛有效要嘛越界，可驗證。

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
    evidence: list[int] = field(default_factory=list)
    scam_types: list[str] = field(default_factory=list)
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

    `checks` 保留所有執行過的檢查，含未命中者，不做摘要或過濾 ——
    UI 顯示依據時需要，消融實驗逐項分析時需要。過濾的責任在呈現層。
    """

    scam_probability: float | None
    confidence: float
    scam_type: str | None
    evidence: list[str]
    actions: list[str]
    checks: list[CheckResult]
