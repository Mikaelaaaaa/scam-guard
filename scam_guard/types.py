"""輸入契約 —— 系統的資料載體，不含任何判斷邏輯。

此模組被所有 check、pipeline 與介面層 import，自身不 import 專案內任何模組。
契約一旦變更即為 BREAKING，須同步修改所有 client。
"""

from dataclasses import dataclass
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
