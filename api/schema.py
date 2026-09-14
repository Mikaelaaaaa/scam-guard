"""`POST /check` 的請求、回應與錯誤契約 —— 只有形狀，不含任何 HTTP 概念。

**兩套欄位名。** 線上的鍵是 `text` / `from` / `at`，取自 `openspec/project.md`
的輸入契約；模型屬性是 `text` / `sender` / `sent_at`，與 `scam_guard.types.Message`
一致。兩者不同的原因是單向的：`from` 是 Python 保留字，寫不進 dataclass，
`at` 作為屬性名短到讀不出語意。也就是說 `types.py` 改名是為了遷就語言，
而 JSON 沒有這個限制 —— 反過來改 `project.md` 等於為了一個 Python 語法問題
去動一份已發布的對外契約。對映以別名完成，線上的鍵名不因此改變。

這個決定只有配上 `extra="forbid"` 才安全：一個看過 `types.py` 而沒看過
`project.md` 的呼叫端會送 `sender`，若允許未知欄位，pydantic 會安靜地丟掉它、
`sender` 取 `None`，請求合法、回應正常，而軌跡檢查永遠讀不到發送者。

**錯誤訊息可以回音哪些值，是一張窮舉的表，不是一條判斷** —— schema 只有四個
欄位，所以列得完。本模組只定義 `ErrorDetail` 的形狀，實際組字串的是
`api/errors.py`，但規則寫在這裡，因為規則屬於契約：

| 違規 | 訊息裡可以出現 |
|---|---|
| `messages` 為空陣列 | 陣列長度（數字） |
| `text` 缺失或型別錯 | 欄位路徑與**期望的型別**；絕不含收到的值 |
| `from` 為空字串或型別錯 | 欄位路徑與「空字串」這個事實；不含值本身 |
| `at` 無時區或格式錯 | 欄位路徑與收到的時間字串 |
| 未知欄位 | **欄位名**；不含它的值 |

`from` 不回音，是因為 `Message.sender` 的 docstring 允許「保留暱稱」——
它可能是一個真實人名；而本層對它定義的違規只有「空字串」與「型別不是字串」，
兩者都能在不引用值的情況下描述完整。`at` 回音，是因為要解釋「這個時間沒有
時區」，唯一有用的訊息就是把那個字串放出來，而時間戳不是訊息內容。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scam_guard.types import ScamType, Verdict

KNOWN_VERDICT_FIELDS: frozenset[str] = frozenset(
    {
        "scam_probability",
        "confidence",
        "scam_type",
        "evidence",
        "actions",
        "checks",
        "redacted",
    }
)
"""本模組看過並各自做過決定的 `Verdict` 欄位名。

回應是**顯式模型**而不是 `Verdict` 的通用序列化，所以 `Verdict` 加欄位時
本層的輸出不會跟著改變 —— 那是設計目的，不是疏漏。代價是沒有任何機制會提醒
「有一個新欄位還沒有人決定它進不進回應」。這個集合加上一條測試就是那個提醒：
不符即失敗，訊息指出新增或消失的欄位名。它不強迫你把新欄位放進回應，
只強迫你看過它一眼。

這條規則要防的事已經發生過一次：`add-redact-apply` 為 `Verdict` 加上的
`redacted` 正是**唯一一個明文禁止進入 HTTP 回應的欄位**，一個直接序列化的
API 會在那個 change 併入的當天把遮蔽投影送給每一個呼叫端，而且沒有任何測試
會變紅。
"""

EVIDENCE_DESCRIPTION = (
    "判定依據，逐行。**此欄位可能包含請求中訊息的原文片段** —— "
    "規則層的依據多半以一段引自原文的句子作為可查證性的來源，剝掉它會讓那些依據"
    "變成要求使用者相信我們。呼叫端 MUST NOT 把此欄位寫入任何 log。"
    "順序與內容逐字等於系統內部的判定依據，不經改寫、過濾或排序。"
)
"""寫在 OpenAPI description 而不是只寫在 design：design 不會出現在 `/docs`，
而 `/docs` 是寫 client 的人唯一會讀的東西。"""

CONFIDENCE_DESCRIPTION = (
    "系統對「有沒有足夠依據下判斷」的信心。**這不是詐騙機率**，也不是校準過的量"
    "——「分數落在 0.8–0.9 的訊息中有 87% 真的是詐騙」說的是 scam_probability "
    "那一側的事。它**是分級而非連續量**，實際輸出只有少數幾個離散值，"
    "因此 MUST NOT 以百分比或進度條呈現：畫成進度條會讓兩個相鄰的值看起來"
    "只差一點點，而它們對應的是兩個不同的證據狀態。"
)

SCAM_PROBABILITY_DESCRIPTION = (
    "詐騙可能性。為 `null` 時代表系統拒答（依據不足，無法判定），"
    "此時 `abstained` 恆為 `true`。**`null` MUST NOT 被讀成「不是詐騙」** —— "
    "請改讀 `abstained`，不要對此欄位做數值比較。"
)

ABSTAINED_DESCRIPTION = (
    "本次是否為拒答。恆等於 `scam_probability is null`。"
    "存在的理由是 JSON 沒有型別系統：`null > 0.5` 在 JavaScript 裡靜靜地是 "
    "`false`，於是「系統沒把握」被讀成「不是詐騙」。拒答是一次成功的推論，"
    "不是錯誤，因此仍以 HTTP 200 回應。"
)


class MessageIn(BaseModel):
    """請求中的單一則訊息。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        description=(
            "訊息文字。必填，但空字串是合法值 —— 貼圖、純圖片或只有空白的訊息"
            "並不是壞掉的請求，偵測核心對它有明確定義的處置。"
        )
    )
    sender: str | None = Field(
        default=None,
        alias="from",
        description=(
            "發送者識別字。缺席與 `null` 同義，皆表示未知。空字串 MUST NOT 被送出"
            "——它不是未知，也不是識別字，接受它會讓所有這種訊息被歸成同一個發送者。"
        ),
    )
    sent_at: datetime | None = Field(
        default=None,
        alias="at",
        description=(
            "發送時間。缺席與 `null` 同義，皆表示未知。提供時 MUST 帶時區："
            "不帶時區的值會在軌跡檢查做時間相減時於一個真實請求的半途拋 "
            "`TypeError`，在邊界拒絕比在那裡失敗好。"
        ),
    )

    @field_validator("sender")
    @classmethod
    def _reject_empty_sender(cls, value: str | None) -> str | None:
        """空字串不是未知。訊息陳述事實，不引用值本身 —— 它可能是真實人名。"""
        if value == "":
            raise ValueError("from 為空字串；未知的發送者請省略此欄位或送 null")
        return value

    @field_validator("sent_at")
    @classmethod
    def _require_timezone(cls, value: datetime | None) -> datetime | None:
        """不帶時區的時間戳原樣拋出，不補任何預設時區。

        訊息引用 `isoformat()` 而不是呼叫端送出的原始字串：驗證器拿到的已經是
        解析後的 `datetime`，原始字串在這個時間點不存在。對 ISO 8601 的輸入
        兩者逐字相同，而時間戳本身不是訊息內容，回音它不洩漏任何原文。
        """
        if value is not None and value.tzinfo is None:
            raise ValueError(f"at 不帶時區：{value.isoformat()}；請附上時區位移或 Z")
        return value


class CheckRequest(BaseModel):
    """`POST /check` 的請求體。"""

    model_config = ConfigDict(extra="forbid")

    messages: list[MessageIn] = Field(
        description=(
            "待判定的對話，最後一則為待判定的訊息，其餘為前文。至少一則。"
            "**本層不對則數或單則長度設任何上限** —— 輸入規模的上限由偵測核心的"
            "既有雙上限處理，請求體的位元組上限由 HTTP 服務層處理。"
        )
    )

    @field_validator("messages")
    @classmethod
    def _reject_empty(cls, value: list[MessageIn]) -> list[MessageIn]:
        """`Request.__post_init__` 已定「至少需要一則訊息」，在邊界重述一次。"""
        if not value:
            raise ValueError(f"messages 至少需要一則訊息，收到的陣列長度為 {len(value)}")
        return value


class ScamTypeOut(BaseModel):
    """詐騙類型的線上表示 —— 穩定識別字與顯示名稱各一個。

    兩個值各防一個具體的失敗模式。只送 `label`：那是 165 打詐儀錶板的
    `CaseTitle` 中文原文，165 改寫某個案類的名稱時我們會跟著更新，而每一個拿
    中文字串做分支的呼叫端會安靜地走進 `else`。只送 `code`：呼叫端要顯示
    「假檢警/假冒公務機關」就得自建一張成員名到中文的對照表，而 `ScamType`
    明說成員值就是顯示用的，因此不需要那張表。

    兩者不是冗餘，是兩個用途（分支用 `code`、顯示用 `label`），
    而它們的穩定性保證不同。
    """

    code: str = Field(description="`ScamType` 的成員名，穩定識別字，分支請用這個。")
    label: str = Field(description="中文顯示名稱，取自 165 打詐儀錶板的案類名稱。")


class CheckResponse(BaseModel):
    """一次判定的回應。六個欄位，逐一列出，**不由 `Verdict` 序列化而來**。

    不含的東西各有理由：證據座標送了也解不對（呼叫端沒有 `Document`，
    重建切句規則的任一條不一致就會指到別的句子，而且不拋例外）；
    `Verdict.checks` 的唯一消費者（UI 展開、消融實驗）都在同程序內，
    不經過 HTTP；`Verdict.redacted` 是可記錄投影，明文禁止進入回應。

    不送的附帶好處是回應成為一個可以逐條列舉的封閉集合：六個欄位，
    其中只有 `evidence` 可能帶原文。這是一個讀 schema 就能做完的稽核。
    """

    scam_probability: float | None = Field(description=SCAM_PROBABILITY_DESCRIPTION)
    abstained: bool = Field(description=ABSTAINED_DESCRIPTION)
    confidence: float = Field(description=CONFIDENCE_DESCRIPTION)
    scam_type: ScamTypeOut | None = Field(
        description=(
            "詐騙類型。未判定出類型時整個欄位為 `null`，"
            "**不是**一個 `code` 與 `label` 皆為 `null` 的物件 ——"
            "「未判定出類型」是一個狀態，不是兩個。"
        )
    )
    evidence: list[str] = Field(description=EVIDENCE_DESCRIPTION)
    actions: list[str] = Field(
        description=(
            "建議採取的行動，逐行。逐字等於系統內部的建議，空陣列時不補任何預設動作"
            "——對一則「明天見」提出行動建議是製造焦慮。"
        )
    )

    @model_validator(mode="after")
    def _abstained_tracks_probability(self) -> "CheckResponse":
        """兩個欄位表達一件事，就必須在建構時綁死。

        形狀取自 `Document` 的 `truncated == (dropped_messages > 0)`：
        日後若出現別的拒答形式，要改的是 spec，不能靜默地改。
        """
        if self.abstained != (self.scam_probability is None):
            raise ValueError(
                f"abstained 與 scam_probability 脫鉤："
                f"abstained={self.abstained!r}、scam_probability={self.scam_probability!r}"
            )
        return self


class ErrorDetail(BaseModel):
    """錯誤的單一形狀。`field` 為 `null` 代表此錯誤不屬於任何欄位
    （JSON 語法錯誤、請求體過大）—— 在這裡 `null` 只有一個意思。"""

    code: str = Field(description="錯誤代碼，供呼叫端分支。")
    field: str | None = Field(
        description="出錯的欄位路徑，如 `messages.0.at`。不屬於任何欄位時為 `null`。"
    )
    message: str = Field(
        description=(
            "人可讀的說明。**MUST NOT 含訊息文字或發送者識別字的值** —— "
            "框架預設的驗證錯誤結構會把違規值原樣放進回應，對 `text` 欄位"
            "那等於把整則訊息寫進錯誤回應。"
        )
    )


class ErrorResponse(BaseModel):
    """錯誤回應的外層。"""

    error: ErrorDetail


def scam_type_out(scam_type: ScamType) -> ScamTypeOut:
    """把 `ScamType` 成員投影成線上表示。

    `code` 恆為某個現存成員的名稱 —— 回應是單向輸出，本模組不提供反方向的
    映射，`ScamType` 的 docstring 已定「未知值 MUST NOT 被映射到任何成員」。
    """
    return ScamTypeOut(code=scam_type.name, label=scam_type.value)


def verdict_to_response(verdict: Verdict) -> CheckResponse:
    """逐欄位映射，`checks` 與 `redacted` 一律不映射。

    `evidence` 與 `actions` 逐字複製 —— 不改寫、不截斷、不過濾、不排序、
    不合併、不新增。尾隨的原文片段照樣通過：那段原文是呼叫端自己在幾百毫秒前
    送進來的 body 的一部分，回應把其中一句還回去，新接收方數為零。

    **這條推論有一個失效條件。** 前提是「呼叫端與提交者之間沒有新的接收方」。
    若出現把判定結果送給提交者以外的人的呼叫端，要改的是本層，
    不是在那個 adapter 裡補一層遮蔽。
    """
    return CheckResponse(
        scam_probability=verdict.scam_probability,
        abstained=verdict.scam_probability is None,
        confidence=verdict.confidence,
        scam_type=None if verdict.scam_type is None else scam_type_out(verdict.scam_type),
        evidence=list(verdict.evidence),
        actions=list(verdict.actions),
    )
