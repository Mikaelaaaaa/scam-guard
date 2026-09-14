"""遮蔽的套用 —— 產生系統中**唯一**可以寫進 log 的那份文字。

系統共有三份文字，各有唯一的消費者。三份不是四份：

| 文字 | 來源 | 唯一消費者 | 若用錯會怎樣 |
|------|------|-----------|-------------|
| 正規化後 | `Document.sentences` | 規則層、URL 層、**LLM** | —— |
| 原文 | `Document.raw_sentences` | 呈現給提交者 | 規則層讀它會漏掉全形寫法 |
| 遮蔽後 | `RedactedText.sentences` | **只有 log** | 規則層或 LLM 讀它會失去訊號，而且沒換到隱私 |

**送 LLM 前不遮蔽，理由是同程序而不是本地。** 界線不在「本地 vs 雲端」，
在**同程序 vs 不同服務**：Gradio 與 Gemma 3 1B 在 HuggingFace Spaces 的同一個
Python 程序裡，使用者的原文與模型的 forward pass 在同一塊 heap。在呼叫模型前
算出一份遮蔽版，原文並不會消失，能讀到遮蔽版的每一個主體本來就讀得到原文 ——
沒有任何接收方因此看不到東西。此規定的失效條件是**跨服務邊界**：
訊息內容若要離開本程序送往另一個服務（外部 API，或同機的另一個 container），
這個決定 MUST 重新評估。

**留下的那個遮蔽點是寫 log 前，它的理由與訊息密度無關而與時間有關：**
一則訊息在程序裡活幾秒，一行 log 活幾個月，會被當時不在場的人 grep、
複製進 issue、轉寄到監控系統。實測 300 則詐騙樣本裡的那 1 則是身分證。

**偵測核心不寫 log**（`scam_guard/` 不做 I/O）。本模組只產出「可記錄投影」，
由 `api/` 與 adapter 負責實際寫入。本模組 MUST NOT import `logging`。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from scam_guard import pii
from scam_guard.normalize import SENTENCE_END, Document
from scam_guard.types import Coord

PLACEHOLDERS: Mapping[str, str] = MappingProxyType(
    {
        pii.TW_ID: "<TW_ID>",
        pii.TW_MOBILE: "<TW_MOBILE>",
        pii.TW_LANDLINE: "<TW_LANDLINE>",
        pii.CREDIT_CARD: "<CREDIT_CARD>",
    }
)
"""每個類型的遮蔽標記。**非空**且**不含切句分隔符**，兩者都有實質作用。

非空：若允許替換成空字串，一個內容只有一個電話號碼的句子遮蔽後會變成空字串，
而空句在 `add-split-sentences` 已被明確排除（「空白片段略過，不產生空句」）——
這裡會從後門造出一個，而且是只存在於遮蔽版的空句。

不含 `。!?;` 與換行：擋掉「改成 `(已遮蔽；共 10 碼)` 比較好懂」這類後續修改。
那個 `;` 會讓遮蔽後的文字若被任何人重新切句就多出一句，而「被任何人重新切句」
在只剩 log 的世界裡正是最可能發生的事 —— 讀 log 的人手上沒有 `Document`。

帶類型：讀 log 的人需要知道那句話在要什麼，`<TW_ID>` 與 `<TW_MOBILE>` 的
差別就是「這則在騙身分證」與「這則在騙電話」的差別。
"""

_FORBIDDEN_IN_PLACEHOLDER = SENTENCE_END + "\n"


def _validate_placeholders() -> None:
    """於模組載入時檢查標記表，不符立刻拋 `ValueError`。

    在載入時而不是在呼叫時檢查：一個含分隔符的標記是**設定錯誤**，
    它應該讓程式起不來，而不是在第某次請求時才產生一份壞掉的 log。
    """
    for entity_type, placeholder in PLACEHOLDERS.items():
        if not placeholder:
            raise ValueError(f"遮蔽標記必須非空：entity_type={entity_type}")
        for character in _FORBIDDEN_IN_PLACEHOLDER:
            if character in placeholder:
                raise ValueError(
                    f"遮蔽標記不得含切句分隔符：entity_type={entity_type}、"
                    f"placeholder={placeholder!r}、character={character!r}"
                )


_validate_placeholders()


@dataclass(frozen=True)
class RedactedText:
    """可記錄投影 —— 遮蔽後的句子、每句的座標，以及依類型分類的遮蔽計數。

    **它不是 `Document`。** 沒有 `raw_sentences`、沒有 `truncated`，
    也不能被當作檢查的輸入傳入。型別不同這件事本身就是規定的一部分。

    **它是唯一被允許寫進 log 的東西。** 「什麼可以寫 log」因此是一個型別問題
    而不是一句叮嚀：外層拿得到這個型別的實例，就等於拿到許可。

    **它自帶 `coords`，因為讀 log 的人沒有 `Document`。** log 行裡會有
    `CheckResult.evidence` 的座標（`(3, 1)`），而 `Document` 在請求結束時就
    消失了。若投影只有句子沒有座標，那些座標就是一串無意義的數字對，
    而唯一「自然」的補救是拿遮蔽後的文字重新切一次句 —— 那會造出第二份切句
    結果，正是本模組要消滅的東西，而且 log reader 不可能複製
    `split_sentences()` 的細節（網址內部不切、空白片段略過不產生空句）。

    ⚠️ **它的保證範圍是四類辨識器，不等於「不含個資」。** 封閉集合未涵蓋的
    類型（姓名、地址、銀行帳號、護照號碼）仍然留在其中。
    MUST NOT 把它描述成已去識別化的資料。

    `RedactedText` 是承載文字的型別裡**唯一**不隱藏 `repr` 內容的一個 ——
    `Message`、`Document`、`NormalizedText`、`Verdict`、`Request` 都不印文字，
    這裡印，因為它的內容依定義已經遮過。這個例外正是它存在的意義：
    要有一個東西可以安全地被印出來，否則 debug 時沒有任何可用的輸出。

    序列以 `tuple`、計數以 `MappingProxyType` 儲存，理由同 `Document`：
    `frozen=True` 只擋欄位重新賦值，擋不住 list 與 dict 的就地修改。
    """

    sentences: Sequence[str]
    coords: Sequence[Coord]
    counts: Mapping[str, int]
    _index: dict[Coord, int] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentences", tuple(self.sentences))
        object.__setattr__(self, "coords", tuple(self.coords))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

        if len(self.sentences) != len(self.coords):
            raise ValueError(
                f"遮蔽後句子與座標必須等長：len(sentences)={len(self.sentences)}、"
                f"len(coords)={len(self.coords)}"
            )
        for position, sentence in enumerate(self.sentences):
            if not sentence:
                raise ValueError(f"遮蔽後句子不可為空字串：索引 {position}")
        for entity_type, count in self.counts.items():
            if count < 0:
                raise ValueError(f"遮蔽計數不可為負：entity_type={entity_type}、count={count}")

        object.__setattr__(self, "_index", {coord: i for i, coord in enumerate(self.coords)})

    def index_of(self, coord: Coord) -> int:
        """座標對應的扁平索引。無效座標拋 `KeyError`，理由同 `Document.index_of()`。

        例外訊息只印座標，不印任何句子內容 —— 例外訊息會進 log。
        """
        if coord not in self._index:
            raise KeyError(f"座標不存在於此 RedactedText：{coord}")
        return self._index[coord]

    def text_at(self, coord: Coord) -> str:
        """座標對應的遮蔽後句子。這是讀 log 的人解析座標的唯一途徑。"""
        return self.sentences[self.index_of(coord)]


def redact_document(doc: Document) -> RedactedText:
    """把一份 `Document` 投影成可記錄的遮蔽後文字。**沒有其他參數。**

    **沒有 `redactor` 參數、沒有 `NullRedactor`、沒有 `Redactor` 協定。**
    「關掉遮蔽看看偵測結果差多少」這個實驗的答案恆等於零，而且是結構上的零：
    遮蔽在全部檢查之後執行、LLM 讀未遮蔽文字，所以它對 `Verdict` 的任何欄位
    都沒有影響。沒有實驗理由可以關它，於是它不該有開關。
    若硬要留一個參數，它的預設值會是唯一正確的行為 —— 一個總是安全的預設
    沒有失敗模式；真正的危險是 `NullRedactor` 本身，它是一個語法合法、
    可以被貼進 production 設定、而且看起來像是有人想過的值。
    把它刪掉，「忘記開」與「故意關」同時變成不可表達。

    **實作是 `doc.sentences` 的逐元素字串映射。** 不呼叫 `normalize_text()`、
    不呼叫 `split_sentences()`、不看換行與標點。於是
    `len(redacted.sentences) == len(doc.sentences)` 不是一個需要驗證的性質，
    它是迴圈結構的必然結果 —— 句數在結構上不可能改變。
    （`__post_init__` 仍然驗證長度，那是為了防止日後有人改成別的實作方式。）

    區間**由後往前**替換，如此不必在每次替換後重算後續區間的偏移量。
    `find_pii()` 回傳的區間依 `start` 排序且互不重疊，反向走訪即為由後往前。

    `Document` 一個欄位都不改：它是 frozen 的，三個序列以 `tuple` 儲存，
    就地改在技術上就辦不到。既有的證據座標因此仍然有效。

    空 `Document` 回傳空的 `RedactedText`，計數全為零，不拋例外。
    """
    sentences: list[str] = []
    counts = dict.fromkeys(pii.ENTITY_TYPES, 0)
    for sentence in doc.sentences:
        redacted = sentence
        for span in reversed(pii.find_pii(sentence)):
            counts[span.entity_type] += 1
            placeholder = PLACEHOLDERS[span.entity_type]
            redacted = redacted[: span.start] + placeholder + redacted[span.end :]
        sentences.append(redacted)
    # 計數涵蓋四個類型的全部鍵，未命中者為 0 —— 讀取端必須能區分「這類遮了零次」
    # 與「這類不存在」，而缺鍵表達不了前者。
    return RedactedText(sentences=sentences, coords=doc.coords, counts=counts)
