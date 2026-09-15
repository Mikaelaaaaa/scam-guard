"""模型輸出的四個欄位，以及強制它們的 GBNF grammar。

**本模組的工作不只是命名四個欄位，它決定了整個 LLM 層的 runtime。**

`transformers` 的 `generate()` 沒有 constrained decoding，能做的只有
「生成 → `json.loads()` → 驗證 → 失敗就重試」，而失敗率是模型的性質，
我們控制不了也量不準。llama.cpp 的 GBNF grammar 在**取樣之前**把所有不符合
文法的 token 的 logit 設為 `-inf`，也就是說產生一個不符合 grammar 的 token
在解碼層面是**不可能**的。剩下的結構失敗只有三種，而且每一種都是
**我們自己的參數**造成的，不是模型的機率行為：

| 剩餘的結構失敗 | 成因 |
|---|---|
| 生成長度用盡 | `max_tokens` 到達時 grammar 尚未到接受狀態 |
| context 溢出 | prompt + 輸出超過 `n_ctx` |
| 逾時中止 | `llm_runtime` 的 deadline 生效 |

**這是整個決定的全部價值：把一個無法觀測的機率換成三個可設定的參數。**

⚠️ **grammar 保證形狀，保證不了語意。** 模型若「想」輸出別的東西，它會被迫
輸出一個**合法但任意**的值 —— `evidence_sentence_ids` 指向不存在的句子、
`category_165` 與規則層矛盾。這個交換把失敗模式從**吵**（`json.loads` 拋例外）
換成**安靜**（一個完全合法的錯誤答案），而接下那個代價的是
`scam_guard.llm.validate` 的語意驗證清單。

**本模組只產生一段字串。** 它 MUST NOT import `llama_cpp` 或任何 runtime ——
grammar 的消費者在頂層的 `llm_runtime/`，見 `scam_guard.llm.check`。
"""

from dataclasses import dataclass
from string import Template

from scam_guard.types import Coord, ScamType

FIELD_NAMES: tuple[str, ...] = (
    "analysis_notes",
    "evidence_sentence_ids",
    "category_165",
    "label",
)
"""輸出物件的四個鍵，**順序即推理順序**，且由 grammar 強制。

自回歸模型先產生的 token 是後產生 token 的條件。把 `label` 放最後，代表結論
以說明與證據為條件；把 `label` 放最前，代表說明與證據以結論為條件 —— 後者就是
「先下結論再補理由」，而補出來的理由對結論沒有任何約束力。

這也是 prompt 組裝層取用欄位名的**唯一**來源。手寫的欄位名與本常數不同步時
沒有任何機制會報告，模型會照著錯的名字輸出，然後被 grammar 擋下，整次判讀作廢。
"""

LABELS: tuple[str, ...] = ("無詐騙話術", "部分詐騙話術", "完整詐騙話術")
"""`label` 的值域，**順序即序關係**（遞增），MUST NOT 以字串比較推導。

`"完整詐騙話術" < "部分詐騙話術"` 在 Python 裡是按 Unicode 碼位比，而那個結果
與此處要的序無關。序關係的唯一來源是本序列的索引，見 `label_rank()`。

**三個字串是為了通過 `add-verdict-render` 的兩張禁用詞表而選的**
（`scam_guard.render` 的 `SPECULATIVE_TERMS` 與 `VERDICT_CLAIMS`）——
`label` 的字面值會經由 `CheckResult.detail` 出現在使用者看得到的地方。
表一的第一個詞就是「可疑」，表二含「是詐騙」「為詐騙」「詐騙訊息」「確定是」。
「無 / 部分 / 完整詐騙話術」是關於**訊息內容**的可陳述事實（這段文字裡有沒有
那些話術），不是關於結論的宣稱。**改這三個字串要同時檢查那兩張表。**

為什麼是三級：二級會逼模型在「有點怪但說不上來」時二選一，而那個中間地帶正是
`add-type-resolve` 說的「LLM 補位」要補的地方；四級以上沒有依據 ——
我們沒有任何數字可以區分「很像」與「非常像」。
"""

MAX_EVIDENCE_IDS = 5
"""`evidence_sentence_ids` 的數量上界，由 grammar 強制。

**這與 `add-verdict-render` 的 `max_evidence_lines` 不是同一個參數，同值是巧合，
MUST NOT 共用常數。** 前者是模型輸出的上限（控制 decode 成本與定位品質），
後者是呈現的上限（控制使用者讀不讀得完）。把巧合寫成共用常數之後，
改一個會意外改到另一個。

下界由 grammar 寫 0：`label` 為 `LABELS[0]` 時陣列為空是合法的，而
「label 非最低級時 MUST 非空」是跨欄位條件，上下文無關文法表達不了，
由 `scam_guard.llm.validate` 補上。
"""

MAX_NOTES_CHARS = 200
"""`analysis_notes` 的字元上界，由 grammar 的字元類重複上界強制。

**無實驗依據。** 太短會讓說明沒有資訊，太長會讓 decode 成本翻倍
（200 字元約 150 token，估算）；`add-ablation` 掃描它，並 MUST 把
「有 / 無 `analysis_notes` 欄位」列為一組對照 —— 若無差異，拿掉這個欄位
可以省下每次判讀約一半的 decode 時間。

以 grammar 的重複上界表達而不是事後截斷：事後截斷發生在 decode 成本
已經付掉之後。
"""

NOTES_CHAR_CLASS = r'[^"\\\n]'
"""`analysis_notes` 的字元類，同時排除雙引號、反斜線與換行。

因此 `analysis_notes` **不可能含逸出序列** —— JSON 解析不需要處理逸出，
而「模型在說明欄位裡塞一段假的 JSON 結構」這條路徑也消失了。
"""

GRAMMAR_TEMPLATE = Template(
    r"""root     ::= "{" notes "," ids "," category "," label "}"
notes    ::= "\"analysis_notes\":\"" $notes_class{0,$max_notes} "\""
ids      ::= "\"evidence_sentence_ids\":[" idlist "]"
idlist   ::= ( pair ( "," pair ){0,$max_extra_ids} )?
pair     ::= "[" int "," int "]"
int      ::= [0-9] | [1-9] [0-9] [0-9]?
category ::= "\"category_165\":" catval
catval   ::= $catval
label    ::= "\"label\":" labval
labval   ::= $labval
"""
)
"""grammar 的樣板。常數的部分寫在這裡，兩個值域由 `build_grammar()` 插入。

**不用 llama-cpp-python 的 `json_schema_to_gbnf()`。** 三個理由：轉換結果不可讀
（18 個中文值的 enum 會展開成一長串無法在 review 時逐項核對的替代式，而 review
是本專案對這類表的主要品管手段）；轉換器的版本升級會**安靜地**改變 grammar，
而那是一個沒有任何測試會報告的行為變更；以及我們需要的約束有一部分
JSON Schema 表達得不好（鍵的固定順序、字串的字元類上界）。

以 `string.Template` 而非 `str.format()` 展開：grammar 裡的 `{0,200}` 是
GBNF 的重複語法，`str.format()` 會把它當成欄位。

⚠️ `{m,n}` 是較新的 llama.cpp 才支援的語法。不支援時 MUST 改為顯式展開
（`( pair ( "," pair )? ? ? ? )?`），**MUST NOT 改成無上界的 `*`** ——
後者拿掉了上界，模型可以輸出 200 個座標，而那是一次 decode 成本的爆炸。
（以 `llama_cpp.LlamaGrammar.from_string()` 實測 0.3.35 支援 `{m,n}`，
見 `tests/test_llm_grammar.py`。）
"""


@dataclass(frozen=True)
class LlmOutput:
    """模型輸出**解析並驗證之後**的形狀。

    這不是模型直接吐出的東西 —— 它是 `scam_guard.llm.validate` 的產物，
    到得了這個型別就代表七條語意條件都過了。

    `analysis_notes` 是模型輸出中唯一的自由文字，而它 **MUST NOT 進入
    `Verdict` 的任何欄位、MUST NOT 進入 `CheckResult.detail`**：
    `add-verdict-render` 定「每一行依據去掉原文片段之後 MUST 等於某個
    `hit=True` 結果的 `detail`」，模型的自由文字不是 `detail`；而放進 `detail`
    等於給模型一條直通呈現層與示範面板的自由文字通道。
    它由一個**注入的記錄器**接收，未注入時丟棄（見 `scam_guard.llm.check`）。
    """

    analysis_notes: str
    evidence_sentence_ids: tuple[Coord, ...]
    category_165: ScamType | None
    label: str


def label_rank(label: str) -> int:
    """`label` 在 `LABELS` 中的索引，即它的強度。

    未知值讓 `ValueError` 傳播 —— 不回傳 -1、不回傳 `None`。一個把未知 label
    當成「最低級」的預設值，會讓一個壞掉的輸出安靜地變成「模型說沒問題」。
    """
    return LABELS.index(label)


def _json_string_branch(value: str) -> str:
    """把一個值包成 GBNF 的字串字面，**兩層引號**。

    第一層是 GBNF 的：終端符號寫在雙引號內，內容為 UTF-8 字面。
    `(`、`)`、`|`、`*`、`+`、`?`、`{`、`}` 這些**只在引號外**是運算子，
    寫在引號內就是普通字元 —— 所以「騙取金融帳戶(卡片)」的括號、
    「假檢警/假冒公務機關」的斜線**都不需要逸出**。GBNF 需要逸出的只有
    `"` 與 `\\`，而 18 個值一個都不含。

    第二層是 JSON 的：這些值出現在 JSON 裡，兩側要有 JSON 的雙引號，
    寫成 GBNF 就是把那兩個引號逸出。結果是六個字元的前後綴：

        "\\"假檢警/假冒公務機關\\""
        ↑ ↑↑                    ↑↑ ↑
        │ ││                    ││ └─ GBNF 的結束引號
        │ ││                    │└─── JSON 的結束引號（逸出）
        │ │└─ 值本身            └──── 逸出用的反斜線
        │ └── JSON 的起始引號（逸出）
        └──── GBNF 的起始引號

    寫錯的方式有兩種，兩種都看起來很像對的：少一層 → grammar 允許模型輸出
    沒有引號的裸中文（JSON 不合法）；多一層 → 模型被迫輸出 `"\\"假投資\\""`，
    `json.loads` 會得到一個帶引號的字串，`ScamType()` 隨即拋 `ValueError`。
    純字串比對擋不住這類錯誤，所以 `tests/test_llm_grammar.py` 的測試是
    **端到端**的：grammar 解析 + `json.loads` + `ScamType()`。
    """
    return f'"\\"{value}\\""'


def build_grammar() -> str:
    """產生本次輸出的 GBNF grammar。

    **grammar 不落檔。** 落檔就有兩個真相來源（`ScamType` 與檔案），而
    `scam_guard/tables/` 放的是**人工維護的對照表**；grammar 有結構、要與程式碼
    一起改，且它的一部分由 `ScamType` 產生，屬邏輯側。

    `catval` 的 18 個分支 MUST 由 `ScamType` 產生，MUST NOT 手抄 —— 手抄的失敗
    模式是 Enum 新增一個成員而 grammar 沒跟上，結果是那個類型**永遠不可能被輸出**，
    而且沒有任何地方會報告。`labval` 同理由 `LABELS` 產生。

    允許 `null`：模型可能看出話術但說不出是哪一種，而那與 `Verdict.scam_type =
    None` 是同一個狀態。允許 `null` 比允許一個萬用值（「其他」）好，好在一個具體
    的地方 —— `null` 在 JSON 裡是一個**型別**不是一個值，不可能被 `ScamType(value)`
    接受，因此不可能成為未知類型的著陸點。
    """
    catval = " | ".join(
        ['"null"'] + [_json_string_branch(scam_type.value) for scam_type in ScamType]
    )
    labval = " | ".join(_json_string_branch(label) for label in LABELS)
    return GRAMMAR_TEMPLATE.substitute(
        notes_class=NOTES_CHAR_CLASS,
        max_notes=MAX_NOTES_CHARS,
        max_extra_ids=MAX_EVIDENCE_IDS - 1,
        catval=catval,
        labval=labval,
    )
