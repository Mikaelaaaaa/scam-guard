"""個資辨識 —— 一個**封閉集合**的四個項目，僅使用標準庫。

這一層唯一的失敗模式是 **over-redaction**（誤遮），不是漏遮。理由不對稱：
漏遮一則身分證的後果是 300 則裡有 1 則的個資進了 log；誤遮一個數字串的後果是
**每一則**含數字的訊息都被削弱。實測 600 則 Cofacts 訊息中，`\\d{10,16}` 這條
沒有結構條件的樣式命中 8–12%，其中多數是誤判（`訂單802045734652`、
`統編93538651`、`寄件碼 E73474605986`、`gclid=Cj0KCQjwkt_UB...`）。

因此本模組的設計原則不是「盡量找出個資」，而是
**「每一項辨識都必須有一個結構性的驗證條件，沒有驗證條件的項目就不辨識」**。
checksum、Luhn、分隔符、IIN 前綴是結構性條件；「長度在 10 到 16 之間」不是。

**辨識項目為封閉集合。新增項目 MUST 先修改 `openspec` 的 `pii-recognizers`
規格，MUST NOT 只增加一條樣式。** 明確不辨識的項目與理由：

| 項目 | 不辨識的理由 |
|------|-------------|
| 銀行帳號 | 沒有通用 checksum；訊息裡的帳號多屬詐騙方，是證據不是待保護的個資 |
| 訂單編號、寄件碼 | 無固定結構，唯一可用的樣式是長度，而長度正是誤判的來源 |
| 統一編號 | 8 位數確有 checksum，但統編是公司的公開資訊，不是個資 |
| 驗證碼 | 見下段 |
| 中文姓名 | `黃金投資` → `黃金`、`王牌業務` → `王牌` 是詐騙語料的高頻詞 |

**驗證碼不需要例外規則。** 四個項目沒有任何一項會命中典型的驗證碼：
6 碼 `123456`、4 碼 `8848`、8 碼 `12345678` 皆不符 —— `TW_ID` 需字母前綴、
`TW_MOBILE` 需 `09` 開頭且十位、`TW_LANDLINE` 需分隔符、`CREDIT_CARD` 需 16 位。
這是四個結構性條件的副產品，不是一張例外清單，所以刻意**不加**
「前 N 字出現『驗證碼』就抑制」這種需要調整視窗參數的上下文規則。

辨識的輸入是**正規化後**的句子（`Document.sentences`），不是原文：NFKC 已把
全形數字收斂為半形，一條 ASCII 樣式就吃得到；在原文上辨識則要為每條樣式寫
全形、半形與混用三個版本，而混用的組合數是指數的。

辨識器**只回報區間，不改寫文字** —— 改寫屬 `scam_guard.redact`。
"""

import operator
import re
from dataclasses import dataclass

from scam_guard.normalize import URL_PATTERN

TW_ID = "TW_ID"
TW_MOBILE = "TW_MOBILE"
TW_LANDLINE = "TW_LANDLINE"
CREDIT_CARD = "CREDIT_CARD"

ENTITY_TYPES: tuple[str, ...] = (TW_ID, TW_MOBILE, TW_LANDLINE, CREDIT_CARD)
"""封閉集合的全部類型。消費端以此列舉建立涵蓋全部類型的計數表。"""


# 身分證字母的兩位數對映。**以常數表達而非以 `ord(ch) - 55` 之類的計算式**：
# I、O、W、X、Y、Z 六個字母不按字母順序（I=34 而非 18、O=35 而非 24、
# W=32、X=30、Y=31、Z=33），任何計算式都得為這六個加例外，而例外寫錯的症狀是
# 少數幾個開頭字母的號碼安靜地算出錯誤的 checksum —— 在低密度語料上看起來
# 與「本來就沒有身分證」一模一樣。
_ID_LETTER_VALUES: dict[str, int] = {
    "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15, "G": 16, "H": 17,
    "I": 34, "J": 18, "K": 19, "L": 20, "M": 21, "N": 22, "O": 35, "P": 23,
    "Q": 24, "R": 25, "S": 26, "T": 27, "U": 28, "V": 29, "W": 32, "X": 30,
    "Y": 31, "Z": 33,
}  # fmt: skip

# 身分證加權和的權重，對應 `n1, n2, d1 … d9`。末兩位同為 1 是規格如此，不是筆誤。
_ID_WEIGHTS: tuple[int, ...] = (1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1)

# 已知發卡機構的 IIN 前綴（Visa 4、MasterCard 51–55、Amex 34/37、JCB 35）。
_CREDIT_IIN_PREFIXES: tuple[str, ...] = ("4", "51", "52", "53", "54", "55", "34", "37", "35")

# 裸數字信用卡號的長度。設計上只收 16 位 —— 13/15/19 位的裸數字與單號在字元層
# 無法區分，而分組書寫的卡號走另一條路徑，不受此長度限制。
_CREDIT_BARE_LENGTH = 16

# 市話的總位數（含區碼）。
_LANDLINE_MIN_DIGITS = 9
_LANDLINE_MAX_DIGITS = 10

# `[A-Z][12]` 加八位數字。前後的邊界條件不在規格裡，是本模組加的：嵌在更長的
# 英數字串中間的十位數（`E73474605986` 這類寄件碼的子串）比較可能是單號而非
# 身分證，而誤遮的代價大於漏遮。
_TW_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z][12][0-9]{8}(?![0-9])")

# `09` 加八位數字，接受 `-` 或單一空白分組。台灣手機號碼沒有 checksum，這是
# 事實，不能假裝有 —— **唯一的驗證條件是數字邊界**，前後不得緊接其他數字。
# 這是四項中最弱的一項，也因此是 over-redaction 最可能的來源；
# `redact.py` 按類型分開計數正是為了讓「是不是手機這項在亂遮」可從計數看出。
_TW_MOBILE_PATTERN = re.compile(r"(?<![0-9])09[0-9]{2}[- ]?[0-9]{3}[- ]?[0-9]{3}(?![0-9])")

# `0[2-8]` 開頭的市話，**必須**以括號包住區碼或在區碼後出現分隔符。
# 裸數字市話（`0227208889`）刻意不辨識：它在字元層與一個十位數的單號無法區分，
# 而它正是 `\\d{10,16}` 那條誤判樣式的子集。放棄裸數字的漏抓量遠小於它的誤判量。
_TW_LANDLINE_PATTERN = re.compile(
    r"(?<![0-9])(?:\(0[2-8][0-9]?\)|0[2-8][0-9]?[- ])[0-9]{3,4}[- ]?[0-9]{3,4}(?![0-9])"
)

# 分組書寫的卡號。分組本身就是結構性條件 —— 訂單編號不會寫成 4-4-4-4。
_CREDIT_GROUPED_PATTERN = re.compile(
    r"(?<![0-9])[0-9]{4}[- ][0-9]{4}[- ][0-9]{4}[- ][0-9]{4}(?![0-9])"
)

# 裸數字卡號。Luhn 通過的機率是 1/10，對任何長度都一樣，所以 Luhn 是必要條件
# 而非充分條件；IIN 前綴與長度 16 是補上的那個結構性條件。
_CREDIT_BARE_PATTERN = re.compile(r"(?<![0-9])[0-9]{16}(?![0-9])")

_DIGITS_PATTERN = re.compile(r"[0-9]")


@dataclass(frozen=True)
class PiiSpan:
    """一個個資命中的位置與類型。

    `start` / `end` 為**該句 `Document.sentences[i]` 內**的半開區間索引，
    不是跨句的、也不是原文（`raw_sentences`）的索引。
    """

    start: int
    end: int
    entity_type: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"區間起點不可為負：start={self.start}")
        if self.end < 0:
            raise ValueError(f"區間終點不可為負：end={self.end}")
        if self.start >= self.end:
            raise ValueError(f"區間必須非空：start={self.start}、end={self.end}")


def _is_valid_tw_id(candidate: str) -> bool:
    """中華民國國民身分證統一編號的校驗碼運算。

    字母映到兩位數 `N`，拆成 `n1 = N // 10` 與 `n2 = N % 10`，
    加權和 `n1*1 + n2*9 + d1*8 + … + d8*1 + d9*1` 能被 10 整除才有效。

    `A123456789` → A=10，和為 `1+0+8+14+18+20+20+18+14+8+9 = 130`，有效。
    """
    letter_value = _ID_LETTER_VALUES[candidate[0]]
    numbers = (letter_value // 10, letter_value % 10, *(int(ch) for ch in candidate[1:]))
    total = sum(number * weight for number, weight in zip(numbers, _ID_WEIGHTS, strict=True))
    return total % 10 == 0


def _luhn_ok(digits: str) -> bool:
    """Luhn 校驗。自右起第二位開始每隔一位加倍，超過 9 則減 9，總和整除 10。"""
    total = 0
    for position, ch in enumerate(reversed(digits)):
        value = int(ch)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _digits_of(text: str) -> str:
    return "".join(_DIGITS_PATTERN.findall(text))


def _find_tw_id(sentence: str) -> list[PiiSpan]:
    return [
        PiiSpan(start=m.start(), end=m.end(), entity_type=TW_ID)
        for m in _TW_ID_PATTERN.finditer(sentence)
        if _is_valid_tw_id(m.group())
    ]


def _find_tw_mobile(sentence: str) -> list[PiiSpan]:
    """手機的全部條件都在樣式裡（`09` 開頭、十位、數字邊界），沒有額外校驗。"""
    return [
        PiiSpan(start=m.start(), end=m.end(), entity_type=TW_MOBILE)
        for m in _TW_MOBILE_PATTERN.finditer(sentence)
    ]


def _find_tw_landline(sentence: str) -> list[PiiSpan]:
    return [
        PiiSpan(start=m.start(), end=m.end(), entity_type=TW_LANDLINE)
        for m in _TW_LANDLINE_PATTERN.finditer(sentence)
        if _LANDLINE_MIN_DIGITS <= len(_digits_of(m.group())) <= _LANDLINE_MAX_DIGITS
    ]


def _find_credit_card(sentence: str) -> list[PiiSpan]:
    """Luhn 為必要條件，分組書寫或已知 IIN 前綴擇一為充分條件。"""
    spans = [
        PiiSpan(start=m.start(), end=m.end(), entity_type=CREDIT_CARD)
        for m in _CREDIT_GROUPED_PATTERN.finditer(sentence)
        if _luhn_ok(_digits_of(m.group()))
    ]
    spans.extend(
        PiiSpan(start=m.start(), end=m.end(), entity_type=CREDIT_CARD)
        for m in _CREDIT_BARE_PATTERN.finditer(sentence)
        if len(m.group()) == _CREDIT_BARE_LENGTH
        and m.group().startswith(_CREDIT_IIN_PREFIXES)
        and _luhn_ok(m.group())
    )
    return spans


_RECOGNIZERS = (_find_tw_id, _find_tw_mobile, _find_tw_landline, _find_credit_card)


def _span_length(span: PiiSpan) -> int:
    return span.end - span.start


def _url_spans(sentence: str) -> list[tuple[int, int]]:
    """不辨識區間。重用切句所用的同一個 `URL_PATTERN`，兩層的判定必須一致。"""
    return [(m.start(), m.end()) for m in URL_PATTERN.finditer(sentence)]


def _starts_inside_url(position: int, url_spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in url_spans)


def _merge_overlaps(spans: list[PiiSpan]) -> list[PiiSpan]:
    """把重疊的命中合併為一個區間，類型取**較長**的那個匹配。

    較長的匹配帶有較多結構條件（`0912-345-678` 的分隔符比裸數字多一層證據），
    所以長度是這裡唯一需要的優先順序，不需要引入信心分數 —— 四個項目就有
    信心分數會逼下游寫門檻，而門檻沒有資料可調。
    """
    merged: list[PiiSpan] = []
    for span in sorted(spans, key=operator.attrgetter("start")):
        if merged and span.start < merged[-1].end:
            previous = merged[-1]
            longer = previous if _span_length(previous) >= _span_length(span) else span
            merged[-1] = PiiSpan(
                start=previous.start,
                end=max(previous.end, span.end),
                entity_type=longer.entity_type,
            )
        else:
            merged.append(span)
    return merged


def find_pii(sentence: str) -> list[PiiSpan]:
    """回傳該句中全部個資的區間，依 `start` 排序且互不重疊。

    輸入為**一個正規化後的句子**（`Document.sentences[i]`），回報的索引相對於
    該句。未命中時回傳空陣列 —— 空陣列是合法且常見的結果，四個項目的覆蓋率
    本來就低，那是設計選擇不是 bug。

    兩條篩選規則：

    1. **起始位置落在網址區間內的匹配一律丟棄。** 這單獨消滅了實測四個誤判中的
       `gclid=Cj0KCQjwkt_UB...`；更重要的是遮蔽若切進網址中段，`add-url-check`
       誤用遮蔽後文字時會抽到半截 domain，黑名單比對直接失效。
    2. **重疊的命中合併為一個，類型取較長的匹配**（見 `_merge_overlaps`）。

    辨識項目為封閉集合，新增項目 MUST 先修改規格。
    """
    url_spans = _url_spans(sentence)
    candidates: list[PiiSpan] = []
    for recognizer in _RECOGNIZERS:
        candidates.extend(
            span for span in recognizer(sentence) if not _starts_inside_url(span.start, url_spans)
        )
    return _merge_overlaps(candidates)
