"""規避偵測 —— 五個訊號，回答「送這則訊息的人是否試圖讓偵測失效」。

其他檢查問「這則訊息在說什麼」；這一個不讀語意，輸入是字元，輸出是
「這段文字被動過手腳」。一則正常的商業簡訊不會把「匯款」寫成「匯 款」，
不會在詞中插零寬空格，不會把「台幣」寫成「ㄊㄞˊ幣」。
這些痕跡與訊息內容無關，卻是關於**發訊者意圖**的直接證據。

**資料來源逐訊號指定，不是一律讀 `raw`**，因為正規化對不同手法做了不同的事：

| 訊號 | 讀 | 理由 |
|---|---|---|
| `evasion_invisible` | `raw_at()` | 字元已被 `normalize_text()` 移除，`text` 裡不存在 |
| `evasion_width_mix` | `raw_at()` | NFKC 已把全形收成半形，混用痕跡在 `text` 裡消失 |
| `evasion_split_word` | `text_at()` | 要做的是**詞比對**，見下 |
| `evasion_homophone` | `text_at()` | 同上 |
| `evasion_zhuyin` | `text_at()` | 同上 |

後三個讀 `text_at()` 的理由是同一個：它們要做詞比對，而 `raw` 裡混著全形數字、
零寬字元與未統一的空白，每一個都會讓比對失敗。同時用了全形空格與零寬空格的
`匯　​款` 在 `raw` 裡是三個不同的分隔字元，讀 `raw` 就得自己再做一次空白統一 ——
那正是 `normalize_text()` 已經做完的事。

**一個反直覺的結果要講明白**：零寬字元被移除，所以 `evasion_split_word` 在 `text`
上**看不到**「用零寬字元拆字」——`匯​款` 在 `text` 裡就是 `匯款`，完全命中原規則。
這不是漏洞，是分工：那個情形由 `evasion_invisible` 從 `raw` 抓到，而言語行為規則
同時仍然正確命中。**兩個訊號各抓一半，合起來沒有缺口。**

**tg-spam 的門檻不能搬，理由可以量化。** `isAbnormalSpacing()` 用兩個比率：
`spaceRatio = spaces / totalChars > 0.3` 與
`shortWordRatio = shortWords / len(words) > 0.7`（短詞定義為 ≤ 3 字），
兩者都建立在 `strings.Fields`（空白分詞）上。中文不以空白分詞：正常中文訊息的
`spaces` 接近 0，`spaceRatio` 恆遠低於 0.3；而 `strings.Fields` 會把整句中文當成
**一個** word，長度遠大於 3，`shortWordRatio` 恆為 0。
**兩個統計量在中文語料上都退化成常數，判別力為零** —— 不是門檻太高或太低，
是它們在中文上沒有分布。本模組因此改用目標詞導向的判定。

**不做大小寫混用偵測**（tg-spam 有）：一則典型的台灣詐騙訊息含的英文可能只有
`ATM`、`LINE`、`APP` 與一個網址，樣本量太小，而正常訊息裡的品牌名稱大小寫本來
就很亂（`Line`、`line`、`LINE`）。看過、評估過、不做，不是漏掉。

**規避偵測不做還原。** 把 `匯 款` 還原成 `匯款` 再讓規則比對，會讓規則層的輸入
變成兩種（`check.py`：座標系必須有唯一的產生者），而且任何夠激進到能抓到規避的
還原規則，都會把正常的斷句黏成詐騙關鍵詞。代價是拆字訊息會少命中一條 Tier-A
（漏報方向），但多命中一條規避訊號 —— 換來的是分數，不是錯誤的類型。
"""

import operator
import re
import unicodedata
from dataclasses import dataclass

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import INVISIBLE, URL_PATTERN, Document
from scam_guard.rules.speech_act import SELF_DIRECTED, TIER_A_RULES
from scam_guard.types import CheckResult, Coord, Request

INVISIBLE_NAMES: dict[str, str] = {
    "\u200b": "零寬空格",
    "\u200c": "零寬非連字",
    "\u200d": "零寬連字",
    "\u2060": "字連接符",
    "\ufeff": "位元組順序標記",
    "\u00ad": "軟連字號",
    "\u200e": "左至右標記",
    "\u200f": "右至左標記",
    "\u202a": "左至右嵌入",
    "\u202b": "右至左嵌入",
    "\u202c": "方向格式還原",
    "\u202d": "左至右覆寫",
    "\u202e": "右至左覆寫",
    "\u2066": "左至右隔離",
    "\u2067": "右至左隔離",
    "\u2068": "第一強字元隔離",
    "\u2069": "方向隔離還原",
}
"""`normalize.INVISIBLE` 每個字元的中文名稱，供 `detail` 使用。

必須**涵蓋** `INVISIBLE` 的全部字元，由測試斷言 —— 缺項會在執行期拋 `KeyError`，
而那是在使用者的請求上炸。`INVISIBLE` 擴充時這張表要同步擴充。
"""

ZWJ = "\u200d"
BOM = "\ufeff"
SOFT_HYPHEN = "\u00ad"

FULLWIDTH_DIGITS = frozenset(chr(code) for code in range(0xFF10, 0xFF1A))
"""全形數字 ０-９（U+FF10–U+FF19）。"""

FULLWIDTH_LATIN = frozenset(chr(code) for code in range(0xFF21, 0xFF5B))
"""全形拉丁字母 Ａ-ｚ（U+FF21–U+FF5A）。"""

SPLIT_SEPARATORS = frozenset(" \t*_-.~·•/\\+|^#=")
"""拆字時可能被插入的分隔字元。

刻意**不含頓號與逗號**。spec 的上界是「空白、標點、星號與底線」，本模組取其子集：
頓號與逗號是**列舉分隔**，而「請勿提供帳號、密碼」是正常的列舉 ——
把它們算成分隔字元，會讓每一則列舉兩個以上項目的正常訊息命中拆字規避。
`normalize.py` 的 `PUNCT` 註解已經為同一件事做過同一個判斷。
"""

MAX_GAP = 2
"""目標詞相鄰兩字之間容許的分隔字元數上限。

「匯 款」是 1 個，「匯..款」是 2 個。上限開到 3 以上會讓「匯了一筆款」
（中間 3 個字）誤命中。2 是在「拆字者插入的分隔符通常只有 1 個」
與「留一點餘裕」之間取的。
"""

HOMOPHONE_VARIANTS: tuple[tuple[str, str], ...] = ()
"""同音／形近變體對照表（變體 → 原詞）。**初版為空，這是誠實的狀態。**

spec 明訂「變體表 MUST 只收有語料佐證的項目；無佐證時 MUST 留空，
MUST NOT 以推測填充」。`dev-data`（Cofacts 抓取與過濾）與本 PR 是**平行**的兩個
PR，實作本 change 時樣本還不存在 —— 編一張沒有樣本根據的變體表是製造假資料，
而假資料會一路流進 `add-metrics` 的報告。空表的規則永不命中，
`add-testset` 之後再補。

**`賬` 相關的項目 MUST NOT 收入。**「賬號」是中國大陸的標準寫法，不是規避。
Cofacts 語料含大量從中國來源轉傳的文字，把「賬號」判為規避會在那一整類訊息上
系統性誤判。

不用 `pypinyin` 之類的拼音函式庫：`scam_guard` 完全依靠標準庫，為一個 Tier-B
訊號引入第一個執行期依賴不划算；而且一個音節在中文對應數十個常用字，
「以拼音相同判定為變體」的 precision 天生就低，不是門檻可以救的。
"""

ZHUYIN = frozenset(chr(code) for code in range(0x3105, 0x3130)) | frozenset("ˇˊˋ˙")
"""注音符號 ㄅ-ㄦ（U+3105–U+312F）與聲調符號 ˇ ´ ` ˙。"""


def _target_words() -> frozenset[str]:
    """拆字偵測的目標詞表 —— **取自 Tier-A 規則自己的詞表，不另建一份**。

    兩份清單會各自漂移，而漂移不會有任何機制報告（`add-scam-type` 建立共用詞彙表
    的理由相同）。Tier-A 新增一個客體詞時，它自動成為拆字偵測的目標。

    排除 `SELF_DIRECTED`（我、本人、客服、回傳…）：那是通用的接收者標記，
    不是詐騙特有的詞彙，把它們當成拆字目標只會製造誤命中。
    排除單字詞：一個字的詞沒有「相鄰兩字」，不可能被拆。
    """
    words: set[str] = set()
    for rule in TIER_A_RULES:
        words |= (rule.predicates | rule.objects | rule.receivers) - SELF_DIRECTED
    return frozenset(word for word in words if len(word) >= 2)


TARGET_WORDS = _target_words()

SEPARATOR_CLASS = f"[{re.escape(''.join(sorted(SPLIT_SEPARATORS)))}]{{0,{MAX_GAP}}}"

SPLIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (word, re.compile(SEPARATOR_CLASS.join(re.escape(char) for char in word)))
    for word in sorted(TARGET_WORDS)
)
"""每個目標詞一個樣式：`c₁ s₁ c₂ s₂ … cₙ`，每個 `sᵢ` 為 0 至 `MAX_GAP` 個分隔字元。

「至少有一個 `sᵢ` 非空」不寫進樣式，而是以比對結果的長度判定
（`len(match) > len(word)`）—— 否則原詞本身就會命中，
而原詞應該由 `add-speech-act-rules` 處理，不是規避訊號。
"""


def _is_wordish(char: str) -> bool:
    """是否為漢字、數字或拉丁字母 —— 也就是「不是 emoji」。"""
    return char.isalnum()


def _is_han(char: str) -> bool:
    return "一" <= char <= "鿿"


def _message_indices(doc: Document) -> list[int]:
    return sorted({coord[0] for coord in doc.coords})


def _peak(counts: list[tuple[Coord, int]]) -> Coord:
    """命中數最多的句子座標。相同時取最前面的一句，結果穩定。"""
    return max(counts, key=operator.itemgetter(1))[0]


def invisible_marks(raw: str, at_message_start: bool) -> list[str]:
    """原文片段中應計數的不可見字元，依出現順序。

    三個排除項，每一個都對應一種會被誤判的正常訊息：

    1. **U+200D 位於兩個 emoji 之間**。`normalize.INVISIBLE` 含零寬連字，
       而那是 emoji ZWJ 序列的連接符：家庭 emoji 是
       `U+1F468 U+200D U+1F469 U+200D U+1F467`，職業 emoji、旗幟與膚色變體
       也都用它。以「≥1 即命中」為門檻，每一則附了家庭 emoji 的正常訊息都會誤命中。
    2. **U+FEFF 位於訊息開頭**。那是 BOM，來自複製貼上或編碼轉換。
    3. **U+00AD（軟連字號）一律不計數**。部分 Windows 來源的文字會帶它，
       而它在中文訊息裡不構成有效的規避（中文不斷字）。

    排除之後門檻是「≥1 個即命中」，因為剩下的字元（零寬空格、零寬非連字、
    雙向覆寫與隔離）**沒有任何正常的中文輸入途徑會產生它們**。

    附帶記錄一個既有行為：`normalize_text()` 移除 U+200D 會把家庭 emoji 拆成三個
    獨立 emoji。那是 `add-normalize-text` 的行為，本 change **不修改它** ——
    規則不看 emoji、LLM 讀到拆開的 emoji 語意不變、呈現層用 `raw_sentences`，
    實際影響為零。本 change 的責任只是不要因為它而誤判。
    """
    marks: list[str] = []
    for index, char in enumerate(raw):
        if char not in INVISIBLE or char == SOFT_HYPHEN:
            continue
        if char == BOM and at_message_start and index == 0:
            continue
        if char == ZWJ:
            before = raw[index - 1] if index > 0 else ""
            after = raw[index + 1] if index + 1 < len(raw) else ""
            if before and after and not _is_wordish(before) and not _is_wordish(after):
                continue
        marks.append(char)
    return marks


def _blank_span(match: re.Match[str]) -> str:
    """以等長的空白取代網址，使 token 邊界成立而位置不位移。"""
    return " " * (match.end() - match.start())


def _tokens(raw: str) -> list[str]:
    """連續的非空白非標點字元為一個 token，網址範圍內的字元先被抹去。

    中文沒有天然的 token 邊界，這個定義是初版，需要實測（見 design 的
    Open Questions）。網址以 `normalize.URL_PATTERN` 判定而非另建一套規則 ——
    `https://example.com/2026` 裡的半形數字是網址的一部分，不是排版混用。
    """
    masked = URL_PATTERN.sub(_blank_span, raw)
    tokens: list[str] = []
    current: list[str] = []
    for char in masked:
        if char.isspace() or unicodedata.category(char).startswith("P"):
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def mixed_width_tokens(raw: str) -> list[str]:
    """同一 token 內同時出現同類字元的全形與半形形式者。

    **必須是同一個 token**。中文排版本來就會混用 ——
    「請於１０月３１日前至 https://example.com/2026 完成」同時有全形與半形數字，
    而它完全正常：兩者分屬不同 token，半形那些還在網址裡。
    跨 token 就命中的話，這個訊號會在正常排版上大量誤命中。
    """
    mixed: list[str] = []
    for token in _tokens(raw):
        digits = any(char in FULLWIDTH_DIGITS for char in token) and any(
            char.isdigit() and char not in FULLWIDTH_DIGITS for char in token
        )
        latin = any(char in FULLWIDTH_LATIN for char in token) and any(
            "a" <= char.lower() <= "z" for char in token
        )
        if digits or latin:
            mixed.append(token)
    return mixed


def split_word_hits(sentence: str) -> list[tuple[str, str]]:
    """句中被拆開的目標詞，回傳 `(目標詞, 句中的實際寫法)`。"""
    hits: list[tuple[str, str]] = []
    for word, pattern in SPLIT_PATTERNS:
        for match in pattern.finditer(sentence):
            if len(match.group(0)) > len(word):
                hits.append((word, match.group(0)))
                break
    return hits


def zhuyin_hits(sentence: str) -> list[str]:
    """夾在兩個漢字之間的注音片段。

    **以位置判定，不以數量判定。** tg-spam 的 `isProhibitedLang()` 數的是整則訊息中
    某個 script 的字母數（門檻 3），但中文訊息裡注音出現在**詞中間**才是規避，
    出現在句首或句尾是正常的語氣用法（「ㄟ你在嗎」「好ㄛ」）。位置比數量有判別力。
    """
    hits: list[str] = []
    index = 0
    total = len(sentence)
    while index < total:
        if sentence[index] not in ZHUYIN:
            index += 1
            continue
        end = index
        while end < total and sentence[end] in ZHUYIN:
            end += 1
        before = sentence[index - 1] if index > 0 else ""
        after = sentence[end] if end < total else ""
        if before and after and _is_han(before) and _is_han(after):
            hits.append(sentence[index:end])
        index = end
    return hits


@dataclass(frozen=True)
class EvasionCheck:
    """五個規避檢查的共用形狀。

    分成五個獨立的 `Check` 而非一個內含五個分支的檢查，沿用 `rule-catalog` 的
    「一條規則一個 `Check`」—— `registry.disable("evasion_homophone")` 就是
    `add-ablation` 的逐條開關。
    """

    name: str
    stage: Stage = Stage.LOCAL

    def _result(self, detail: str, evidence: list[Coord]) -> CheckResult:
        """規避訊號的固定輸出形狀：非硬證據、不指向類型。

        規避是關於發訊者的意圖，不指向任何詐騙類型，所以它永遠不是硬證據。
        權重由 `weights.toml` 以 `(name, hard)` 查得，本層不攜帶數值。
        """
        return CheckResult(
            name=self.name,
            hit=True,
            detail=detail,
            evidence=evidence,
            scam_types=[],
            hard=False,
        )


@dataclass(frozen=True)
class InvisibleCharCheck(EvasionCheck):
    """不可見字元 —— 讀 `raw_at()`，**以整則訊息為計數範圍**。

    計數範圍是訊息不是句子：`NormalizedText` 把被刪除的字元歸入**前一個**區間，
    落在句子邊界上的零寬字元因此會被算到前一句的 `raw_sentences[i]` 上。
    逐句獨立計數會讓同一則訊息的規避痕跡依邊界位置而有不同的分布。
    `evidence` 仍指向單一句子（命中數最多的那一句），契約不變。
    """

    name: str = "evasion_invisible"

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results: list[CheckResult] = []
        for message_index in _message_indices(doc):
            counts: list[tuple[Coord, int]] = []
            marks: list[str] = []
            for position, index in enumerate(doc.message_range(message_index)):
                found = invisible_marks(doc.raw_sentences[index], at_message_start=position == 0)
                counts.append((doc.coords[index], len(found)))
                marks.extend(found)
            if not marks:
                continue
            summary = "、".join(
                f"{INVISIBLE_NAMES[char]} U+{ord(char):04X} ×{marks.count(char)}"
                for char in dict.fromkeys(marks)
            )
            results.append(
                self._result(f"原文含 {len(marks)} 個不可見字元：{summary}", [_peak(counts)])
            )
        return results


@dataclass(frozen=True)
class WidthMixCheck(EvasionCheck):
    """全形半形混用 —— 讀 `raw_at()`，NFKC 已把混用痕跡從 `text` 裡抹掉。

    **五個訊號裡誤判風險最高的一個。** 中文排版習慣本來就會混用（句中數字用全形、
    網址裡用半形），緩解是「同一 token 內」與「排除網址」兩層，
    加上它是 Tier-B、不指向類型 —— 誤命中只貢獻 `0.6` 的權重，
    不會產生「這是某某詐騙」的錯誤輸出。這仍是最可能在實測後被降權或移除的一個。
    """

    name: str = "evasion_width_mix"

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results: list[CheckResult] = []
        for message_index in _message_indices(doc):
            counts: list[tuple[Coord, int]] = []
            tokens: list[str] = []
            for index in doc.message_range(message_index):
                found = mixed_width_tokens(doc.raw_sentences[index])
                counts.append((doc.coords[index], len(found)))
                tokens.extend(found)
            if not tokens:
                continue
            results.append(
                self._result(
                    f"同一 token 內全形與半形混用：{'、'.join(tokens)}",
                    [_peak(counts)],
                )
            )
        return results


@dataclass(frozen=True)
class SplitWordCheck(EvasionCheck):
    """拆字 —— 讀 `text_at()`，目標詞導向，門檻是「命中 1 個目標詞」。

    不用空白比率或短詞比率：那兩個統計量在中文上沒有分布（見模組 docstring）。
    比對的對象是具體的詞而不是統計量，所以門檻也是詞而不是比率 ——
    一則正常訊息不會把「監管帳戶」寫成「監 管 帳 戶」。
    """

    name: str = "evasion_split_word"

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results: list[CheckResult] = []
        for message_index in _message_indices(doc):
            evidence: list[Coord] = []
            hits: list[tuple[str, str]] = []
            for index in doc.message_range(message_index):
                found = split_word_hits(doc.sentences[index])
                if found:
                    evidence.append(doc.coords[index])
                    hits.extend(found)
            if not hits:
                continue
            written = "、".join(f"{word} 被寫成「{actual}」" for word, actual in hits)
            results.append(self._result(f"目標詞被插入分隔字元拆開：{written}", evidence))
        return results


@dataclass(frozen=True)
class HomophoneCheck(EvasionCheck):
    """同音／形近變體 —— 讀 `text_at()`，以顯式對照表判定。

    `variants` 是欄位而非直接讀模組常數，使測試能在不污染 `HOMOPHONE_VARIANTS`
    的前提下驗證命中路徑 —— 表為空時這個檢查永不命中，
    而「永不命中」與「命中邏輯寫錯」在測試上必須能分開。
    """

    name: str = "evasion_homophone"
    variants: tuple[tuple[str, str], ...] = HOMOPHONE_VARIANTS

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results: list[CheckResult] = []
        for message_index in _message_indices(doc):
            evidence: list[Coord] = []
            found: list[str] = []
            for index in doc.message_range(message_index):
                sentence = doc.sentences[index]
                hits = [
                    f"{variant}（應為{origin}）"
                    for variant, origin in self.variants
                    if variant in sentence
                ]
                if hits:
                    evidence.append(doc.coords[index])
                    found.extend(hits)
            if not found:
                continue
            results.append(self._result(f"同音或形近變體：{'、'.join(found)}", evidence))
        return results


@dataclass(frozen=True)
class ZhuyinCheck(EvasionCheck):
    """注音夾在漢字之間 —— 讀 `text_at()`。

    ⚠️ **這條規則缺乏語料佐證。** 它的存在是因為 `workplan.yaml` 列了注音，
    不是因為量測到它在台灣詐騙語料中的頻率。`add-ablation` 應優先量它的貢獻，
    貢獻為零就該移除 —— 一條不會命中的規則只是讓 `Verdict.checks` 更長。
    """

    name: str = "evasion_zhuyin"

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results: list[CheckResult] = []
        for message_index in _message_indices(doc):
            evidence: list[Coord] = []
            found: list[str] = []
            for index in doc.message_range(message_index):
                hits = zhuyin_hits(doc.sentences[index])
                if hits:
                    evidence.append(doc.coords[index])
                    found.extend(hits)
            if not found:
                continue
            results.append(self._result(f"注音符號夾在漢字之間：{'、'.join(found)}", evidence))
        return results


EVASION_CHECKS: tuple[EvasionCheck, ...] = (
    InvisibleCharCheck(),
    WidthMixCheck(),
    SplitWordCheck(),
    HomophoneCheck(),
    ZhuyinCheck(),
)


def register_evasion_checks(registry: CheckRegistry) -> None:
    """把五個規避檢查逐一註冊進 registry。"""
    for check in EVASION_CHECKS:
        registry.register(check)
