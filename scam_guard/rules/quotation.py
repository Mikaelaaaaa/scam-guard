"""引述偵測 —— 本專案唯一一個目標不是找出詐騙，而是**擋住規則層自己**的檢查。

**防詐宣導文比真詐騙訊息含有更多詐騙關鍵字。** 這不是推測：

    最近很多假檢警詐騙，會叫你把錢匯到監管帳戶，千萬不要相信

這一句同時命中 `safe_account` 與 `secrecy_demand` 的詞表，而它是宣導。
真正的假檢警訊息通常只講其中一件事 —— 宣導文為了講完整個手法會把話術全部列出來。
關鍵詞密度與詐騙性在這裡是**反相關**的。`實驗結果.md` 記錄了同一現象在 embedding
層的表現：查詢一則假投資詐騙時前三個鄰居的相似度是 0.891 / 0.890 / 0.889，
標籤卻是一詐兩合，其中一則內容是在**描述**作手手法。

規則層偵測的是「訊息裡有沒有這個言語行為」，而引述文本裡確實**有**那個言語行為
—— 只是說話的人在複述別人的話。本檢查的職責到「把這則訊息交給 LLM」為止，
不自己判斷引述的內容是真是假（那需要理解整則訊息的意圖，是 L3 的工作）。

**誤判的兩個方向代價不對稱，因此寧可多命中**：

- 把真詐騙判成引述 → 短路被否決，`Stage.EXPENSIVE` 照常執行。
  所有訊號原封不動，代價是一次 LLM 呼叫。
- 把宣導判成詐騙 → 硬證據觸發短路、LLM 被跳過，系統對一則防詐宣導文輸出
  「詐騙可能性 0.9」。代價是一次對外可見的誤判，而誤判率是強制驗收指標。

兩者相差好幾個數量級。高 recall、低 precision 是刻意的取捨。
**但這個推論有前提：它只在 LLM 掛載時成立**，見 `QUOTE_WEIGHT`。
"""

import re
from dataclasses import dataclass

from scam_guard.check import Stage
from scam_guard.normalize import Document, normalize_text
from scam_guard.types import CheckResult, Coord, Request

NAME = "quotation"
"""檢查名稱 —— MUST 等於 `scam_guard.pipeline.QUOTATION_CHECK`。

`pipeline._should_short_circuit()` 以**字面名稱**認出這個檢查並否決短路。
一致性由 `tests/test_quotation_check.py` 的一行斷言保證，**不由 import 保證**：
import 方向是 `rules/ → normalize.py → types.py`，而 `pipeline.py` 在最上層，
讓 `rules/` import 它會產生一條由下往上的邊。目前不會循環（`pipeline` 不
import `rules`，registry 由呼叫端組裝），但那是一個隨時可能被打破的巧合，
而 Python 沒有循環 import 的保護。測試比 import 弱，但代價只有一行。
"""

QUOTE_WEIGHT = -1.5
"""**負**權重 —— 純規則模式下對宣導文的唯一保護，且是**佔位值**。

純規則模式（尚未掛 LLM，或 `add-ablation` 刻意關閉）下 registry 裡沒有任何
`Stage.EXPENSIVE` 檢查，否決短路之後什麼都不會發生 —— `expensive` 是空 list，
跑不跑都一樣。而 `rule-signals` 這個 PR 的定位正是 baseline，
一個在 baseline 上沒有效果的檢查等於承認 baseline 完全無法處理宣導文。

`-1.5` 與 Tier-A 的 `2.5`、Tier-B 的 `0.6` 在同一個 log-odds 尺度上，
量級介於兩者之間：「一條 Tier-A 命中加上引述命中，淨值仍為正但不足以直接定讞」。

**已知限制要誠實講**：一則命中三條 Tier-A 的宣導文（同群組取 max 後為 `2.5`）
加上 `-1.5` 淨值仍為正 `1.0` —— 負權重擋不住多條 Tier-A 同時命中的宣導文，
而那恰恰是宣導文的特徵。`add-ablation` 應把這件事量出來寫進報告，
不要以「規則系統誤判率低」的籠統說法帶過。
"""

QUOTE_CATEGORY = "引號"
ATTRIBUTION_CATEGORY = "來源歸屬"
FORWARD_CATEGORY = "轉傳格式"
AWARENESS_CATEGORY = "宣導框架"

ATTRIBUTION = frozenset(
    {
        "有人傳給我",
        "有人傳這個",
        "有人傳來",
        "我收到",
        "朋友收到",
        "剛收到",
        "對方說",
        "他說",
        "她說",
        "據說",
        "聽說",
        "網路上流傳",
        "詐騙集團會",
        "歹徒會",
        "對方會",
    }
)
"""來源歸屬 —— 把話語歸給另一個人。"""

FORWARD_MARKERS = frozenset(
    {"轉傳", "轉發", "※", "以下訊息", "幫忙轉發", "請大家小心", "分享給大家"}
)
"""轉傳格式的**詞彙**標記。結構標記（時間戳行）另見 `TIMESTAMP_LINE`。"""

AWARENESS_FRAME = frozenset(
    {
        "千萬不要相信",
        "千萬別相信",
        "這是詐騙",
        "是詐騙集團",
        "請勿受騙",
        "切勿上當",
        "165",
        "反詐騙",
        "提醒大家",
        "警方呼籲",
        "近期常見",
        "新型態詐騙",
        "本行絕不會",
        "絕不會要求",
    }
)
"""宣導框架。

這一類天生 precision 低 —— 「本行絕不會以電話要求您操作 ATM」是真實銀行簡訊的
標準句，它會命中。方向是「把正常訊息判得更正常」，無害：沒有硬證據的正常訊息
本來就會被送去 LLM（`_should_short_circuit()` 在無硬證據時本來就回傳 `False`），
**對絕大多數正常訊息，引述誤命中的代價是零**。

「近期常見」比 tasks 列舉的「近期常見手法」短，是刻意的：對抗樣本
「近期常見假冒銀行詐騙，為保障您的權益請點此完成帳戶驗證」中間夾著別的字，
比對整個「近期常見手法」會漏掉它，而那正是本檢查要認出的偽裝形狀。
"""

QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ("「", "」"),
    ("『", "』"),
    ("“", "”"),
    ('"', '"'),
    ("'", "'"),
)
"""引號的成對寫法。彎引號 `“”` 不在 tasks 列舉中，補上的理由是 NFKC 不會把它
收斂成任何其他寫法，漏掉它的方向是引述漏命中 —— 而那是代價大的那個方向。
"""

MIN_QUOTE_LEN = 8
"""引號內文字的長度門檻 —— **本 change 唯一需要調的參數**。

引號在中文訊息中的用法很廣：「『限時』特價」是強調、歌名書名用引號、品牌名用引號。
以引號單獨作為命中條件的 precision 會很低，因此引號**單獨命中不足以觸發**，
必須引號內的文字達到門檻才計入。

`8` 沒有實測依據，是從「引述一段話術至少要一個子句的長度」推出的起點，
而強調用法通常只有兩三個字。`add-ablation` 應把它納入掃描，
與 `Limits` 的兩個上限同等對待。
"""

TIMESTAMP_LINE = re.compile(r"^\s*(?:上午|下午)?\s*(?:\d{4}/\d{1,2}/\d{1,2}|\d{1,2}[:/]\d{1,2})")
"""行首的時間戳樣式 —— LINE 對話匯出格式的每一行都長這樣。

這條標記的價值在於它**不依賴任何詞彙**：詐騙者無法透過改寫話術繞過它，
而其他三類標記都可以。
"""

MIN_TIMESTAMP_LINES = 2
"""時間戳行數的門檻。

單一時間戳可能是訊息內容的一部分（「請於 09/20 前完成」），
兩行以上的時間戳在正常訊息中沒有理由出現 —— 那是貼上來的對話紀錄，
而貼上來的對話紀錄就是「有人傳給我，幫我看看」。
"""


def _nested_spans(sentence: str, opening: str, closing: str) -> list[tuple[int, int]]:
    """以深度計數取**最外層**的成對區間。未成對者略過，不拋例外。"""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    for index, char in enumerate(sentence):
        if char == opening:
            if depth == 0:
                start = index
            depth += 1
        elif char == closing and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, index))
    return spans


def _alternating_spans(sentence: str, delimiter: str) -> list[tuple[int, int]]:
    """開閉同形的引號（`"` `'`）依出現順序兩兩配對，落單的最後一個略過。"""
    positions = [index for index, char in enumerate(sentence) if char == delimiter]
    return [(positions[i], positions[i + 1]) for i in range(0, len(positions) - 1, 2)]


def quoted_spans(sentence: str) -> list[tuple[int, int]]:
    """句中成對引號的區間 `[開引號位置, 閉引號位置]`，依位置排序。

    巢狀引號取最外層，不遞迴 —— 引述的是整段話，不是話裡的話。
    被別的區間包住的區間一律丟棄，跨引號型別亦然（`「他說『…』」`）。

    只有開引號而無閉引號時該處略過：引號在真實訊息中經常不成對，
    為此拋例外會讓一個常見的書寫習慣變成執行期錯誤。
    """
    spans: list[tuple[int, int]] = []
    for opening, closing in QUOTE_PAIRS:
        if opening == closing:
            spans.extend(_alternating_spans(sentence, opening))
        else:
            spans.extend(_nested_spans(sentence, opening, closing))
    return sorted(
        span
        for span in spans
        if not any(other[0] < span[0] and span[1] < other[1] for other in spans)
    )


def long_quotes(sentence: str) -> list[str]:
    """句中長度達 `MIN_QUOTE_LEN` 的引號內容。"""
    return [
        sentence[start + 1 : end]
        for start, end in quoted_spans(sentence)
        if end - start - 1 >= MIN_QUOTE_LEN
    ]


def _found_words(sentence: str, words: frozenset[str]) -> list[str]:
    return sorted(word for word in words if word in sentence)


def _timestamp_lines(text: str) -> int:
    """整則訊息中符合時間戳樣式的行數。

    以**訊息的換行**判定而非以句子判定：`split_sentences()` 把換行當分隔符
    並且不把它納入句子，行的概念在 `Document` 裡已經消失了。
    這裡對原文再正規化一次只是為了把全形數字與冒號收斂成 ASCII，
    不產生任何座標 —— 座標系仍然只有 `build_document()` 一個產生者。
    """
    lines = normalize_text(text).text.split("\n")
    return sum(1 for line in lines if TIMESTAMP_LINE.match(line))


@dataclass(frozen=True)
class QuotationCheck:
    """引述偵測。**四類標記共用一個 `Check`，整則請求最多產出一筆結果。**

    這與 `rule-catalog` 的「一條規則一個 `Check`」看起來衝突，需要交代：
    `pipeline` 以 `r.name == QUOTATION_CHECK` 比對名稱，拆成四個檢查會讓短路
    否決的條件從「引述命中」變成「四者之一命中」，而 `pipeline` 得改成比對前綴
    或維護一張名單。既有契約優先於目錄慣例。消融粒度停在 `quotation` 這一層
    足夠 —— `add-ablation` 要量的是「關掉引述偵測後宣導文的誤判率變多少」，
    不是「四類標記各貢獻多少」。

    一筆而非每個標記一筆：三個 URL 是三個獨立的訊號，四類引述標記不是 ——
    它們共同支撐同一個結論（這則訊息在引述別人的話）。

    **不標記被引述的區段，不抑制其他規則的命中。** 引號經常不成對、
    「他說」的管轄範圍沒有任何標記可以界定，猜錯而把真詐騙的要求句劃進引述範圍，
    該條 Tier-A 就不命中，訊號**完全消失**，`Verdict.checks` 裡連記錄都沒有。
    這與自帶碼豁免選擇「降級而非消滅」是同一個判斷。
    """

    name: str = NAME
    stage: Stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        categories: dict[str, list[str]] = {}
        coords: set[Coord] = set()

        for coord, sentence in zip(doc.coords, doc.sentences):
            found = {
                QUOTE_CATEGORY: long_quotes(sentence),
                ATTRIBUTION_CATEGORY: _found_words(sentence, ATTRIBUTION),
                FORWARD_CATEGORY: _found_words(sentence, FORWARD_MARKERS),
                AWARENESS_CATEGORY: _found_words(sentence, AWARENESS_FRAME),
            }
            for category, markers in found.items():
                if markers:
                    categories.setdefault(category, []).extend(markers)
                    coords.add(coord)

        for message_index in sorted({coord[0] for coord in doc.coords}):
            if _timestamp_lines(req.messages[message_index].text) < MIN_TIMESTAMP_LINES:
                continue
            categories.setdefault(FORWARD_CATEGORY, []).append("時間戳行")
            coords.update(
                doc.coords[index]
                for index in doc.message_range(message_index)
                if TIMESTAMP_LINE.match(doc.sentences[index])
            )

        if not categories:
            return []

        ordered = [
            category
            for category in (
                QUOTE_CATEGORY,
                ATTRIBUTION_CATEGORY,
                FORWARD_CATEGORY,
                AWARENESS_CATEGORY,
            )
            if category in categories
        ]
        markers = sorted({marker for category in ordered for marker in categories[category]})
        return [
            CheckResult(
                name=self.name,
                hit=True,
                weight=QUOTE_WEIGHT,
                detail=(
                    f"命中 {len(ordered)} 類引述標記：{'、'.join(ordered)}；"
                    f"具體標記：{'、'.join(markers)}"
                ),
                evidence=sorted(coords),
                scam_types=[],
                hard=False,
            )
        ]
