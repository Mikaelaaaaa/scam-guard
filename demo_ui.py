"""兩份介面層共用的標記層 —— 畫面上的每一段 HTML 都由這裡產生。

`app.py`（Gradio）與 `docs/pages_app.py`（Pyodide 靜態站台）是兩個載體，
但它們呈現的是同一個 `Verdict`。判定卡、偵測細節、訊息氣泡、逸出、狀態分類、
信心分級與樣式表因此各只有一份實作，放在這裡。

**本模組 MUST NOT import gradio。** 這不只是為了避開 `pyproject.toml` 的
`banned-api`：Pyodide 裡沒有 gradio，而靜態站台要 import 這個模組，
所以「不依賴介面框架」是它在那一側能被使用的**前提**。這條界線由
`tests/test_demo_ui.py` 斷言，不靠自律維持。

**不放進 `scam_guard/`。** 那個套件宣告自己不做 I/O、不知道有 LINE 或 Gradio，
把產 HTML 的模組放進去是把界線推倒。終點仍是 `app.py` 原本的交棒註記寫的那個：
文案層由 `add-verdict-render` 在 `scam_guard/` 內持有（`Verdict.evidence` 與
`Verdict.actions` 本來就是它產的），而本模組是**標記層**，那一層本來就不該進核心。

**本模組已知而刻意不解的一件事：** `CheckResult.indeterminate=True` 落在五個
狀態術語之外。`check_state()` 對它會 `raise` 而不是猜一個術語 —— 今天不會發生
（唯一的產生者是 `domain_age`，兩個部署都不註冊它），它落地的那一天會是一個
大聲的失敗。處置屬 `add-domain-age` 的部署，不屬本模組。
"""

import html
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from scam_guard import pii
from scam_guard.normalize import Document
from scam_guard.pipeline import NOT_HIT, SKIPPED
from scam_guard.render import QUOTE_SEPARATOR
from scam_guard.scoring import Score, compute_score, is_decision
from scam_guard.types import CheckResult, Coord, Message, Verdict
from scam_guard.weights import WeightTable

# ---------------------------------------------------------------------------
# 型別
# ---------------------------------------------------------------------------

PiiSpan: TypeAlias = tuple[int, int, str]
"""個資區間 `(start, end, type)`，座標在**正規化後的句子**上，對齊
`add-pii-recognizers` 的契約（辨識器只回報區間與類型，不改寫文字）。"""

PiiRecognizer: TypeAlias = Callable[[str], list[PiiSpan]]

UnregisteredCheck: TypeAlias = tuple[str, str, str]
"""`(識別字, 中文名, 一行理由)`。由各部署的組裝層提供 —— 哪些檢查沒有註冊
是那一側的事實，不是本模組知道的事。"""


# ---------------------------------------------------------------------------
# 五個狀態術語
# ---------------------------------------------------------------------------

TERM_CONCLUSIVE = "Conclusive"
TERM_INDICATIVE = "Indicative"
TERM_CLEAR = "Clear"
TERM_SKIPPED = "Skipped"
TERM_DISABLED = "Disabled"

QUIET_TERMS = (TERM_CLEAR, TERM_SKIPPED)
"""收在第二層的兩個術語：檢查跑過沒命中，或因為前面已足以判定而沒跑。"""

NOTE_SEPARATOR = "；"
"""`detail` 內部的分段符。

⚠️ 這是 `scam_guard.rules.speech_act._detail()` 的**私有實作細節**，沒有任何契約，
而且只有言語行為規則用它（URL 層與規避偵測的 `detail` 根本沒有分號）。
`tests/test_demo_ui.py` 拿真實規則的 `summary` 與 `fact` 逐字比對鎖住它 ——
那一行 join 改掉的當下測試會紅，而不是介面上安靜地出現一行四十個字的標題。
"""


# ---------------------------------------------------------------------------
# 文案
# ---------------------------------------------------------------------------

TITLE_SCAM = "很可能是詐騙"
TITLE_SIGNALS = "有可疑訊號，但不足以判定"
TITLE_UNDECIDED = "無法判定"

LEAD_SIGNALS = "找到 {count} 項訊號，但它們加起來還不夠下結論"
SCORE_LABEL = "詐騙分數 {value:.2f}"
GATE_LABEL = "判定門檻 {gate:.2f}"
SCORING_SCORE = "分數 {value:.2f}"
SCORING_GATE = "門檻 {gate:.2f}"
SCORING_PROBABILITY = "機率 {probability:.0%}"
SCORING_CONFIDENCE = "信心 {confidence:.2f}"
LEAD_UNDECIDED = "系統沒有找到足夠的依據。這不代表它安全，只代表系統沒有看出訊號。"
VERIFY_LINE = "不確定的時候，撥打 165 反詐騙專線查證。"
"""無法判定卡上的固定內容。

它不是建議清單的一員 —— `Verdict.actions` 在無命中時刻意是空陣列（`render.py`：
「對一則『明天見』說『建議撥打 165』是製造焦慮」）。這一句屬於「無法判定」
這張卡本身，與那張卡一起出現、一起消失，不會掛在一則有判定的訊息下面。
"""

WHY_HEADING = "為什麼"
ACTIONS_HEADING = "你可以這樣做"
CONFIDENCE_PREFIX = "信心"

BAND_HIGH = "高"
BAND_MEDIUM = "中"
BAND_LOW = "低"

NO_NEW_SIGNAL = "這一輪沒有出現新的訊號。"

RANKING_HEADING = "累積命中的訊號（第 {turns} 輪）"
RANKING_EMPTY = "這 {turns} 輪裡沒有任何訊號命中。"
RANKING_NO_COORD = "—"

DETAILS_SUMMARY = "偵測細節（共 {total} 項）"
DETAILS_SUMMARY_WITH_HITS = "偵測細節（共 {total} 項，{hits} 項命中）"
QUIET_SUMMARY = "其餘 {count} 項沒有命中（{breakdown}）"
TRUNCATION_LINE = "訊息太長，最舊的 {dropped} 則沒有納入這次判定。"
DROPPED_LABEL = "未納入本次判定"

PII_HEADING = "訊息裡的個人資料"
PII_NOT_MOUNTED = "這個版本沒有開啟個人資料標註。"
PII_NO_HIT = "沒有在訊息裡找到個人資料。"
PII_SCOPE_NOTE = (
    "能認出來的只有身分證字號、手機號碼、市話與信用卡號四種。"
    "<b>姓名與地址不在辨識範圍內</b>，要認出這兩種得換一套誤判率很高的模型，"
    "本服務預設關閉它。"
)
PII_NORMALIZED_NOTE = (
    "上面顯示的是系統整理過的句子（統一全形半形、去掉隱藏字元），不是原文。"
    "標示只指出位置與類型，不改寫任何文字。"
)

PII_LABELS: Mapping[str, str] = MappingProxyType(
    {
        pii.TW_ID: "身分證字號",
        pii.TW_MOBILE: "手機號碼",
        pii.TW_LANDLINE: "市話",
        pii.CREDIT_CARD: "信用卡號",
    }
)
"""四個類型在畫面上的中文顯示名。**鍵取自 `scam_guard.pii` 的常數，不寫字面值** ——
常數改名時這裡是 import 失敗，不是畫面上多一個標不出名字的東西。
`tests/test_demo_ui.py` 斷言鍵集合等於 `pii.ENTITY_TYPES`，封閉集合日後加第五類時
那條測試會紅，而不是介面安靜地掉回英文識別字。

⚠️ 這是本模組**唯一**用到 `scam_guard.pii` 的地方，而且只取字串常數，
不呼叫 `find_pii()`。標記層拿到的是**詞彙**不是**辨識器**：
`recognizer is None` 仍然是唯一決定「有沒有掛載」的東西，那個決定留在各自的介面層。

`TW_ID` 的顯示名刻意**不排他**。駕照號與軍人補給證號與國民身分證共用
`[A-Z][12]\\d{8}` 的形狀並通過**同一個** checksum，在字元層不可區分 ——
標到它們標到的確實是個資，只是不必然是國民身分證。
"""

GATE_HEADROOM = 1.5
SCORE_HEADROOM = 1.2
"""刻度尺右界的兩個係數：右界 = `max(門檻 × 1.5, 分數 × 1.2)`。

**分數沒有上限**（一條 Tier-A 就 2.5，多條可以到 5 以上），所以刻度尺不能把
判定門檻畫成右端點 —— 那會讀成「滿分是 1.50，而 1.20 快滿了」，
而實際意義完全相反：1.20 是**還沒到門檻**。門檻因此必須落在軸的中間某處，
分數在它左邊或右邊都畫得下。

兩個係數是編出來的，記在這裡：1.5 讓門檻落在軸長的三分之二處（分數為 0 時
軸仍有意義），1.2 讓超過門檻的分數右邊仍留一段空白，不會頂到邊。
它們不影響任何 requirement，改動只改變留白多寡。
"""

MIN_BAR_WIDTH = 12.0
"""長條的最小可見寬度（百分比）。

一次命中不該是一個像素，而以次數當絕對寬度會讓長條在一則短訊息上全部縮成
看不見。12 沒有依據，它只是「一根看得見的短棒」—— 這是本模組唯一一個編出來的
數字，改它不影響任何 requirement。
"""


# ---------------------------------------------------------------------------
# 逸出與錨點
# ---------------------------------------------------------------------------


def escaped(value: str) -> str:
    """使用者輸入進入標記之前的唯一入口。

    輸入是真實詐騙訊息，裡面有 `<`、`&`、完整網址，不逸出就是一個 XSS，
    而兩個載體都是公開的頁面。
    """
    return html.escape(value).replace("\n", "<br>")


def anchor_id(coord: Coord) -> str:
    """證據座標對應的 HTML 錨點 id。純 HTML 與 CSS，不需要 JavaScript。"""
    message_index, sentence_index = coord
    return f"s-{message_index}-{sentence_index}"


# ---------------------------------------------------------------------------
# 資料拆解
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceLine:
    """一行依據拆開之後的三個部分。`quote` 為 `None` 代表那一行沒有原文引用。"""

    title: str
    notes: tuple[str, ...]
    quote: str | None


def split_detail(detail: str) -> tuple[str, tuple[str, ...]]:
    """把 `CheckResult.detail` 切成標題與附註。

    規則只有一條：依全形分號切段，第一段是標題，其餘是附註。只有一段時附註為空
    tuple —— 那是同一條規則的自然結果，不是一個分支。
    """
    segments = detail.split(NOTE_SEPARATOR)
    return segments[0], tuple(segments[1:])


def split_evidence_line(line: str) -> EvidenceLine:
    """把 `Verdict.evidence` 的一行拆成標題、附註與原文引用。

    以 `render.QUOTE_SEPARATOR` 的**第一次**出現切掉引用：`detail` 依 `types.py`
    的規定不得含原文片段，而既有 `detail` 的分段用的是 `事實：`（沒有 `「`）。
    `render.py` 的往返契約（「去掉原文片段之後 MUST 等於某個 `hit=True` 結果的
    `detail`」）因此在這裡成立。

    `CONTRADICTION_NOTE` 與 `TRUNCATION_NOTE` 兩行不含引用也不含分號，走同一條
    路徑後成為只有標題的一項。不需要特例。
    """
    head, separator, tail = line.partition(QUOTE_SEPARATOR)
    quote = tail.removesuffix("」") if separator else None
    title, notes = split_detail(head)
    return EvidenceLine(title=title, notes=notes, quote=quote)


def check_state(result: CheckResult) -> str:
    """把一筆 `CheckResult` 分類為四個術語之一（`Disabled` 不從這裡來）。

    判定依據是 `scam_guard.pipeline` 公開的 `NOT_HIT` 與 `SKIPPED` 兩個常數，
    **不是字串字面值** —— 介面層 import 核心是允許的方向。

    無法分類即 `raise`，不猜。猜錯會讓一個回報了奇怪 `detail` 的檢查看起來
    只是狀態標籤不同；而根治它需要 `CheckResult` 新增 `skipped` 欄位，
    那屬 `add-check-pipeline` 的後續，不屬本模組。
    """
    if result.hit:
        return TERM_CONCLUSIVE if result.hard else TERM_INDICATIVE
    if result.detail == NOT_HIT:
        return TERM_CLEAR
    if result.detail == SKIPPED:
        return TERM_SKIPPED
    raise ValueError(
        f"無法分類的檢查記錄：name={result.name!r}、hit=False、detail={result.detail!r}"
        f"（未命中的 detail 只能是 pipeline.NOT_HIT 或 pipeline.SKIPPED）"
    )


def confidence_band(confidence: float, table: WeightTable) -> str | None:
    """信心的三級標籤。`None` 代表低於地板 —— 那是「無法判定」，卡上不顯示等級。

    切點**向 `WeightTable` 取值**，不寫死 0.90 / 0.55 / 0.40：`add-metrics`
    校準時會移動那八個數字，寫死字面值會讓標籤在校準後指向錯誤的等級，
    而沒有任何地方會報告。

    三級講的是「系統憑什麼下這個判斷」而不是「這個數字有多大」：高的充要條件是
    `confidence.py` 第一級（存在硬證據且無上限生效），中對應第二級，
    低只可能來自某個上限。與該模組「信心 MUST NOT 讀取分數的大小」同一個立場。
    """
    if confidence < table.threshold("confidence_floor"):
        return None
    if confidence >= table.threshold("base_hard"):
        return BAND_HIGH
    if confidence >= table.threshold("base_multi_group"):
        return BAND_MEDIUM
    return BAND_LOW


def verdict_state(verdict: Verdict, table: WeightTable) -> str:
    """判定卡的三個狀態之一，回傳該狀態的標題。

    狀態**不從百分比的大小推導** —— 機率是未校準的（`sigmoid_offset` 與
    `sigmoid_temperature` 在 `weights.toml` 裡都是 `placeholder`），
    從一個未校準的數字劃「很可能／可能／不太可能」是在編級距。

    「有機率」與「做出詐騙判定」是兩條線，今天就會分開：兩條分屬不同群組的
    Tier-B 命中時分數 1.2 低於 `decision_score`，而信心 0.55 高於地板，
    於是機率不是 `None` 但 `is_decision()` 為假。二分法會對這則訊息印
    「很可能是詐騙 77%」，而計分層明確地**沒有**判它是詐騙。

    `is_decision` 由**重算** `compute_score()` 取得，不在這裡比較
    `score.value >= 1.5`：`Verdict` 沒有攜帶 `Score`，而 `compute_score()` 是
    `(results, table)` 的純函式，同樣的輸入必然得到同樣的輸出。抄一份門檻比較
    才是第二個實作。
    """
    if verdict.scam_probability is None:
        return TITLE_UNDECIDED
    score = compute_score(verdict.checks, table)
    if is_decision(score, verdict, table):
        return TITLE_SCAM
    return TITLE_SIGNALS


# ---------------------------------------------------------------------------
# 判定卡
# ---------------------------------------------------------------------------


def _facts(verdict: Verdict, state: str, table: WeightTable) -> str:
    """標題列右側的量。

    三個狀態各自帶不同的量，而這不是版面偷懶：
    「很可能是詐騙」帶機率、類型與信心等級 —— 系統做了判定，三者都是那個判定的一部分。
    「有可疑訊號」只帶信心等級 —— 機率是一個計分層不認可的數字，印出來就是把
    「未達門檻」包裝成「七成七像詐騙」；類型是一個關於「這是哪一種詐騙」的主張，
    而此時系統連「是不是」都還沒下結論。信心等級答的是「有沒有足夠依據」，
    那正是這張卡在講的事，所以留著。
    「無法判定」什麼都不帶。
    """
    band = confidence_band(verdict.confidence, table)
    cells: list[str] = []
    if state == TITLE_SCAM:
        cells.append(f'<span class="fact-main">{verdict.scam_probability:.0%}</span>')
        if verdict.scam_type is not None:
            cells.append(f'<span class="fact">{escaped(verdict.scam_type.value)}</span>')
    if state != TITLE_UNDECIDED and band is not None:
        cells.append(f'<span class="fact">{CONFIDENCE_PREFIX} {escaped(band)}</span>')
    if not cells:
        return ""
    return f'<div class="card-facts">{"".join(cells)}</div>'


def _lead(verdict: Verdict, state: str) -> str:
    """標題底下的一句話。有判定時不需要 —— 「為什麼」清單接著就講了。"""
    if state == TITLE_UNDECIDED:
        return f'<p class="card-lead">{LEAD_UNDECIDED}</p><p class="card-lead">{VERIFY_LINE}</p>'
    if state == TITLE_SIGNALS:
        hits = sum(1 for result in verdict.checks if result.hit)
        return f'<p class="card-lead">{LEAD_SIGNALS.format(count=hits)}</p>'
    return ""


def scale_bound(value: float, gate: float) -> float:
    """刻度尺的右界。門檻**永遠不在最右端**，分數超過門檻時也畫得下。"""
    return max(gate * GATE_HEADROOM, value * SCORE_HEADROOM)


def render_score_scale(score: Score, table: WeightTable) -> str:
    """「還差多遠」的刻度尺 —— 只出現在「有可疑訊號，但不足以判定」這一態。

    這一態的畫面上不放機率百分比：`77%` 與判定成立時的 `92%` 長得一模一樣，
    使用者分不出哪一個是系統認可的判定、哪一個是未達門檻的中間值。
    分數與門檻的相對位置自己說明了為什麼不判定，不需要再寫一句解釋。

    **不寫成「1.20 / 1.50」。** 那個斜線的意思是「滿分 1.50」，而 1.50 是**門檻**
    不是滿分 —— 分數沒有上限。門檻值向 `WeightTable` 取，不寫死。
    """
    gate = table.threshold("decision_score")
    bound = scale_bound(score.value, gate)
    return (
        '<div class="scale">'
        f'<div class="scale-labels"><span class="scale-score">'
        f"{SCORE_LABEL.format(value=score.value)}</span>"
        f'<span class="scale-gate">{GATE_LABEL.format(gate=gate)}</span></div>'
        '<div class="scale-axis">'
        f'<i class="scale-fill" style="width:{100.0 * score.value / bound:.1f}%"></i>'
        f'<i class="scale-mark" style="left:{100.0 * gate / bound:.1f}%"></i></div>'
        f'<div class="scale-ticks"><span>0</span>'
        f'<span class="scale-tick-gate" style="left:{100.0 * gate / bound:.1f}%">'
        f"{gate:.2f}</span></div>"
        "</div>"
    )


def scoring_line(
    verdict: Verdict, score: Score | None, table: WeightTable, show_probability: bool
) -> str:
    """偵測細節最上面的計分摘要。

    機率只在**卡上沒有印它**的時候出現在這裡：一個未達判定門檻的百分比放在
    卡片第一層會被讀成判定，放在這裡則有脈絡 —— 旁邊就是分數、門檻與逐項訊號。
    卡上已經印了機率時再印一次，就是這個 change 要消掉的那種重複。
    """
    if score is None:
        return ""
    parts = [
        SCORING_SCORE.format(value=score.value),
        SCORING_GATE.format(gate=table.threshold("decision_score")),
    ]
    if show_probability:
        parts.append(SCORING_PROBABILITY.format(probability=score.probability))
    parts.append(SCORING_CONFIDENCE.format(confidence=verdict.confidence))
    return f'<div class="scoring">{" · ".join(parts)}</div>'


def render_why(verdict: Verdict) -> str:
    """「為什麼」清單，逐項取自 `Verdict.evidence`，**只取標題與附註**。

    原文引用留給偵測細節：同一段原文在兩個地方各出現一次，就是這個 change 要
    消掉的那種重複。

    `Verdict.evidence` 為空時整塊不輸出 —— 不印一句「（未經文案渲染的原始檢查
    明細）」加上逐筆 `name：detail`。那條路徑今天仍可達（只有引述訊號命中且未被
    採計時），而正確的處置是這塊不存在，命中的項目照常出現在偵測細節裡：
    資訊沒有遺失，畫面上不會有一句需要解釋的話。
    """
    if not verdict.evidence:
        return ""
    items: list[str] = []
    for line in verdict.evidence:
        parsed = split_evidence_line(line)
        notes = "".join(f'<div class="why-note">{escaped(note)}</div>' for note in parsed.notes)
        items.append(f'<li><div class="why-title">{escaped(parsed.title)}</div>{notes}</li>')
    return f'<section class="why"><h4>{WHY_HEADING}</h4><ul>{"".join(items)}</ul></section>'


def render_actions(verdict: Verdict) -> str:
    """「你可以這樣做」，**逐字**等於 `Verdict.actions` 的元素。

    為空時整塊不輸出，且不補任何保底建議：判定層以空陣列表達「這則訊息不該給
    建議」，介面層在那之上再補一句，正好是它明文反對的行為。
    """
    if not verdict.actions:
        return ""
    items = "".join(f"<li>{escaped(action)}</li>" for action in verdict.actions)
    return f'<section class="todo"><h4>{ACTIONS_HEADING}</h4><ul>{items}</ul></section>'


def render_verdict_card(
    verdict: Verdict,
    doc: Document,
    table: WeightTable,
    unregistered: Sequence[UnregisteredCheck],
    recognizer: PiiRecognizer | None = None,
) -> str:
    """判定卡 —— 判定結果在畫面上的**唯一**出處。

    取代了原本的「判定列 + 判定結果散文段 + 右欄面板」三個區塊：同一則假檢警
    訊息原本會在三個地方各講一次機率、信心與類型。
    """
    state = verdict_state(verdict, table)
    score = None if verdict.scam_probability is None else compute_score(verdict.checks, table)
    scale = ""
    if state == TITLE_SIGNALS and score is not None:
        scale = render_score_scale(score, table)
    scoring = scoring_line(verdict, score, table, show_probability=state != TITLE_SCAM)
    return (
        '<div class="card">'
        f'<div class="card-head"><div class="card-title">{escaped(state)}</div>'
        f"{_facts(verdict, state, table)}</div>"
        f"{scale}"
        f"{_lead(verdict, state)}"
        f"{render_why(verdict)}"
        f"{render_actions(verdict)}"
        f"{render_details(verdict, doc, unregistered, recognizer, scoring)}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# 偵測細節
# ---------------------------------------------------------------------------


def render_quote(result: CheckResult, doc: Document) -> str:
    """命中項目的原文引用，連向對話中的那一句。

    **無效座標讓 `KeyError` 向上傳播，不吞。** `Document.index_of()` 的 docstring
    寫了理由：無效座標代表產生它的檢查算錯了。吞掉它會讓一個算錯座標的檢查
    看起來只是少一條引用。
    """
    if not result.evidence:
        return ""
    quotes = [
        f'<a class="quote" href="#{anchor_id(coord)}">「{escaped(doc.raw_at(coord))}」</a>'
        for coord in result.evidence
    ]
    return f'<div class="quote-list">{"".join(quotes)}</div>'


def _item(title: str, name: str, term: str, body: str) -> str:
    """偵測細節的一列。中文名與英文識別字並列，識別字**不翻譯也不改寫**。"""
    return (
        f'<div class="item item-{term.lower()}">'
        f'<div class="item-head"><span class="item-title">{escaped(title)}</span>'
        f'<span class="item-id">{escaped(name)}</span>'
        f'<span class="item-term">{escaped(term)}</span></div>'
        f"{body}</div>"
    )


def _quiet_item(name: str, term: str) -> str:
    """第二層的一列：只有識別字與術語。

    沒有命中的項目**沒有中文名可取** —— 它的 `detail` 就是 `NOT_HIT` 或
    `SKIPPED` 這兩個狀態常數本身，拿去當名稱會讓二十幾列每一列都寫著
    「未命中　solicit_otp　Clear」，同一件事講兩次。中文名在命中時才存在
    （那時 `detail` 是規則的摘要），所以它在第一層出現、在這裡不出現。
    """
    return (
        f'<div class="item item-{term.lower()}"><div class="item-head">'
        f'<span class="item-id item-id-solo">{escaped(name)}</span>'
        f'<span class="item-term">{escaped(term)}</span></div></div>'
    )


def _quiet_summary(terms: Sequence[str]) -> str:
    """第二層的收合列。**逐態報數**，不把兩種狀態混成一個「其他」。

    `Skipped` 與 `Clear` 不是同一件事：前者是系統看都沒看（前面已經足以判定），
    後者是看過了沒有訊號。一個把兩者合併的數字會讓使用者以為全部都跑過了。
    """
    breakdown = "、".join(
        f"{terms.count(term)} 項 {term}" for term in QUIET_TERMS if terms.count(term)
    )
    return QUIET_SUMMARY.format(count=len(terms), breakdown=breakdown)


def render_details(
    verdict: Verdict,
    doc: Document,
    unregistered: Sequence[UnregisteredCheck],
    recognizer: PiiRecognizer | None = None,
    scoring: str = "",
) -> str:
    """偵測細節：預設收合，命中與未開啟的項目在第一層，未命中的再收一層。

    **一項都不丟。** 「系統檢查過什麼」是可稽核性的一部分，看得見「跑過了沒命中」
    才知道系統真的看過。但 27 項裡通常只有一兩項有話要說，平鋪 26 個「未命中」
    會讓使用者先讀完一整排看不懂的東西才找得到那一行。

    本區塊不重述判定：沒有信心等級的標籤、沒有詐騙類型，只有中文名、識別字、
    狀態、原文引用與計分的數值。`scoring` 那一行是**數值**不是判定 ——
    它與逐項訊號放在一起才有脈絡，而卡片第一層已經印過的東西不會再印一次。
    """
    hits = [result for result in verdict.checks if result.hit]
    quiet = [result for result in verdict.checks if not result.hit]
    total = len(verdict.checks) + len(unregistered)

    rows: list[str] = [scoring]
    if doc.truncated:
        rows.append(
            f'<div class="truncation">{TRUNCATION_LINE.format(dropped=doc.dropped_messages)}</div>'
        )
    for result in hits:
        title, notes = split_detail(result.detail)
        body = "".join(f'<div class="item-note">{escaped(note)}</div>' for note in notes)
        rows.append(
            _item(title, result.name, check_state(result), body + render_quote(result, doc))
        )
    for name, label, reason in unregistered:
        rows.append(
            _item(label, name, TERM_DISABLED, f'<div class="item-note">{escaped(reason)}</div>')
        )
    if quiet:
        terms = [check_state(result) for result in quiet]
        inner = "".join(_quiet_item(result.name, term) for result, term in zip(quiet, terms))
        rows.append(
            f'<details class="quiet"><summary>{_quiet_summary(terms)}</summary>{inner}</details>'
        )

    summary = (
        DETAILS_SUMMARY_WITH_HITS.format(total=total, hits=len(hits))
        if hits
        else DETAILS_SUMMARY.format(total=total)
    )
    return (
        f'<details class="details"><summary>{summary}</summary>'
        f"{''.join(rows)}{render_pii_block(doc, recognizer)}</details>"
    )


# ---------------------------------------------------------------------------
# 累積命中排行
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingRow:
    """排行的一列。`count` 為 `None` 代表這個訊號命中但沒有句子位置。"""

    name: str
    title: str
    term: str
    count: int | None


TERM_ORDER: Mapping[str, int] = MappingProxyType({TERM_CONCLUSIVE: 0, TERM_INDICATIVE: 1})
"""排序時 `term` 的先後。Conclusive 在前 —— 它單獨足以下結論，
出現一次的份量大於 Indicative 出現五次，用次數混排會讓弱訊號靠刷次數爬到前面。"""


def ranking_key(row: RankingRow) -> tuple[int, int, str]:
    """排行的排序鍵：先 term、再次數降序、最後識別字升序。

    定義在模組層而非以 closure 傳入：本專案禁止巢狀 `def` 與 closure。
    第三段鍵不可省 —— 沒有它，同分列的順序由 `build_registry()` 的註冊順序決定，
    測試寫不穩。
    """
    return (TERM_ORDER[row.term], -row.count if row.count is not None else 1, row.name)


def hit_ranking(checks: Sequence[CheckResult]) -> list[RankingRow]:
    """命中訊號的排行。**次數是相異座標數，也就是命中的句子數。**

    `Verdict` 是對整段對話算的，不是逐輪的，所以「輪數」這個單位根本取不到 ——
    `Verdict.checks` 裡沒有任何欄位指向輪次。句子數取得到而且是對的單位：
    同一輪裡兩句都命中 `safe_account`，那是兩個證據。

    同名的多筆結果先合併（一個檢查可以產出多筆），取座標的**聯集**而不是加總 ——
    同一個座標被同一條規則報兩次是同一個證據，理由與 `confidence._hit_groups()`
    用群組數而不是筆數相同。

    命中但無座標的訊號照常列出（`types.py` 明文「空 `evidence` 是合法值，
    此時 `hit` 仍可為 True」）：把它算成 1 是把兩個不同的單位放在同一根長條上，
    濾掉它是讓一個真的命中消失。

    排序：`Conclusive` 全部在前（單獨足以下結論，出現一次的份量大於 `Indicative`
    出現五次）、級內依次數降序、同分時依識別字升序。最後一段是為了測試寫得穩 ——
    沒有它，同分項目的順序由註冊順序決定，而那個順序會因 `build_registry()`
    多一行而改變。
    """
    coords: dict[str, set[Coord]] = {}
    seen: dict[str, CheckResult] = {}
    for result in checks:
        if not result.hit:
            continue
        coords.setdefault(result.name, set()).update(result.evidence)
        seen.setdefault(result.name, result)

    rows = [
        RankingRow(
            name=name,
            title=split_detail(seen[name].detail)[0],
            term=check_state(seen[name]),
            count=len(coords[name]) or None,
        )
        for name in seen
    ]
    return sorted(rows, key=ranking_key)


def bar_width(count: int | None, largest: int) -> float:
    """長條的寬度百分比。最大者滿格，其餘按比例，命中但無座標者為空。

    **不以次數當絕對寬度** —— 一次命中不該是一個像素，而且那樣會讓長條在一則
    短訊息上全部縮成看不見。
    """
    if count is None:
        return 0.0
    return max(MIN_BAR_WIDTH, 100.0 * count / largest)


def render_ranking(checks: Sequence[CheckResult], turns: int) -> str:
    """對練模式的累積命中排行，位於判定卡與對話之間。

    這個元件與受害方的回應分工：排行是**全部、按量**，受害方是**一條、按時序**。
    兩者不重複。

    輪數是**使用者送出的訊息數**，不是 `Document` 的訊息數：受害方的回應不回灌
    `Request`，而 `doc` 在截斷發生時訊息數會少於送出數。

    零命中時顯示一句話，不顯示空框，也不列出 27 個 0 —— 那是偵測細節的工作。
    """
    rows = hit_ranking(checks)
    head = f'<div class="ranking-head">{RANKING_HEADING.format(turns=turns)}</div>'
    if not rows:
        return (
            f'<div class="ranking">{head}'
            f'<p class="ranking-empty">{RANKING_EMPTY.format(turns=turns)}</p></div>'
        )
    counted = [row.count for row in rows if row.count is not None]
    largest = max(counted) if counted else 1
    body = "".join(
        f'<div class="rank-row"><span class="rank-name">{escaped(row.title)}</span>'
        f'<span class="rank-bar"><i style="width:{bar_width(row.count, largest):.0f}%"></i></span>'
        f'<span class="rank-count">'
        f"{RANKING_NO_COORD if row.count is None else row.count}</span></div>"
        for row in rows
    )
    return f'<div class="ranking">{head}{body}</div>'


# ---------------------------------------------------------------------------
# 受害方的回應
# ---------------------------------------------------------------------------


def victim_reply(verdict: Verdict, spoken: Sequence[str]) -> tuple[str, list[str]]:
    """受害方這一輪說的話，以及更新後的「已說過」清單。

    **每輪至多一條新依據。** `Verdict` 是對整段對話算的，所以把三段全說出來的
    實作會在第二輪逐字重複第一輪的八行 —— 畫面上是兩面一模一樣的文字牆。

    取「第一個沒說過的」就是取最強的那一個，**不需要新的排序**：
    `render_evidence()` 已經依群組貢獻降序排好，矛盾與截斷兩句附在最後。
    在這裡重排一次就是第二個排序實作。

    沒有新依據時說 `NO_NEW_SIGNAL` 且**不**把它記進 `spoken`：它陳述的是
    「這一輪沒有新東西」，而那件事本來就會重複發生。不重複的對象是依據行。
    """
    for line in verdict.evidence:
        if line in spoken:
            continue
        parsed = split_evidence_line(line)
        parts = [parsed.title, *parsed.notes[:1]]
        return "。".join(parts) + "。", [*spoken, line]
    return NO_NEW_SIGNAL, list(spoken)


# ---------------------------------------------------------------------------
# 個人資料標示
# ---------------------------------------------------------------------------


def pii_label(entity_type: str) -> str:
    """類型識別字對應的中文顯示名。**不存在時拋 `KeyError`，不回退。**

    `PII_LABELS.get(entity_type, entity_type)` 會把「掛上了一個不合契約的辨識器」
    變成「畫面上偶爾出現一個英文字串」，而後者沒有人會回報。
    例外訊息帶上那個值，讓它自己說出是誰不合契約。
    """
    if entity_type not in PII_LABELS:
        raise KeyError(f"未知的個資類型，沒有顯示名可用：entity_type={entity_type}")
    return PII_LABELS[entity_type]


def check_pii_spans(text: str, spans: Sequence[PiiSpan]) -> None:
    """驗證區間落在 `text` 內、互不重疊且依序遞增。違反時拋 `ValueError`。

    **不排序後硬吃、不略過、不截斷。** 三者都會把「產生區間的元件算錯了」
    變成「標註偶爾標到隔壁幾個字」，而後者不拋例外、畫面上看起來只是標得怪，
    在一個 recall 本來就低的功能上沒有人看得出來。

    這裡的區間**已經解析到 `text` 的座標系上** —— 正規化句子上的區間直接來自
    辨識器，原文片段上的區間來自 `Document.raw_bounds_at()`。
    """
    previous: PiiSpan | None = None
    for span in spans:
        start, end, _entity_type = span
        if start < 0:
            raise ValueError(f"標註起點不可為負：start={start}")
        if start >= end:
            raise ValueError(f"標註區間必須非空：start={start}、end={end}")
        if end > len(text):
            raise ValueError(f"標註終點不可超過文字長度：end={end}、長度={len(text)}")
        if previous is not None and start < previous[1]:
            raise ValueError(
                f"標註區間不可重疊：前一段 start={previous[0]}、end={previous[1]}，"
                f"這一段 start={start}、end={end}"
            )
        previous = span


def render_pii_highlight(text: str, spans: Sequence[PiiSpan]) -> str:
    """在 `text` 上做字元層級標註。`spans` 的座標**必須已經解析到 `text` 上**。

    正規化句子與原文片段共用這一份實作 —— 兩者的差別只在區間怎麼算出來，
    標記怎麼組是同一件事。

    ⚠️ **先用原始索引切片，再對每一段各自逸出。** `html.escape()` 是一對多的
    （`&` → `&amp;` 是 1→5），先逸出整句再用區間切片，從第一個 `&` 之後的所有
    索引全部位移，標註會標到隔壁幾個字 —— 而 `&` 在帶追蹤參數的釣魚連結裡
    幾乎必然出現。這個錯不會拋例外，畫面上只是標到錯的字。

    **類型是一個文字節點，不是 `title` 也不是 `aria-label`。** 觸控裝置上
    `:hover` 由瀏覽器自行決定行為、`title` 完全不顯示，而手機是這個服務的主要
    載體；`aria-label` 掛在 `<mark>` 上會**取代**內部文字，報讀器會唸出
    「身分證字號」而不唸出號碼本身。`title` 保留，但只是指標裝置上的冗餘。

    **串接點不引入任何原文沒有的字元。** `.lines` 是 `white-space: pre-wrap`，
    在那個容器裡 HTML 原始碼的換行與縮排會被渲染成空白。
    """
    check_pii_spans(text, spans)
    pieces: list[str] = []
    cursor = 0
    for start, end, entity_type in spans:
        label = escaped(pii_label(entity_type))
        pieces.append(escaped(text[cursor:start]))
        pieces.append(
            f'<mark class="pii-mark" title="{label}">{escaped(text[start:end])}'
            f'<span class="pii-kind">{label}</span></mark>'
        )
        cursor = end
    pieces.append(escaped(text[cursor:]))
    return "".join(pieces)


def render_raw_sentence(doc: Document, coord: Coord, recognizer: PiiRecognizer | None) -> str:
    """一句**原文片段**，個資的位置與類型標在上面。未掛載辨識器時只做逸出。

    座標換算走 `Document.raw_bounds_at()`，**絕不以 `find()` 回頭搜尋原文**：
    辨識器看到的是正規化句子，氣泡顯示的是原文，兩者的索引不同。NFKC 有一對多
    （`㈱` → `(株)`）也有多對一（`\\r\\n` → `\\n`），全形數字或零寬字元時
    `raw.find()` 必然搜不到 —— 實測全形身分證的正規化區間是 `(37, 47)` 而
    `find()` 回 `-1`。那條路的後果不是例外，是標註在被規避手法改寫過的訊息上
    安靜消失，而那正是 `add-evasion-check` 在偵測的那些訊息。

    **不 catch `raw_bounds_at()` 的 `KeyError` 與 `ValueError`。** 能觸發它們的
    只有兩件事：掛上了不合契約的辨識器，或 `Document` 的偏移表算錯 ——
    兩者都是程式錯誤，與 `render_quote()` 對無效座標的處置一致。
    """
    raw = doc.raw_at(coord)
    if recognizer is None:
        return escaped(raw)
    spans = [
        (*doc.raw_bounds_at(coord, start, end), entity_type)
        for start, end, entity_type in recognizer(doc.text_at(coord))
    ]
    return render_pii_highlight(raw, spans)


def pii_conversation_note(doc: Document, recognizer: PiiRecognizer | None) -> str:
    """對話區塊末尾的個資說明。**與標註同層，不收合。**

    標註一旦出現在氣泡上，使用者的推論路徑會是「系統會標個資 → 它沒標 →
    這則沒個資」，而那個推論是錯的：姓名、地址、銀行帳號、護照號碼全部不在四條
    樣式裡。揭露因此必須看得見，不能留在 `render_pii_block()` 那個預設收合的
    `<details>` 裡。

    「沒有開啟」與「沒有找到」是兩個狀態，兩段文字不同且不同時出現：前者是
    「沒有人在看」，後者是「看過了，沒有」。
    """
    if recognizer is None:
        return f'<div class="note">{PII_NOT_MOUNTED}</div>'
    found = any(recognizer(sentence) for sentence in doc.sentences)
    no_hit = "" if found else PII_NO_HIT
    return f'<div class="note">{no_hit}{PII_SCOPE_NOTE}</div>'


def render_pii_block(doc: Document, recognizer: PiiRecognizer | None) -> str:
    """偵測細節裡的個資區塊。這裡的標示標在**正規化句子**上。

    **與氣泡上的標註不重複，兩者標的是不同的東西**：氣泡標原文（使用者寫的
    那個樣子），這裡標正規化後的句子（系統實際拿去比對的那個樣子），
    `PII_NORMALIZED_NOTE` 講的就是這個差別。一則以全形數字書寫的身分證，
    氣泡上標的是全形、這裡標的是半形。

    「沒有開啟」與「沒有找到」是兩個狀態，文字必須不同：前者是「沒有人在看」，
    後者是「看過了，沒有」。在一個以展示偵測能力為目的的畫面上混淆這兩者，
    剛好是最糟的謊。

    本路徑**不 import 也不呼叫任何遮蔽程式碼**：標示指出位置與類型、文字不變、
    呈現給使用者；遮蔽改寫文字、只有 log 能讀。兩條不同的路徑。
    """
    scope = f'<div class="note">{PII_SCOPE_NOTE}</div>'
    if recognizer is None:
        return (
            f'<div class="pii"><h4>{PII_HEADING}</h4>'
            f'<div class="note">{PII_NOT_MOUNTED}</div>{scope}</div>'
        )
    rows: list[str] = []
    for coord, sentence in zip(doc.coords, doc.sentences):
        spans = recognizer(sentence)
        if not spans:
            continue
        rows.append(
            f'<div class="pii-row"><span class="coord">({coord[0]},{coord[1]})</span>'
            f"{render_pii_highlight(sentence, spans)}</div>"
        )
    if not rows:
        return (
            f'<div class="pii"><h4>{PII_HEADING}</h4>'
            f'<div class="note">{PII_NO_HIT}</div>{scope}</div>'
        )
    return (
        f'<div class="pii"><h4>{PII_HEADING}</h4>{"".join(rows)}'
        f'<div class="note">{PII_NORMALIZED_NOTE}</div>{scope}</div>'
    )


# ---------------------------------------------------------------------------
# 對話與輸入回顯
# ---------------------------------------------------------------------------


def render_message(
    message_index: int,
    message: Message,
    doc: Document,
    sender_label: str,
    recognizer: PiiRecognizer | None,
) -> str:
    """單一則訊息的氣泡，逐句渲染並帶錨點。

    座標不存在於 `Document` 的訊息標示為「未納入本次判定」—— 它還在畫面上，
    但系統其實沒讀到它。不顯示的話，使用者會以為系統看過全部。

    個資標在**字元上**，不在句尾掛類型計數標籤：字元層級的標註落地之後，
    句尾再掛一個「身分證字號 ×1」就是同一行裡把同一件事講兩次。
    """
    positions = doc.message_range(message_index)
    if not positions:
        return (
            '<div class="bubble them dropped">'
            f'<div class="who">{escaped(sender_label)} · 第 {message_index + 1} 則 · '
            f"{DROPPED_LABEL}</div>"
            f'<div class="lines">{escaped(message.text)}</div></div>'
        )
    parts = [
        f'<span class="sentence" id="{anchor_id(doc.coords[position])}">'
        f"{render_raw_sentence(doc, doc.coords[position], recognizer)}</span>"
        for position in positions
    ]
    return (
        '<div class="bubble them">'
        f'<div class="who">{escaped(sender_label)} · 第 {message_index + 1} 則</div>'
        f'<div class="lines">{"".join(parts)}</div></div>'
    )


def render_conversation(
    messages: Sequence[Message],
    replies: Sequence[str],
    doc: Document,
    sender_label: str,
    reply_label: str,
    recognizer: PiiRecognizer | None = None,
) -> str:
    """對話（對練模式）或輸入回顯（「這是詐騙嗎」模式）。

    用自繪的 HTML 而非 `gr.Chatbot`：後者的單位是一則訊息，而這裡需要的最小
    單位是**句子**（證據座標的粒度就是句子），還要在句子裡標出個資的位置。

    末尾的那一行是辨識範圍的揭露，**不收合** —— 見 `pii_conversation_note()`。
    """
    blocks: list[str] = []
    for message_index, message in enumerate(messages):
        blocks.append(render_message(message_index, message, doc, sender_label, recognizer))
        if message_index < len(replies):
            blocks.append(
                f'<div class="bubble me"><div class="who">{escaped(reply_label)}</div>'
                f'<div class="lines">{escaped(replies[message_index])}</div></div>'
            )
    return (
        f'<div class="conversation">{"".join(blocks)}{pii_conversation_note(doc, recognizer)}</div>'
    )


# ---------------------------------------------------------------------------
# 樣式表
# ---------------------------------------------------------------------------

THEME_VARIABLES = (
    "--body-text-color",
    "--body-text-color-subdued",
    "--background-fill-secondary",
    "--block-background-fill",
    "--border-color-primary",
    "--color-accent",
    "--color-accent-soft",
)
"""本樣式表用到的全部主題變數。

Gradio 端這七個已經隨明暗模式切換，不需要做任何事；`docs/index.html` 不是
Gradio，所以它在自己的 `:root` 定義同名的七個並以 `prefers-color-scheme`
覆寫一次 —— 寫死的色碼因此全部集中在那兩個區塊裡。

⚠️ 樣式表 MUST NOT 出現任何寫死的色碼，也 MUST NOT 用 `var(--x, #fallback)`
的預設值形式：那個 fallback 就是一個躲在括號裡的色碼。gradio 6 的明暗模式
**鎖不住**（使用者可在頁尾的設定面板自己切，設定 persist 在瀏覽器，
Python 端沒有任何寫法能覆寫它），所以「只支援淺色」不是一個可以選的選項。
"""

CSS = """
/* 顏色一律取自主題變數。寫死的色碼會在深色模式下變成淺字寫在淺底上 ——
   那是在開發者的淺色螢幕上永遠看不到的錯。 */

.sg-head h1 { font-size: 1.5rem; font-weight: 700; margin: 0 0 .3rem;
  color: var(--body-text-color); }
.sg-head p { margin: .2rem 0; font-size: .92rem; line-height: 1.7;
  color: var(--body-text-color); }

/* 判定卡 */
.card { border: 1px solid var(--border-color-primary); border-radius: 14px;
  padding: 1rem 1.2rem; background: var(--background-fill-secondary);
  color: var(--body-text-color); }
.card-head { display: flex; flex-wrap: wrap; gap: .6rem 1rem; align-items: baseline;
  justify-content: space-between; }
.card-title { font-size: 1.35rem; font-weight: 700; line-height: 1.4; }
.card-facts { display: flex; flex-wrap: wrap; gap: .5rem; align-items: baseline; }
.fact-main { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
.fact { font-size: .82rem; padding: .05rem .5rem; border-radius: 999px;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color); }
.card-lead { margin: .5rem 0 0; font-size: .95rem; line-height: 1.75; }
.card h4 { font-size: .85rem; font-weight: 700; margin: 0 0 .4rem;
  color: var(--body-text-color-subdued); letter-spacing: .04em; }
.why, .todo { margin-top: 1rem; }
.why ul, .todo ul { margin: 0; padding: 0; list-style: none; }
.why li { padding-left: .9rem; border-left: 2px solid var(--color-accent);
  margin-bottom: .55rem; }
.why-title { font-size: .98rem; font-weight: 600; line-height: 1.6; }
.why-note { font-size: .85rem; line-height: 1.65; color: var(--body-text-color-subdued); }
.todo li { font-size: .95rem; line-height: 1.7; padding-left: .9rem; position: relative; }
.todo li::before { content: "→"; position: absolute; left: 0;
  color: var(--color-accent); }

/* 分數刻度尺 —— 門檻標在它真正的位置上，軸的右界超過門檻，
   所以它讀起來是「分數在門檻左邊」而不是「進度條快滿了」。 */
.scale { margin-top: .9rem; }
.scale-labels { display: flex; gap: 1.2rem; font-size: .85rem;
  color: var(--body-text-color); }
.scale-score { font-weight: 700; }
.scale-gate { color: var(--body-text-color-subdued); }
.scale-axis { position: relative; height: .5rem; margin: .35rem 0 .2rem;
  border-radius: 999px; background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary); }
.scale-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;
  background: var(--color-accent); }
.scale-mark { position: absolute; top: -.25rem; bottom: -.25rem; width: 2px;
  background: var(--body-text-color); }
.scale-ticks { position: relative; height: 1rem; font-size: .7rem;
  color: var(--body-text-color-subdued); }
.scale-ticks span { position: absolute; left: 0; transform: translateX(-50%); }
.scale-ticks span:first-child { transform: none; }
.scoring { font-family: ui-monospace, monospace; font-size: .76rem; margin-bottom: .5rem;
  color: var(--body-text-color-subdued); }

/* 偵測細節 */
.details { margin-top: 1.1rem; padding-top: .7rem;
  border-top: 1px solid var(--border-color-primary); }
.details > summary, .quiet > summary { cursor: pointer; font-size: .82rem;
  color: var(--body-text-color-subdued); }
.details[open] > summary { margin-bottom: .6rem; }
.item { border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: .45rem .65rem; margin-bottom: .35rem; background: var(--block-background-fill); }
.item-conclusive { border-left: 3px solid var(--color-accent);
  background: var(--color-accent-soft); }
.item-indicative { border-left: 3px solid var(--color-accent); }
.item-head { display: flex; flex-wrap: wrap; gap: .45rem; align-items: baseline; }
.item-title { font-size: .88rem; font-weight: 600; color: var(--body-text-color); }
.item-id { font-family: ui-monospace, monospace; font-size: .7rem;
  color: var(--body-text-color-subdued); }
.item-id-solo { font-size: .78rem; }
.item-term { font-size: .68rem; margin-left: auto; padding: 0 .35rem;
  border: 1px solid var(--border-color-primary); border-radius: 4px;
  color: var(--body-text-color-subdued); }
.item-note { font-size: .8rem; line-height: 1.6; margin-top: .2rem;
  color: var(--body-text-color-subdued); }
.quote-list { display: flex; flex-direction: column; gap: .15rem; margin-top: .3rem; }
a.quote { font-size: .8rem; line-height: 1.6; color: var(--color-accent);
  word-break: break-all; }
.quiet { margin-top: .5rem; }
.quiet[open] > summary { margin-bottom: .4rem; }
.quiet .item { background: transparent; padding: .3rem .6rem; }
.truncation { border: 1px dashed var(--border-color-primary); border-radius: 6px;
  padding: .35rem .6rem; margin-bottom: .5rem; font-size: .82rem;
  color: var(--body-text-color); }

/* 累積命中排行 */
.ranking { border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: .8rem 1rem; background: var(--block-background-fill); }
.ranking-head { font-size: .85rem; font-weight: 700; margin-bottom: .55rem;
  color: var(--body-text-color); }
.ranking-empty { margin: 0; font-size: .88rem; color: var(--body-text-color-subdued); }
.rank-row { display: flex; align-items: center; gap: .6rem; margin-bottom: .3rem; }
.rank-name { flex: 0 1 16rem; font-size: .85rem; color: var(--body-text-color); }
.rank-bar { flex: 1 1 6rem; height: .5rem; border-radius: 999px;
  background: var(--background-fill-secondary); overflow: hidden; }
.rank-bar i { display: block; height: 100%; background: var(--color-accent); }
.rank-count { flex: 0 0 2rem; text-align: right; font-size: .82rem;
  font-variant-numeric: tabular-nums; color: var(--body-text-color-subdued); }

/* 對話氣泡 ——
   `.bubble.them` 是**偵測語義**（被檢查的那一方），與版面的左右無關。
   使用者打的字與貼上的訊息都是 them，靠右；系統扮演的受害方是 me，靠左。
   class 名稱與 `Message.sender` 的值不要「順手改回來」。 */
.conversation { display: flex; flex-direction: column; gap: .55rem; }
.bubble { border-radius: 12px; padding: .5rem .8rem; max-width: 88%;
  border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); color: var(--body-text-color); }
.bubble.them { align-self: flex-end; background: var(--color-accent-soft); }
.bubble.me { align-self: flex-start; }
.bubble.dropped { opacity: .6; border-style: dashed; background: transparent; }
.who { font-size: .7rem; color: var(--body-text-color-subdued); margin-bottom: .2rem; }
.lines { white-space: pre-wrap; line-height: 1.75; }
.sentence { padding: 1px 2px; border-radius: 3px; }
.sentence:target { background: var(--color-accent-soft);
  outline: 2px solid var(--color-accent); }

/* 個人資料標示 */
.pii { margin-top: .7rem; padding-top: .6rem;
  border-top: 1px solid var(--border-color-primary); }
.pii h4 { font-size: .8rem; }
.pii-mark { background: var(--color-accent-soft); color: var(--body-text-color);
  border-bottom: 2px solid var(--color-accent); border-radius: 3px;
  padding: 0 2px; }
/* 類型標籤**永遠可見**。縮小、弱化，但始終在文件流裡 ——
   觸控裝置上 `:hover` 要先點一下且會吃掉該次點擊，`title` 完全不顯示，
   而手機是這個服務的主要載體。把類型藏在 hover 後面等於在主要載體上刪掉它。 */
.pii-kind { font-size: .7rem; margin-left: .2rem;
  color: var(--body-text-color-subdued); }
/* hover 只做強調：改顏色與底線粗細。
   MUST NOT 改 display / visibility / opacity —— 那會讓標籤變成 hover 限定，
   正是上面那條規則反對的事。 */
.pii-mark:hover { border-bottom-width: 3px; }
.pii-mark:hover .pii-kind { color: var(--body-text-color); }
.pii-row { font-family: ui-monospace, monospace; font-size: .8rem; margin: .25rem 0;
  color: var(--body-text-color); word-break: break-all; }
.coord { color: var(--body-text-color-subdued); margin-right: .4rem; }

/* 範例標籤與頁尾 */
.chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.sg-foot { margin-top: 1.2rem; padding-top: .7rem;
  border-top: 1px solid var(--border-color-primary); }
.sg-foot summary { cursor: pointer; font-size: .75rem;
  color: var(--body-text-color-subdued); }
.note { font-size: .78rem; line-height: 1.7; color: var(--body-text-color-subdued);
  margin-top: .3rem; }
.note b { color: var(--body-text-color); }
.note ul { margin: .3rem 0; padding-left: 1.1rem; }
.note a { color: var(--color-accent); }
.notice { border: 1px solid var(--color-accent); border-radius: 8px;
  background: var(--color-accent-soft); color: var(--body-text-color);
  padding: .5rem .8rem; font-weight: 700; margin-bottom: .6rem; }
"""

COLOUR_LITERAL = re.compile(
    r"#(?:[0-9A-Fa-f]{3,4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})\b|rgba?\(|hsla?\("
)
"""寫死色彩值的形狀。`tests/test_demo_ui.py` 拿它掃 `CSS`。

放在這裡而不是只放在測試檔：它是這條界線的定義，而定義與被檢查的對象放在
同一個模組裡，改樣式表的人看得到它。
"""
