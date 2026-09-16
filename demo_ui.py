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

**persona 生成層的驗證住在這裡，理由與標記層相同。** `PolishValidator` 與它的兩個
允許集合原本在 `app.py`，而兩個載體（Gradio 與瀏覽器）都需要它 —— 兩份實作各改
一次，第二次就是它的利息。它不碰任何介面框架，搬家是機械的。
**它驗證的是輸出，不產生輸出**：產生那一段文字的是模型，而模型住在載體那一側。

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
from scam_guard.llm.validate import SCAM_SIGNAL, SUSPICIOUS_SIGNAL
from scam_guard.ngram import NGRAM_SIGNAL
from scam_guard.normalize import Document
from scam_guard.pipeline import NOT_HIT, SKIPPED
from scam_guard.render import QUOTE_SEPARATOR
from scam_guard.scoring import Score, compute_score, is_decision
from scam_guard.types import CheckResult, Coord, Message, ScamType, Verdict
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

PRACTICE_PERSONA = """你是一位角色名稱叫「善良市民」的普通台灣市民，
個性善良、有禮貌，也有基本的防詐意識。
遇到陌生或可疑的要求，你會禮貌但堅定地拒絕，並用一兩句話說出你為什麼覺得
不對勁。你不會辱罵對方，也不會長篇說教。你只根據系統已經看出來的線索起疑，
不會編造你不知道的細節，也不會提供任何個人資料、驗證碼、帳號或金錢。"""

PRACTICE_INSTRUCTIONS = """你是收到下面這則訊息的市民。
系統對這則訊息的偵測結果放在 `<detection>` 標籤裡。
請你以善良市民的口氣，禮貌但堅定地拒絕對方的要求，並用一兩句話說明你為什麼起疑。
只根據 `<detection>` 裡的線索，不要提到任何裡面沒有的數字、網址或詐騙類型。
只輸出你要說的那一兩句話。

{context}"""

PRACTICE_AVATAR_ALT = "善良市民的頭像"
SCAMMER_AVATAR_ALT = "邪惡詐騙犯的頭像"
DETECTION_PERCENTAGE = re.compile(r"\d+(?:\.\d+)?%")
DETECTION_URL = re.compile(
    r"(?i)(?:https?://|www\.)[^\s，。；）]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s，。；）]*)?"
)

LEAD_SIGNALS = "找到 {count} 項訊號，但它們加起來還不夠下結論"
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

QUIET_SUMMARY = "其餘 {count} 項沒有命中（{breakdown}）"
TRUNCATION_LINE = "訊息太長，最舊的 {dropped} 則沒有納入這次判定。"
DROPPED_LABEL = "未納入本次判定"

PII_NOT_MOUNTED = "這個版本沒有開啟個人資料標註。"

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


# ---------------------------------------------------------------------------
# 四個偵測源 —— `verdict.checks` 的重新分組，不改任何判定邏輯
# ---------------------------------------------------------------------------

URL_PREFIX = "url_"
"""網址層所有檢查的共同前綴（`url_blocklist`、`url_shortener`、`url_tld_risk`…）。

判別走前綴而非一張名單：URL 層日後新增檢查時自動落進網址源，面板不必改。
"""

SEMANTIC_SIGNALS = frozenset({SCAM_SIGNAL, SUSPICIOUS_SIGNAL})
"""語意源的兩個成員名。取自核心常數而非字面值 —— 常數改名時這裡是 import 失敗，
不是面板上安靜地把語意層歸錯源。"""

SOURCE_URL = "網址"
SOURCE_RULE = "規則"
SOURCE_CLASSIFIER = "分類器"
SOURCE_SEMANTIC = "語意"

STATE_HIT = "命中"
STATE_CLEAR = "未命中"
STATE_SKIP = "未執行"

STATE_CLASS: Mapping[str, str] = MappingProxyType(
    {STATE_HIT: "hit", STATE_CLEAR: "clear", STATE_SKIP: "skip"}
)
"""三態對應的 CSS class 後綴。三態的區分不只靠顏色 —— 狀態字本身就是文字。"""


@dataclass(frozen=True)
class Source:
    """一個偵測源的宣告：顯示名，以及命中時是否附相異檢查數。

    `counts` 只對可能有多個成員的源（網址、規則）為真：分類器與語意各只有一個
    成員名，`命中 N 項` 的 N 恆為 1，是廢話，不顯示。
    """

    label: str
    counts: bool


SOURCES: tuple[Source, ...] = (
    Source(SOURCE_URL, True),
    Source(SOURCE_RULE, True),
    Source(SOURCE_CLASSIFIER, False),
    Source(SOURCE_SEMANTIC, False),
)
"""四個源的固定顯示順序：網址 → 規則 → 分類器 → 語意。

順序是宣告的不是算的，渲染照序取，不需要排序鍵。
"""


def source_of(name: str) -> str:
    """把一筆 `CheckResult.name` 判到恰好一個源。「規則」是殘量。

    判別走穩定識別字（`url_` 前綴、分類器訊號名、語意訊號名），不是一張規則名單：
    言語行為規則每落地一條就多一個 `name`，殘量分類讓新規則自動落進規則源。
    """
    if name.startswith(URL_PREFIX):
        return SOURCE_URL
    if name == NGRAM_SIGNAL:
        return SOURCE_CLASSIFIER
    if name in SEMANTIC_SIGNALS:
        return SOURCE_SEMANTIC
    return SOURCE_RULE


@dataclass(frozen=True)
class SourceStatus:
    """一個源在畫面上的一格：顯示名、三態狀態、命中的相異檢查數、是否顯示計數。"""

    label: str
    state: str
    count: int
    counts: bool


def source_statuses(
    checks: Sequence[CheckResult], unregistered: Sequence[UnregisteredCheck]
) -> list[SourceStatus]:
    """把 `checks` 與 `unregistered` 重新分組成四個源的狀態。

    三態聚合（優先序）：任一成員命中（`Conclusive`/`Indicative`）為命中；否則任一
    成員跑過未命中（`Clear`）為未命中；否則未執行（`Skipped`、或只在 `unregistered`
    而不在 `checks`、或該源根本沒有成員）。短路與未掛載都落在未執行，不顯示為未命中。

    語意源的未執行要判對，同時看 `checks` 與 `unregistered`：`unregistered` 的成員
    一律計為未執行（它們沒有跑），不改變已由 `checks` 決定的命中或未命中。

    計數用相異 `hit=True` 的 `name` 數（`hit_ranking()` 那條長條數的是命中句子數，
    兩者不同，混用會讓同一個數字在兩處指不同的東西）。
    """
    terms: dict[str, list[str]] = {source.label: [] for source in SOURCES}
    hit_names: dict[str, set[str]] = {source.label: set() for source in SOURCES}
    for result in checks:
        label = source_of(result.name)
        terms[label].append(check_state(result))
        if result.hit:
            hit_names[label].add(result.name)
    for name, _label, _reason in unregistered:
        terms[source_of(name)].append(TERM_SKIPPED)

    statuses: list[SourceStatus] = []
    for source in SOURCES:
        member_terms = terms[source.label]
        if any(term in (TERM_CONCLUSIVE, TERM_INDICATIVE) for term in member_terms):
            state = STATE_HIT
        elif TERM_CLEAR in member_terms:
            state = STATE_CLEAR
        else:
            state = STATE_SKIP
        statuses.append(
            SourceStatus(
                label=source.label,
                state=state,
                count=len(hit_names[source.label]),
                counts=source.counts,
            )
        )
    return statuses


def render_source_grid(
    checks: Sequence[CheckResult], unregistered: Sequence[UnregisteredCheck]
) -> str:
    """四源狀態列：四格平鋪，每格顯示源名與三態；網址與規則命中時附「命中 N 項」。

    狀態字本身（未執行 / 未命中 / 命中）就是文字，顏色是冗餘強化不是唯一載體 ——
    深淺兩色下都讀得出，色盲也讀得出。
    """
    cells: list[str] = []
    for status in source_statuses(checks, unregistered):
        count = (
            f'<span class="source-count">命中 {status.count} 項</span>'
            if status.state == STATE_HIT and status.counts
            else ""
        )
        cells.append(
            f'<div class="source source-{STATE_CLASS[status.state]}">'
            f'<span class="source-name">{status.label}</span>'
            f'<span class="source-state">{status.state}</span>{count}</div>'
        )
    return f'<div class="source-grid">{"".join(cells)}</div>'


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


def build_detection_context(verdict: Verdict, table: WeightTable) -> str:
    """把確定性偵測結果組成 persona 可見的 XML；不帶原文、機率或信心。"""
    lines = [
        "<detection>",
        f"  <verdict>系統判定：{html.escape(verdict_state(verdict, table))}</verdict>",
    ]
    if verdict.scam_type is not None:
        lines.append(f"  <type>{html.escape(verdict.scam_type.value)}</type>")
    lines.append("  <signals>")
    lines.extend(
        f"    <signal>{html.escape(detection_signal_title(line))}</signal>"
        for line in verdict.evidence
    )
    lines.extend(("  </signals>", "</detection>"))
    return "\n".join(lines)


def detection_signal_title(evidence_line: str) -> str:
    """取依據標題並移除可能嵌在確定性 detail 裡的 URL 與百分比。"""
    title = split_evidence_line(evidence_line).title
    without_urls = DETECTION_URL.sub("[網址已省略]", title)
    return DETECTION_PERCENTAGE.sub("[比率已省略]", without_urls)


def practice_prompt(verdict: Verdict, table: WeightTable) -> str:
    """組出善良市民的 system persona 與只含偵測結果的生成指引。"""
    return f"{PRACTICE_PERSONA}\n\n{practice_instructions(verdict, table)}"


def practice_instructions(verdict: Verdict, table: WeightTable) -> str:
    """組出 persona 的 user 指引；瀏覽器側會另以 system role 傳入 persona。"""
    context = build_detection_context(verdict, table)
    return PRACTICE_INSTRUCTIONS.format(context=context)


def verdict_segments(verdict: Verdict) -> tuple[str, ...]:
    """可供 persona 引用的全部確定性素材；刻意去掉每一行的原文引用。"""
    segments: list[str] = []
    for line in verdict.evidence:
        parsed = split_evidence_line(line)
        segments.extend((parsed.title, *parsed.notes))
    return tuple(segments)


# ---------------------------------------------------------------------------
# 判定卡
# ---------------------------------------------------------------------------


def _facts(verdict: Verdict, state: str, table: WeightTable) -> str:
    """標題列右側的量。

    三個狀態各自帶不同的量，而這不是版面偷懶：
    「很可能是詐騙」帶機率、類型與信心等級 —— 系統做了判定，三者都是那個判定的一部分。
    「有可疑訊號」帶目前的機率、類型與信心等級，但標題仍由判定門檻決定；
    刻度尺只把同一個機率圖形化，不另造級距。
    「無法判定」什麼都不帶。
    """
    band = confidence_band(verdict.confidence, table)
    cells: list[str] = []
    if state != TITLE_UNDECIDED and verdict.scam_probability is not None:
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


def render_score_scale(score: Score) -> str:
    """把同一個機率畫成刻度尺，不另外印第二份百分比。

    `width` 是同一個機率的圖形屬性，不是文字節點；尺上不寫數字、不劃級距，
    避免把未校準的機率包裝成另一套風險分類。
    """
    probability = min(max(score.probability, 0.0), 1.0)
    return (
        '<div class="scale" role="img" aria-label="機率刻度尺">'
        '<div class="scale-axis">'
        f'<i class="scale-fill" style="width:{100.0 * probability:.1f}%"></i></div>'
        "</div>"
    )


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
) -> str:
    """判定卡 —— 判定結果在畫面上的**唯一**出處。

    取代了原本的「判定列 + 判定結果散文段 + 右欄面板」三個區塊：同一則假檢警
    訊息原本會在三個地方各講一次機率、信心與類型。
    """
    return (
        '<div class="demo-result">'
        f"{render_analysis_panel(verdict, table, unregistered)}"
        f"{render_detection_details(verdict, doc, unregistered)}"
        "</div>"
    )


def split_result(rendered: str) -> tuple[str, str]:
    """把共用完整輸出投影到載體的全寬面板與右欄，不複製畫面上的事實。"""
    prefix = '<div class="demo-result">'
    divider = '</section><section class="detection-detail">'
    suffix = "</section></div>"
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
        raise ValueError("共用判定標記缺少 demo-result 外框")
    body = rendered.removeprefix(prefix).removesuffix(suffix)
    panel, separator, details = body.partition(divider)
    if not separator:
        raise ValueError("共用判定標記缺少分析面板或偵測細節")
    return f"{panel}</section>", f'<section class="detection-detail">{details}</section>'


def render_analysis_panel(
    verdict: Verdict, table: WeightTable, unregistered: Sequence[UnregisteredCheck]
) -> str:
    """全寬判定摘要：三態、四源狀態列、唯一一份機率、信心、類型與圖形刻度。

    四源列插在 `card-head`（三態標題 + 機率 + 信心 + 類型）之後、刻度尺之前 ——
    上方那個三態標題就是「綜合判定」，四源列不另印第二份，否則機率/類型/信心會
    多出一次，`redo-demo-layout` 的「各恰好一次」性質會紅。
    """
    state = verdict_state(verdict, table)
    score = None if verdict.scam_probability is None else compute_score(verdict.checks, table)
    scale = "" if score is None else render_score_scale(score)
    return (
        '<section class="analysis-panel card">'
        f'<div class="card-head"><div class="card-title">{escaped(state)}</div>'
        f"{_facts(verdict, state, table)}</div>"
        f"{render_source_grid(verdict.checks, unregistered)}"
        f"{scale}"
        f"{_lead(verdict, state)}"
        "</section>"
    )


def render_detection_details(
    verdict: Verdict,
    doc: Document,
    unregistered: Sequence[UnregisteredCheck],
) -> str:
    """右欄內容；判定摘要的機率、信心與類型不在這裡重複。

    不再重複個資：訊息氣泡上已有 inline 的字元標註（`render_message` 那條路徑），
    右欄再標一次同一批個資是同一事實出現兩次。順序因此是「命中的檢查 → 為什麼 →
    完整檢查」。
    """
    hits = [result for result in verdict.checks if result.hit]
    hit_rows = []
    for result in hits:
        title, notes = split_detail(result.detail)
        body = "".join(f'<div class="item-note">{escaped(note)}</div>' for note in notes)
        hit_rows.append(
            _item(title, result.name, check_state(result), body + render_quote(result, doc))
        )
    hit_markup = "".join(hit_rows) or '<p class="detail-empty">這次沒有檢查命中。</p>'
    return (
        '<section class="detection-detail">'
        "<h3>命中的檢查</h3>"
        f'<div class="hit-checks">{hit_markup}</div>'
        f"{render_why(verdict)}"
        f"{render_details(verdict, doc, unregistered)}"
        f"{render_actions(verdict)}"
        "</section>"
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
) -> str:
    """偵測細節：預設收合，命中與未開啟的項目在第一層，未命中的再收一層。

    **一項都不丟。** 「系統檢查過什麼」是可稽核性的一部分，看得見「跑過了沒命中」
    才知道系統真的看過。但 27 項裡通常只有一兩項有話要說，平鋪 26 個「未命中」
    會讓使用者先讀完一整排看不懂的東西才找得到那一行。

    本區塊不重述判定：沒有信心等級的標籤、沒有詐騙類型，只有中文名、識別字、
    狀態與原文引用。判定摘要的機率、信心分級與類型不在這裡重複。
    """
    hits = [result for result in verdict.checks if result.hit]
    quiet = [result for result in verdict.checks if not result.hit]
    total = len(verdict.checks) + len(unregistered)

    rows: list[str] = []
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

    summary = f"完整檢查（{total} 項）"
    return f'<details class="details"><summary>{summary}</summary>{"".join(rows)}</details>'


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
# persona 生成層的驗證 —— 三條前綴可判定的條件
# ---------------------------------------------------------------------------

DIGITS = re.compile(r"\d+")
URL_MARKERS = ("http", "www.")

POLISH_NOT_INJECTED = "這句回應直接來自判定結果，沒有經過語言模型生成。"
POLISH_STREAMING = "語言模型正在以善良市民的角色生成回應。"
POLISH_ACCEPTED = "善良市民的回應通過數字、網址與詐騙類型三項檢查。"
POLISH_DISCARDED = "生成內容加入了判定結果沒有的資訊，已整句丟棄，改用原本的回應。"
POLISH_FAILED = "語言模型生成失敗，已改用直接來自判定結果、未經模型生成的回應。"


def allowed_numbers(segments: Sequence[str]) -> frozenset[str]:
    """確定性層全部素材裡的連續數字。persona 只能使用這些數字。"""
    return frozenset(match.group() for part in segments for match in DIGITS.finditer(part))


def allowed_types(verdict: Verdict) -> frozenset[str]:
    """本次 `Verdict` 中出現過的 `ScamType` 值。

    含 `verdict.scam_type` 與命中檢查回報的 `scam_types` —— 兩者都是字面上
    「出現於 `Verdict`」的類型。其餘 `ScamType` 成員一律視為模型發明的。
    """
    values: set[str] = set()
    if verdict.scam_type is not None:
        values.add(verdict.scam_type.value)
    for result in verdict.checks:
        if result.hit:
            values.update(scam_type.value for scam_type in result.scam_types)
    return frozenset(values)


class PolishValidator:
    """persona 輸出的三條可驗證條件，逐字元判定。

    三條都是**前綴可判定的**：一旦違規字元出現，後續文字不可能讓它變回合規。
    因此驗證可以在串流過程中即時進行，違規即中止 —— 若等全文產生完才驗證，
    使用者已經看過那段唬爛了。前綴可判定讓「串流」與「fail-closed」並存。

    1. 數字：維護當前的連續數字串，它不是任何允許數字的前綴即違規。
    2. 網址：出現 `http` 或 `www.` 即違規（確定性層的輸出不含網址）。
    3. 類型：出現未在本次 `Verdict` 中的 `ScamType` 值即違規 —— 這擋掉最危險
       的那種唬爛：判定沒有類型，受害方卻說「這是假檢警」。

    已知代價：模型把「兩項訊號」寫成「2 項訊號」就會違規。接受 —— 這一層寧可
    退回模板句，也不要放行一個會講數字的模型，而且丟棄是可見的，不是靜默。
    """

    def __init__(self, segments: Sequence[str], verdict: Verdict) -> None:
        self._numbers = allowed_numbers(segments)
        self._forbidden_types = frozenset(
            scam_type.value for scam_type in ScamType
        ) - allowed_types(verdict)
        self._text = ""
        self._digits = ""

    @property
    def text(self) -> str:
        """目前為止已通過驗證的文字。"""
        return self._text

    def feed(self, chunk: str) -> bool:
        """餵入一段新產生的文字。回傳 `False` 代表違規，呼叫端 MUST 中止串流。"""
        for character in chunk:
            self._text += character
            if character.isdigit():
                self._digits += character
                if not any(number.startswith(self._digits) for number in self._numbers):
                    return False
            else:
                self._digits = ""
            for marker in URL_MARKERS:
                if self._text.endswith(marker):
                    return False
            for value in self._forbidden_types:
                if self._text.endswith(value):
                    return False
        return True


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


def pii_conversation_note(recognizer: PiiRecognizer | None) -> str:
    """對話區塊末尾的個資說明。

    辨識器未掛載時說明本版本沒有開啟個資標註（「沒有人在看」與「看過了沒有」是
    兩件事，前者仍需說明）。已掛載時不輸出任何說明：個資偵測結果由氣泡上的 inline
    標註（`render_pii_highlight`）承接，辨識範圍不再於畫面文字說明，無命中時也不輸出
    「沒有找到個資」——「沒標」與「沒有」的差別由標註本身呈現。
    """
    if recognizer is None:
        return f'<div class="note">{PII_NOT_MOUNTED}</div>'
    return ""


# ---------------------------------------------------------------------------
# 對話與輸入回顯
# ---------------------------------------------------------------------------


def render_message(
    message_index: int,
    message: Message,
    doc: Document,
    sender_label: str,
    recognizer: PiiRecognizer | None,
    sender_avatar: str | None = None,
) -> str:
    """單一則訊息的氣泡，逐句渲染並帶錨點。

    座標不存在於 `Document` 的訊息標示為「未納入本次判定」—— 它還在畫面上，
    但系統其實沒讀到它。不顯示的話，使用者會以為系統看過全部。

    個資標在**字元上**，不在句尾掛類型計數標籤：字元層級的標註落地之後，
    句尾再掛一個「身分證字號 ×1」就是同一行裡把同一件事講兩次。
    """
    avatar = (
        f'<img class="persona-avatar" src="{html.escape(sender_avatar, quote=True)}" '
        f'alt="{SCAMMER_AVATAR_ALT}">'
        if sender_avatar is not None
        else ""
    )
    positions = doc.message_range(message_index)
    if not positions:
        return (
            f'<div class="persona-sender"><div class="bubble them dropped">'
            f'<div class="who">{escaped(sender_label)} · 第 {message_index + 1} 則 · '
            f"{DROPPED_LABEL}</div>"
            f'<div class="lines">{escaped(message.text)}</div></div>{avatar}</div>'
        )
    parts = [
        f'<span class="sentence" id="{anchor_id(doc.coords[position])}">'
        f"{render_raw_sentence(doc, doc.coords[position], recognizer)}</span>"
        for position in positions
    ]
    return (
        f'<div class="persona-sender"><div class="bubble them">'
        f'<div class="who">{escaped(sender_label)} · 第 {message_index + 1} 則</div>'
        f'<div class="lines">{"".join(parts)}</div></div>{avatar}</div>'
    )


def render_conversation(
    messages: Sequence[Message],
    replies: Sequence[str],
    doc: Document,
    sender_label: str,
    reply_label: str,
    recognizer: PiiRecognizer | None = None,
    reply_avatar: str | None = None,
    sender_avatar: str | None = None,
) -> str:
    """對話（對練模式）或輸入回顯（「這是詐騙嗎」模式）。

    用自繪的 HTML 而非 `gr.Chatbot`：後者的單位是一則訊息，而這裡需要的最小
    單位是**句子**（證據座標的粒度就是句子），還要在句子裡標出個資的位置。

    末尾只在辨識器未掛載時附一行說明 —— 見 `pii_conversation_note()`。
    """
    blocks: list[str] = []
    for message_index, message in enumerate(messages):
        blocks.append(
            render_message(message_index, message, doc, sender_label, recognizer, sender_avatar)
        )
        if message_index < len(replies):
            avatar = (
                f'<img class="persona-avatar" src="{html.escape(reply_avatar, quote=True)}" '
                f'alt="{PRACTICE_AVATAR_ALT}">'
                if reply_avatar is not None
                else ""
            )
            blocks.append(
                f'<div class="persona-reply">{avatar}<div class="bubble me">'
                f'<div class="who">{escaped(reply_label)}</div>'
                f'<div class="lines">{escaped(replies[message_index])}</div></div></div>'
            )
    return f'<div class="conversation">{"".join(blocks)}{pii_conversation_note(recognizer)}</div>'


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

/* 共用產品版面：tab 由載體放在最上，分析面板全寬，下面兩欄等寬。 */
.demo-main { width: 100%; max-width: 72rem; margin: 0 auto; overflow-x: clip; }
.analysis-slot { width: 100%; margin: .8rem 0 1rem; }
.analysis-slot .detection-detail { display: none; }
.demo-columns { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 1rem; align-items: start; }
.input-column, .detail-column { min-width: 0; }
.detail-column .analysis-panel { display: none; }
.detection-detail { border: 1px solid var(--border-color-primary); border-radius: 14px;
  padding: 1rem; background: var(--block-background-fill); color: var(--body-text-color); }
.detection-detail h3 { margin: 0 0 .55rem; font-size: .9rem;
  color: var(--body-text-color); }
.detection-detail h3:not(:first-child) { margin-top: 1rem; }
.detail-empty { margin: 0; font-size: .85rem; color: var(--body-text-color-subdued); }
@media (max-width: 48rem) {
  .demo-columns { grid-template-columns: minmax(0, 1fr); }
}

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

/* 四源狀態列 —— card-head 之後、刻度尺之前。四個偵測源各一格。
   三態靠狀態字（未執行 / 未命中 / 命中）區分，顏色與淡化是冗餘強化不是唯一載體，
   顏色一律取自主題變數，深淺兩色下皆可見。 */
.source-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .5rem; margin-top: .9rem; }
.source { border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: .45rem .55rem; display: flex; flex-direction: column; gap: .15rem;
  background: var(--block-background-fill); }
.source-name { font-size: .78rem; color: var(--body-text-color-subdued); }
.source-state { font-size: .95rem; font-weight: 700; color: var(--body-text-color); }
.source-count { font-size: .72rem; color: var(--body-text-color-subdued); }
.source-hit { border-left: 3px solid var(--color-accent);
  background: var(--color-accent-soft); }
.source-clear { border-left: 3px solid var(--border-color-primary); }
.source-skip { border-left: 3px solid var(--border-color-primary); opacity: .6; }
@media (max-width: 40rem) {
  .source-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

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
.persona-reply { align-self: flex-start; display: flex; align-items: flex-end; gap: .45rem;
  max-width: 92%; }
.persona-sender { align-self: flex-end; display: flex; align-items: flex-end; gap: .45rem;
  max-width: 92%; }
.persona-reply .bubble.me { align-self: auto; max-width: 100%; }
.persona-sender .bubble.them { align-self: auto; max-width: 100%; }
.persona-avatar { width: 2.5rem; height: 2.5rem; flex: 0 0 2.5rem; border-radius: 50%;
  object-fit: cover; border: 1px solid var(--border-color-primary); }
.bubble.dropped { opacity: .6; border-style: dashed; background: transparent; }
.who { font-size: .7rem; color: var(--body-text-color-subdued); margin-bottom: .2rem; }
.lines { white-space: pre-wrap; line-height: 1.75; }
.sentence { padding: 1px 2px; border-radius: 3px; }
.sentence:target { background: var(--color-accent-soft);
  outline: 2px solid var(--color-accent); }

/* 個人資料標示（氣泡上的 inline 標註） */
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
