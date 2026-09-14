"""Gradio 介面 —— 偵測核心的第一個看得見的消費者。

本檔是介面層的**唯一**檔案，位置由兩個既有約束指定：`pyproject.toml` 的
`banned-api` 對 `gradio` 的訊息明寫「Gradio 相關程式碼請放在 `app.py`」、
`per-file-ignores` 只豁免 `"app.py"`；HuggingFace Spaces 亦固定執行 repo
根目錄的 `app.py`。因此本 change 不新增目錄、不修改 ruff 的兩張表。

三層結構，由內而外：

    scam_guard.detect()              偵測核心（本檔不修改它一行）
        ↓
    確定性渲染層（render_lines）      純 Python，無模型，三段純文字
        ↓
    ├─ 模式二「這是詐騙嗎」：直接顯示，MUST NOT 經過模型
    └─ 模式一「詐騙對練」：可選的措辭潤飾層，受三條前綴可判定的條件約束

更新順序是 requirement 而非實作細節：面板 MUST 在潤飾層產生任何字元之前完成
更新，受害方的回應 MUST 以 streaming 呈現。兩者合起來是專題論點的現場證據 ——
規則層零成本可稽核，模型昂貴。因此送出的 handler 是 generator function，
第一次 `yield` 已含完整面板。
"""

import html
import json
import operator
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

import gradio as gr

from scam_guard.check import CheckRegistry
from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect
from scam_guard.redact import redact_document
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import CheckResult, Coord, Message, Request, ScamType, Verdict

# ---------------------------------------------------------------------------
# 組裝層：註冊、上限、可選依賴的注入點
# ---------------------------------------------------------------------------

LIMITS: Limits = DEFAULT_LIMITS
"""上限的單一來源。**同一個變數**同時傳給 `detect()` 與 `build_document()`。

兩處用不同的 `Limits` 會產生不同的丟棄則數，於是同一個座標 `(3, 0)` 在兩邊
指向不同的句子 —— `detect()` 的 docstring 為此警告過「座標系必須有唯一的產生者」。
"""

SENDER_THEM = "them"
"""模式一的發送者標示。`project.md` 的輸入契約範例寫的就是 `"from": "them"`。"""

SAMPLES_PATH = Path(__file__).with_name("demo_samples.json")

UNREGISTERED_CHECKS: tuple[tuple[str, str], ...] = (("domain_age", "未注入 RDAP 解析器"),)
"""組裝層**明確知道其存在、但選擇不註冊**的檢查，以及不註冊的理由。

只收錄已實作且有 spec 定義「未註冊」狀態的檢查。尚未實作的檢查不列入 ——
把它們寫進清單等於臆測未來的名稱，而名稱一旦不符，畫面上會永遠掛著一行
指向不存在的東西的「未註冊」。
"""

PiiSpan: TypeAlias = tuple[int, int, str]
"""個資區間 `(start, end, type)`，座標在**正規化後的句子**上，對齊
`add-pii-recognizers` 的契約（辨識器只回報區間與類型，不改寫文字）。"""

PiiRecognizer: TypeAlias = Callable[[str], list[PiiSpan]]
Polisher: TypeAlias = Callable[[Sequence[str]], Iterator[str]]
TranscriptLogger: TypeAlias = Callable[[list[str]], None]

PII_RECOGNIZER: PiiRecognizer | None = None
"""個資辨識器。未注入時面板顯示「未掛載」，MUST NOT 顯示「未偵測到個資」。"""

POLISHER: Polisher | None = None
"""模式一的措辭潤飾層。輸入只有確定性渲染層的三段，**簽章中沒有訊息原文**。"""

TRANSCRIPT_LOGGER: TranscriptLogger | None = None
"""模式一的原文記錄器。以注入表達而非布林開關 —— 一個預設為 False 的布林是一個
可以被貼進設定檔、看起來像是有人想過的值；而「沒有注入」是不可能被誤解的狀態。
**公開部署預設不注入。** 注入時 UI 才顯示「本模式的輸入會被記錄」。"""


def build_registry() -> CheckRegistry:
    """建立本介面唯一的檢查註冊表。每落地一項檢查，此處多一行。

    **URL 層的五個檢查今日不註冊**：`register_url_checks()` 需要一份
    `PublicSuffixList`（由 `tools/fetch_psl.py` 落到被版控排除的 `data/`）
    與一張權重表（`add-weight-table` 尚未落地）。兩者在 HF Spaces 上 clone
    出來的 repo 裡都不存在，而猜一組權重就是臆測。落地後此處多兩行。
    """
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    return registry


REGISTRY: CheckRegistry = build_registry()


# ---------------------------------------------------------------------------
# 範例庫
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """一則進版控的範例訊息。`source_uri` 為 CC BY-SA 4.0 的姓名標示要求。"""

    label: str
    text: str
    source_uri: str


def load_samples(path: Path = SAMPLES_PATH) -> list[Sample]:
    """載入範例庫。

    **缺檔即 `raise`，不降級。** 這個檔案進版控，缺少它代表安裝壞了，不是一個
    正常狀態。不讀 `data/`、不內建預設清單、不回傳空清單 —— 「有 `data/` 就讀
    `data/`、沒有就用內嵌」會讓 HF Spaces 永遠走其中一條、本機永遠走另一條，
    於是兩條路徑的差異不會被任何人發現。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"範例庫檔案不存在：{path.name}（預期位於 repo 根目錄 {path}）。"
            f"此檔進版控，缺少它代表安裝不完整。"
        )
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if "samples" not in loaded:
        raise ValueError(f"範例庫檔案缺少 samples 欄位：{path.name}")
    samples: list[Sample] = []
    for entry in loaded["samples"]:
        for field_name in ("label", "text", "source_uri"):
            if field_name not in entry:
                raise ValueError(
                    f"範例庫的一筆資料缺少 {field_name} 欄位：{path.name}，該筆為 {entry!r}"
                )
        samples.append(
            Sample(label=entry["label"], text=entry["text"], source_uri=entry["source_uri"])
        )
    return samples


SAMPLES: list[Sample] = load_samples()


# ---------------------------------------------------------------------------
# 確定性渲染層 —— 兩個模式共用的唯一文案來源
#
# 交棒事項：`add-line-adapter` 落地時 MUST 把本層移到兩個介面都能 import 的
# 位置（`adapters/` 不能 import `app.py`，方向反了）。終點是
# `add-verdict-render` 在 `scam_guard/` 內持有它 —— `verdict.evidence` 與
# `verdict.actions` 本來就是它產的。現在不預先抽象：只有一個消費者時抽出一個
# 共用模組，是「不做沒被要求的彈性」要擋的東西。
# ---------------------------------------------------------------------------

UNDECIDED = "尚無法判定這則訊息是不是詐騙。"
RAW_GROUNDS_PREFIX = "（未經文案渲染的原始檢查明細）"
FALLBACK_ACTION = "撥打 165 反詐騙專線查證。"
FALLBACK_NOTE = "此建議不基於本次判定。"
UNDETERMINED_CELL = "未判定"


@dataclass(frozen=True)
class VerdictLines:
    """三段台詞，**純文字**，不含任何標記語言。

    各介面自行把段落包成自己的呈現形式：Gradio 包成 HTML、未來的 LINE adapter
    包成純文字。若本層吐 HTML，`add-line-adapter` 只能自己重寫一份，然後兩份
    文案開始漂移。

    缺料的段為 `None`（整段**不存在**），不是空字串 —— 空字串會在呈現層變成
    一個看不出是「沒有資料」還是「資料是空的」的空白區塊。
    """

    judgement: str | None
    grounds: str | None
    advice: str | None

    def segments(self) -> list[str]:
        """存在的段，依判定、依據、建議的順序。"""
        return [part for part in (self.judgement, self.grounds, self.advice) if part is not None]


def render_lines(verdict: Verdict) -> VerdictLines:
    """把 `Verdict` 渲染成三段台詞。不呼叫模型，兩個模式共用。

    判定段在 `scam_probability is None` 時陳述尚無法判定並且**不輸出任何數字** ——
    `types.py` 的 `Verdict` docstring 已把這條寫成契約：「呼叫端 MUST 顯示
    『無法判定』而非數字」。0% 是一個答案，而正確的狀態是沒有答案。

    依據段有**兩個可分辨的狀態**而非一個 fallback：`verdict.evidence` 由
    `add-verdict-render` 填（屬 `scoring`），但 `rule-signals` 早於 `scoring`
    落地，那段時間裡面板有三十幾行、命中好幾條，若只認 `verdict.evidence`
    依據段就會是空的而受害方會說「尚無訊號」—— 那是錯的。因此改用命中檢查的
    `name` 與 `detail`，並在段首標示本段未經文案渲染。
    """
    if verdict.scam_probability is None:
        if verdict.confidence == 0.0 and verdict.scam_type is None:
            judgement = None
        elif verdict.scam_type is None:
            judgement = UNDECIDED
        else:
            judgement = f"{UNDECIDED}訊號指向的類型是{verdict.scam_type.value}。"
    else:
        judgement = (
            f"這則訊息是詐騙的機率為 {verdict.scam_probability:.0%}，"
            f"系統對本次判定的信心值為 {verdict.confidence:.2f}。"
        )
        if verdict.scam_type is not None:
            judgement += f"類型為{verdict.scam_type.value}。"

    hits = [result for result in verdict.checks if result.hit]
    if verdict.evidence:
        grounds = "\n".join(verdict.evidence)
    elif hits:
        grounds = "\n".join(
            [RAW_GROUNDS_PREFIX] + [f"{result.name}：{result.detail}" for result in hits]
        )
    else:
        grounds = None

    advice = "\n".join(verdict.actions) if verdict.actions else None

    return VerdictLines(judgement=judgement, grounds=grounds, advice=advice)


def system_status_line(check_count: int) -> str:
    """三段全缺時，模式一說的系統狀態句。

    依據是**組裝層的事實**（註冊了幾個檢查），不是對 `Verdict` 的猜測。
    兩句都可驗證，而且每落地一個檢查就會自動改變。
    """
    if check_count == 0:
        return "目前沒有註冊任何檢查，系統看不出這則訊息的任何訊號。"
    return f"{check_count} 項檢查全部未命中，尚無足以判定的訊號。"


def fallback_advice() -> str:
    """保底建議動作，兩個模式皆顯示，且標示為不基於本次判定。

    它是一個無論這則訊息是真是假都不會讓使用者受損的動作，正是
    `add-verdict-render` 對建議動作的定義。
    """
    return f"{FALLBACK_ACTION}（{FALLBACK_NOTE}）"


def practice_speech(lines: VerdictLines, check_count: int) -> list[str]:
    """模式一的受害方台詞。三段全缺時改說系統狀態句。"""
    segments = lines.segments()
    if not segments:
        segments = [system_status_line(check_count)]
    return [*segments, fallback_advice()]


def inquiry_answer(lines: VerdictLines) -> list[str]:
    """模式二的輸出。三段全缺時陳述尚無法判定，並仍給出保底動作 ——
    使用者帶著一個真實的問題來，一句「系統狀態」回答不了他。"""
    segments = lines.segments()
    if not segments:
        segments = [UNDECIDED]
    return [*segments, fallback_advice()]


# ---------------------------------------------------------------------------
# 措辭潤飾層的驗證 —— 三條前綴可判定的條件
# ---------------------------------------------------------------------------

DIGITS = re.compile(r"\d+")
URL_MARKERS = ("http", "www.")

POLISH_NOT_INJECTED = "措辭潤飾：未注入（顯示確定性渲染層的原樣輸出）"
POLISH_STREAMING = "措辭潤飾：產生中"
POLISH_ACCEPTED = "措辭潤飾：已通過驗證"
POLISH_DISCARDED = "措辭潤飾：已丟棄（未通過驗證）"


def allowed_numbers(segments: Sequence[str]) -> frozenset[str]:
    """確定性層輸出中的每一段連續數字。潤飾層只能使用這些數字。"""
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
    """「不得新增事實」的三條可驗證條件，逐字元判定。

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
# 兩個模式各自的 Request 組法
# ---------------------------------------------------------------------------

BLANK_LINE = re.compile(r"\n[^\S\n]*\n")


def practice_messages(previous: Sequence[Message], text: str) -> list[Message]:
    """模式一：把新的一則接到對話後面。

    `sender` 全部填 `"them"`（人扮演詐騙方）；`sent_at` 填**實際送出時間**，
    不填 `None` —— `None` 的語意是「不知道」，而模式一知道。為了讓下游好看而
    謊報「不知道」，與「禁止臆測性 fallback」是同一條原則的反面。

    受害方的台詞 MUST NOT 出現在回傳值中：它是 `Verdict` 的呈現，回灌之後
    下一輪 `detect()` 會讀到自己上一輪的輸出並對它產生訊號。
    """
    return [*previous, Message(text=text, sender=SENDER_THEM, sent_at=datetime.now(tz=UTC))]


def build_inquiry_request(text: str) -> Request:
    """模式二：以**空行**切分為多則。單則即 `Request.from_text()`。

    不解析 `2026/09/14 10:00 小明` 這種 LINE 轉傳的時間戳行 —— 那個格式隨
    LINE 版本與語言設定改變，猜錯的後果是把一則訊息切成六則，每一則只有半句話，
    座標系跟著錯。解析屬 `add-line-adapter`。

    `sender` 與 `sent_at` 一律為 `None`，因為系統真的不知道。
    """
    chunks = [chunk.strip() for chunk in BLANK_LINE.split(text)]
    kept = [chunk for chunk in chunks if chunk]
    if not kept:
        raise gr.Error("輸入為空：請貼上收到的訊息內容")
    if len(kept) == 1:
        return Request.from_text(kept[0])
    return Request(messages=[Message(text=chunk) for chunk in kept])


# ---------------------------------------------------------------------------
# 分析面板
# ---------------------------------------------------------------------------

STATE_HIT = "hit"
STATE_NOT_HIT = "not-hit"
STATE_SKIPPED = "skipped"
STATE_UNREGISTERED = "unregistered"

STATE_LABELS = {
    STATE_HIT: "命中",
    STATE_NOT_HIT: "執行了未命中",
    STATE_SKIPPED: "因短路未執行",
    STATE_UNREGISTERED: "未註冊",
}

PII_NOT_MOUNTED = "PII 辨識器：未掛載"
PII_NO_HIT = "未偵測到個資"
PII_SCOPE_NOTE = (
    "辨識範圍只有身分證、手機、市話、信用卡四類。"
    "<b>姓名與地址不在辨識範圍內</b> —— 它們需要另一條預設關閉的選配路徑"
    "（gated 模型，模型卡自述 hard-negative 誤報率 77.5%）。"
)


def check_state(result: CheckResult) -> str:
    """把一筆 `CheckResult` 分類為四種狀態之一（未註冊不從這裡來）。

    判定依據是 `scam_guard.pipeline` 公開的 `NOT_HIT` 與 `SKIPPED` 兩個常數，
    **不是字串字面值** —— 介面層 import 核心是允許的方向。

    交棒事項：根治「靠 `detail` 字串判定未執行」需要 `CheckResult` 新增
    `skipped` 欄位（檢查自己也可以回傳 `detail="因短路未執行"`，那會被誤分類），
    屬 `add-check-pipeline` 的後續。在那之前，無法分類即 `raise`，不猜。
    """
    if result.hit:
        return STATE_HIT
    if result.detail == NOT_HIT:
        return STATE_NOT_HIT
    if result.detail == SKIPPED:
        return STATE_SKIPPED
    raise ValueError(
        f"無法分類的檢查記錄：name={result.name!r}、hit=False、detail={result.detail!r}"
        f"（未命中的 detail 只能是 pipeline.NOT_HIT 或 pipeline.SKIPPED）"
    )


def escaped(value: str) -> str:
    """使用者輸入進入標記之前的唯一入口。

    輸入是真實詐騙訊息，裡面有 `<`、`&`、完整網址，不逸出就是一個 XSS，
    而這是一個公開的 Space。
    """
    return html.escape(value).replace("\n", "<br>")


def anchor_id(coord: Coord) -> str:
    """證據座標對應的 HTML 錨點 id。純 HTML 與 CSS，不需要 JavaScript。"""
    message_index, sentence_index = coord
    return f"s-{message_index}-{sentence_index}"


def render_verdict_row(verdict: Verdict) -> str:
    """判定列：三個並列且各自標示名稱的量，橫跨最上方全寬。

    不合併成單一指標、不用三級燈號 —— `project.md` 已經否決過顏色標籤。
    詐騙機率與信心值是兩個獨立的量：前者答「是不是詐騙」，後者答「有沒有足夠
    依據下判斷」。
    """
    if verdict.scam_probability is None:
        probability = UNDETERMINED_CELL
    else:
        probability = f"{verdict.scam_probability:.0%}"
    scam_type = UNDETERMINED_CELL if verdict.scam_type is None else escaped(verdict.scam_type.value)
    return (
        '<div class="verdict-row">'
        f'<div class="cell"><div class="cell-label">詐騙機率</div>'
        f'<div class="cell-value">{probability}</div></div>'
        f'<div class="cell"><div class="cell-label">信心值</div>'
        f'<div class="cell-value">{verdict.confidence:.2f}</div>'
        f'<div class="cell-note">系統對本次判定的自我信心，不是命中率或準確率</div></div>'
        f'<div class="cell"><div class="cell-label">詐騙類型</div>'
        f'<div class="cell-value">{scam_type}</div></div>'
        "</div>"
    )


def render_evidence(result: CheckResult, doc: Document) -> str:
    """把證據座標展開成指向對話中該句的連結。

    **無效座標讓 `KeyError` 向上傳播，不吞。** `Document.index_of()` 的
    docstring 寫了理由：無效座標代表產生它的檢查算錯了。面板吞掉它會讓一個
    算錯座標的檢查看起來只是少一條依據。
    """
    if not result.evidence:
        return ""
    links = [
        f'<a class="evidence" href="#{anchor_id(coord)}">'
        f"({coord[0]},{coord[1]}) {escaped(doc.raw_at(coord))}</a>"
        for coord in result.evidence
    ]
    return '<div class="evidence-list">' + "".join(links) + "</div>"


def render_checks(verdict: Verdict, doc: Document) -> str:
    """偵測項目逐筆顯示，**不過濾未命中者**。行數由 `Verdict.checks` 決定，
    新增一項檢查不需要修改本函式。"""
    rows: list[str] = []
    for result in verdict.checks:
        state = check_state(result)
        hard = '<span class="badge-hard">硬證據</span>' if result.hit and result.hard else ""
        rows.append(
            f'<div class="check state-{state}">'
            f'<div class="check-head"><span class="check-name">{escaped(result.name)}</span>'
            f'<span class="badge">{STATE_LABELS[state]}</span>{hard}</div>'
            f'<div class="check-detail">{escaped(result.detail)}</div>'
            f"{render_evidence(result, doc)}"
            "</div>"
        )
    for name, reason in UNREGISTERED_CHECKS:
        rows.append(
            f'<div class="check state-{STATE_UNREGISTERED}">'
            f'<div class="check-head"><span class="check-name">{escaped(name)}</span>'
            f'<span class="badge">{STATE_LABELS[STATE_UNREGISTERED]}</span></div>'
            f'<div class="check-detail">{escaped(reason)}</div>'
            "</div>"
        )
    return "".join(rows)


def pii_counts(sentence: str, recognizer: PiiRecognizer) -> list[tuple[str, int]]:
    """某個正規化句子命中的個資類型與筆數，依類型名稱排序。"""
    counts: dict[str, int] = {}
    for _start, _end, pii_type in recognizer(sentence):
        counts[pii_type] = counts.get(pii_type, 0) + 1
    return sorted(counts.items(), key=operator.itemgetter(0))


def render_pii_tags(sentence: str, recognizer: PiiRecognizer | None) -> str:
    """句子層級的個資標籤，掛在氣泡中該句之後。**不改寫文字。**

    粒度只到句子是資料結構上的限制：辨識器的輸入是**正規化後**的句子，回報的
    區間在正規化座標系裡，而氣泡顯示的是 `raw_at()` 的原文片段；要把區間映到
    原文需要句內的字元對映，而今天的 `Document` 沒有保留它。

    **這是一個可替換的函式。** `add-sentence-offsets` 落地後 `Document` 每句
    會攜帶到原文片段的偏移對映（`raw_bounds_at(coord, start, end)`），屆時把
    本函式換成原文上的字元層級底線即可，改動集中於此一處。

    絕不以 `raw.find(sentence)` 或在原文中搜尋 PII 片段來補救：NFKC 有一對多
    與多對一，全形數字或零寬字元時必然搜不到，需要一條「找不到就不標」的規則 ——
    那會讓規避手法同時關掉 PII 標註，一個可被利用的沉默。
    """
    if recognizer is None:
        return ""
    tags = [
        f'<span class="pii-tag">{escaped(pii_type)} ×{count}</span>'
        for pii_type, count in pii_counts(sentence, recognizer)
    ]
    return "".join(tags)


def render_pii_highlight(sentence: str, spans: Sequence[PiiSpan]) -> str:
    """在**正規化句子**上做字元層級標示。

    ⚠️ **先用原始索引切片，再對每一段各自逸出。** `html.escape()` 是一對多的
    （`&` → `&amp;` 是 1→5），先逸出整句再用區間切片，從第一個 `&` 之後的所有
    索引全部位移，標示框會標到錯的字 —— 而 `&` 在釣魚連結的 query string 裡
    幾乎必然出現。這個錯不會拋例外，畫面上只是標到隔壁幾個字。
    """
    pieces: list[str] = []
    cursor = 0
    for start, end, pii_type in sorted(spans, key=operator.itemgetter(0)):
        pieces.append(escaped(sentence[cursor:start]))
        pieces.append(
            f'<mark class="pii-mark" title="{escaped(pii_type)}">'
            f"{escaped(sentence[start:end])}</mark>"
        )
        cursor = end
    pieces.append(escaped(sentence[cursor:]))
    return "".join(pieces)


def render_pii_block(doc: Document, recognizer: PiiRecognizer | None) -> str:
    """面板的個資區塊。字元層級的標示只出現在這裡，且標在正規化句子上。

    「未掛載」與「未偵測到個資」是兩個狀態，文字必須不同：前者是「沒有人在看」，
    後者是「看過了，沒有」。在一個以展示偵測能力為目的的 demo 裡混淆這兩者，
    剛好是最糟的謊。

    本路徑**不 import 也不呼叫任何遮蔽（redact）程式碼**：標註指出位置與類型、
    文字不變、呈現給使用者；遮蔽改寫文字、只有 log 能讀。兩條不同的路徑。
    """
    scope = f'<div class="note">{PII_SCOPE_NOTE}</div>'
    if recognizer is None:
        return f'<div class="pii"><h4>個人資料標註</h4><div>{PII_NOT_MOUNTED}</div>{scope}</div>'

    rows: list[str] = []
    for coord, sentence in zip(doc.coords, doc.sentences):
        spans = recognizer(sentence)
        if not spans:
            continue
        rows.append(
            f'<div class="pii-row"><span class="coord">({coord[0]},{coord[1]})</span>'
            f"{render_pii_highlight(sentence, spans)}</div>"
        )
    body = "".join(rows) if rows else f"<div>{PII_NO_HIT}</div>"
    return (
        '<div class="pii"><h4>個人資料標註</h4>'
        '<div class="note">以下顯示的是<b>正規化後</b>的句子，不是原文片段。標註只指出'
        "位置與類型，不改寫任何文字。</div>"
        f"{body}{scope}</div>"
    )


def render_panel(
    verdict: Verdict,
    doc: Document,
    check_count: int,
    recognizer: PiiRecognizer | None,
) -> str:
    """偵測項目與理由。**不含潤飾狀態** —— 那是模型那一側的事，放進來會讓
    「潤飾層未注入」與「潤飾層已注入的第一次產出」兩種情況的面板內容不同，
    而 requirement 要求它們相同。"""
    truncation = ""
    if doc.truncated:
        truncation = (
            f'<div class="truncation">已丟棄最舊的 {doc.dropped_messages} 則，'
            "它們未納入本次判定。</div>"
        )
    return (
        '<div class="panel"><h4>偵測項目與理由</h4>'
        f'<div class="registered">已註冊 {check_count} 個檢查</div>'
        f"{truncation}"
        f'<div class="checks">{render_checks(verdict, doc)}</div>'
        f"{render_pii_block(doc, recognizer)}"
        "</div>"
    )


EXTERNAL_BLOCK = (
    '<div class="external"><h4>外部查詢</h4>'
    '<div class="check state-unregistered">'
    '<div class="check-head"><span class="check-name">domain_age</span>'
    '<span class="badge">未註冊</span></div>'
    '<div class="check-detail">未注入 RDAP 解析器</div></div>'
    '<div class="note">本 demo <b>目前不對任何外部服務發出請求</b>。'
    "開啟網域年齡檢查後，會把訊息中網址的 <b>registrable domain</b>"
    "（不含 path、query string 與子網域）送到該網域的註冊局。"
    "這個 Space 任何人都能開，注入之後每一個訪客輸入的每一個網域都會從共用的"
    "出口 IP 發出查詢 ——「顯式做的決定」在公開端點上換了主體，因此不注入。</div>"
    "</div>"
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
    """
    positions = doc.message_range(message_index)
    if not positions:
        return (
            '<div class="bubble them dropped">'
            f'<div class="who">{escaped(sender_label)} · 第 {message_index + 1} 則 · '
            "未納入本次判定</div>"
            f'<div class="dropped-text">{escaped(message.text)}</div></div>'
        )
    parts: list[str] = []
    for position in positions:
        coord = doc.coords[position]
        parts.append(
            f'<span class="sentence" id="{anchor_id(coord)}">'
            f"{escaped(doc.raw_sentences[position])}</span>"
            f"{render_pii_tags(doc.sentences[position], recognizer)}"
        )
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
    recognizer: PiiRecognizer | None,
) -> str:
    """對話（模式一）或輸入回顯（模式二）。

    用自繪的 HTML 而非 `gr.Chatbot`：後者的單位是一則訊息，而這裡需要的最小
    單位是**句子**（證據座標的粒度就是句子），還要在句子之後掛 PII 標籤。
    """
    blocks: list[str] = []
    for message_index, message in enumerate(messages):
        blocks.append(render_message(message_index, message, doc, sender_label, recognizer))
        if message_index < len(replies):
            blocks.append(
                '<div class="bubble me"><div class="who">受害方（由判定渲染）</div>'
                f'<div class="lines">{escaped(replies[message_index])}</div></div>'
            )
    return f'<div class="conversation">{"".join(blocks)}</div>'


def render_answer(segments: Sequence[str]) -> str:
    """模式二的輸出區塊。段落之間可分辨，缺料的段根本不在 `segments` 裡。"""
    body = "".join(f'<p class="segment">{escaped(part)}</p>' for part in segments)
    return f'<div class="answer"><h4>判定結果</h4>{body}</div>'


# ---------------------------------------------------------------------------
# Event handlers —— 全部定義於模組層，狀態以顯式參數傳入與傳出
# ---------------------------------------------------------------------------


def practice_submit(
    text: str,
    messages: list[Message],
    replies: list[str],
) -> Iterator[tuple[list[Message], list[str], str, str, str, str, str]]:
    """模式一的送出處理，**generator function**。

    第一次 `yield` 帶完整面板與確定性層的台詞；其後每次 `yield` 追加潤飾層的
    新片段。面板 MUST 在模型產生任何字元之前完成更新 —— 若實作成「等模型跑完
    再一起更新」，畫面仍然正確，只是慢，而**沒有任何測試會報告展示效果消失**。
    因此測試驗證的是第一次產出的內容，不是最終畫面。
    """
    if not text.strip():
        raise gr.Error("輸入為空：請輸入一則詐騙方會說的話")

    updated = practice_messages(messages, text)
    if TRANSCRIPT_LOGGER is not None:
        # 顯式取 `message.text`：`add-redact-apply` 落地後 `Message.__repr__`
        # 不再印文字，靠 f-string 寫出來的會是 `len=83`。這不是繞過那條規則，
        # 是承認它 —— repr 的封鎖防的是意外洩漏，而這是一個刻意的、
        # 有注入前提的記錄。
        TRANSCRIPT_LOGGER([message.text for message in updated])

    request = Request(messages=updated)
    verdict = detect(request, REGISTRY, limits=LIMITS)
    document = build_document(request.messages, LIMITS)

    check_count = len(REGISTRY.enabled())
    segments = practice_speech(render_lines(verdict), check_count)
    baseline = "\n".join(segments)

    updated_replies = [*replies, baseline]
    row = render_verdict_row(verdict)
    panel = render_panel(verdict, document, check_count, PII_RECOGNIZER)
    conversation = render_conversation(updated, updated_replies, document, "詐騙方", PII_RECOGNIZER)
    status = POLISH_NOT_INJECTED if POLISHER is None else POLISH_STREAMING
    yield updated, updated_replies, conversation, row, panel, status, ""

    if POLISHER is None:
        return

    validator = PolishValidator(segments, verdict)
    for chunk in POLISHER(segments):
        if not validator.feed(chunk):
            updated_replies[-1] = baseline
            yield (
                updated,
                updated_replies,
                render_conversation(updated, updated_replies, document, "詐騙方", PII_RECOGNIZER),
                row,
                panel,
                POLISH_DISCARDED,
                "",
            )
            return
        updated_replies[-1] = validator.text
        yield (
            updated,
            updated_replies,
            render_conversation(updated, updated_replies, document, "詐騙方", PII_RECOGNIZER),
            row,
            panel,
            POLISH_STREAMING,
            "",
        )
    yield (
        updated,
        updated_replies,
        render_conversation(updated, updated_replies, document, "詐騙方", PII_RECOGNIZER),
        row,
        panel,
        POLISH_ACCEPTED,
        "",
    )


def inquiry_submit(text: str) -> tuple[str, str, str, str]:
    """模式二的送出處理。

    **不呼叫潤飾層，即使已注入。** 這是產品路徑，使用者在問一個關於自己安危的
    問題，模型在這條路徑上連措辭都不經手 —— 這是「LLM 不能有最終話語權」最直接
    的實作。而且模式二未來要給 LINE 用，那裡沒有串流可以展示速度差。

    **本模式不寫任何記錄。** 合法來源只有 `Verdict.redacted`（`add-redact-apply`
    的 `RedactedText`），而今天那個欄位不存在 —— 沒有合法來源就不寫，不找替代品。
    交棒事項：`add-redact-apply` 落地後，此處 MUST 改為只寫 `Verdict.redacted`
    的 `sentences` / `coords` / `counts`，MUST NOT 寫入 `Request`、`Message.text`
    或 `Verdict.evidence`。這解掉該 change 自記的「投影可能沒有消費者」。
    """
    request = build_inquiry_request(text)
    verdict = detect(request, REGISTRY, limits=LIMITS)
    document = build_document(request.messages, LIMITS)

    check_count = len(REGISTRY.enabled())
    segments = inquiry_answer(render_lines(verdict))
    return (
        render_conversation(request.messages, (), document, "收到的訊息", PII_RECOGNIZER),
        render_verdict_row(verdict),
        render_panel(verdict, document, check_count, PII_RECOGNIZER),
        render_answer(segments),
    )


def fill_sample(label: str) -> str:
    """把範例填入輸入框，**不自動送出**。"""
    for sample in SAMPLES:
        if sample.label == label:
            return sample.text
    raise ValueError(f"範例庫中沒有這個標籤：{label!r}")


# ---------------------------------------------------------------------------
# 介面組裝
# ---------------------------------------------------------------------------

CSS = """
.verdict-row { display: flex; gap: 1.5rem; width: 100%; padding: .8rem 1rem;
  border: 1px solid #c6cbd2; border-radius: 8px; }
.verdict-row .cell { flex: 1; }
.cell-label { font-size: .8rem; color: #5a6069; letter-spacing: .05em; }
.cell-value { font-size: 1.5rem; font-weight: 700; }
.cell-note { font-size: .72rem; color: #5a6069; }
.conversation { display: flex; flex-direction: column; gap: .6rem; }
.bubble { border-radius: 10px; padding: .5rem .75rem; max-width: 92%; }
.bubble.them { background: #eef1f5; align-self: flex-start; }
.bubble.me { background: #e3eaf4; align-self: flex-end; border: 1px solid #b9c6d8; }
.bubble.dropped { opacity: .55; border: 1px dashed #8a8f98; }
.who { font-size: .72rem; color: #5a6069; margin-bottom: .25rem; }
.lines { white-space: pre-wrap; line-height: 1.7; }
.sentence { padding: 1px 2px; border-radius: 3px; }
.sentence:target { background: #dbe7f5; outline: 2px solid #2b4c7e; }
.pii-tag { font-size: .7rem; background: #dfe3e8; border-radius: 3px;
  padding: 0 .3rem; margin: 0 .2rem; }
.pii-mark { background: #dbe7f5; border-bottom: 2px solid #2b4c7e; }
.check { border-left: 4px solid #cfd4da; padding: .3rem .6rem; margin-bottom: .3rem; }
.check.state-hit { border-left: 4px solid #2b4c7e; background: #eef2f8; font-weight: 600; }
.check.state-not-hit { border-left: 4px solid #cfd4da; opacity: .7; }
.check.state-skipped { border-left: 4px dashed #8a8f98; }
.check.state-unregistered { border-left: 4px dotted #8a8f98; font-style: italic; }
.check-head { display: flex; gap: .4rem; align-items: baseline; }
.check-name { font-family: ui-monospace, monospace; }
.badge { font-size: .68rem; border: 1px solid #8a8f98; border-radius: 3px; padding: 0 .25rem; }
.badge-hard { font-size: .68rem; border: 1px solid #2b4c7e; border-radius: 3px;
  padding: 0 .25rem; }
.check-detail { font-size: .82rem; color: #3d434b; }
.evidence-list { display: flex; flex-direction: column; gap: .15rem; margin-top: .2rem; }
a.evidence { font-size: .78rem; text-decoration: underline; }
.registered { font-weight: 700; margin-bottom: .4rem; }
.truncation { border: 1px dashed #8a8f98; padding: .3rem .5rem; margin-bottom: .4rem; }
.note { font-size: .74rem; color: #5a6069; line-height: 1.6; margin-top: .4rem; }
.panel, .external, .answer, .pii { border: 1px solid #c6cbd2; border-radius: 8px;
  padding: .6rem .8rem; margin-bottom: .6rem; }
.pii-row { font-family: ui-monospace, monospace; font-size: .8rem; margin: .2rem 0; }
.coord { color: #5a6069; margin-right: .4rem; }
.segment { margin: .4rem 0; white-space: pre-wrap; }
.notice { border: 2px solid #2b4c7e; background: #eef2f8; padding: .5rem .8rem;
  font-weight: 700; border-radius: 6px; }
"""

PRIVACY_NOTE = """
<details><summary><b>這個 demo 對你輸入的內容做了什麼（請先讀這段）</b></summary>
<div class="note">
<p><b>應用層</b>：「這是詐騙嗎」模式<b>不寫任何記錄</b> —— 唯一合法的記錄來源是
判定結果攜帶的可記錄投影（遮蔽後的句子、座標與分類計數），而那個欄位今天還不存在，
沒有合法來源就不寫，不找替代品。「詐騙對練」模式只有在組裝層注入記錄器時才記錄，
本次部署未注入時畫面上不會出現記錄標示。</p>
<p><b>存取記錄</b>：Gradio / uvicorn 的 access log 只含 HTTP 方法、路徑與狀態碼；
訊息文字走 POST body 與 WebSocket，不進 access log。</p>
<p><b>不在本系統控制範圍內的部分</b>：部署平台（HuggingFace Spaces）的容器標準輸出
保留在 Space 的 logs 分頁，Space 擁有者可見；未捕捉的 traceback 會落在那裡，
而 traceback 的區域變數可能含完整原文。<b>我們控制不了這一層。</b></p>
<p><b>模型與介面在同一個程序中</b>，未遮蔽的原文與模型在同一塊記憶體裡；
遮蔽不改變這個事實，它只在寫入 log 前有意義。</p>
<p>對話狀態存在瀏覽器工作階段中，程序重啟即消失，不落地成檔案。</p>
</div></details>
"""

INTRO = """
<div class="note">
<p><b>詐騙對練</b>：你扮演詐騙方打字，受害方由系統回應，而受害方說的每一句
都是判定結果的對話化呈現 —— 它只會說判定裡有的東西，<b>聊不起來是設計不是缺陷</b>。
這個模式的時間軸反映的是<b>打字的節奏，不是真實詐騙的時間分布</b>，
未來的軌跡檢查在這裡會得到無意義的結果。</p>
<p><b>這是詐騙嗎</b>：貼上你真的收到的訊息，輸出是判定、依據與建議動作。
這是產品本身，不是 demo 功能。</p>
<p>面板會在<b>毫秒級</b>完成更新，早於受害方的第一個字元 —— 那個速度差本身就是
「能用規則判的就不要叫 LLM」的現場證據。</p>
</div>
"""


def samples_listing() -> str:
    """範例庫的出處清單。CC BY-SA 4.0 的姓名標示要求以每筆的連結滿足。"""
    items = "".join(
        f"<li>{escaped(sample.label)} —— "
        f'<a href="{escaped(sample.source_uri)}" target="_blank" rel="noopener">出處</a></li>'
        for sample in SAMPLES
    )
    return (
        "<details><summary>範例訊息的出處（Cofacts，CC BY-SA 4.0）</summary>"
        f'<div class="note"><ul>{items}</ul>'
        "範例內容的授權為 CC BY-SA 4.0，與本專案其餘部分的 MIT 授權不同。</div></details>"
    )


def transcript_notice(logger: TranscriptLogger | None) -> str:
    """原文記錄器已注入時的顯著標示。未注入時不顯示 —— 那是兩個不同的狀態。"""
    if logger is None:
        return ""
    return (
        '<div class="notice">本模式的輸入會被記錄：組裝層已注入原文記錄器，'
        "你在這裡打的每一則訊息原文都會被寫出。不要在此貼上真實的個人資料。</div>"
    )


def build_demo() -> gr.Blocks:
    """組裝介面。

    `gr.Blocks` 而非 `gr.Interface`：flagging（使用者按下 flag 會把原文連同輸出
    寫進 `.gradio/flagged/dataset.csv`，一個落地的檔案）只存在於 `gr.Interface`，
    Blocks 沒有這條路徑。這是顯式的選擇而不是預設值，連同 `analytics_enabled=False`
    一起構成「框架內建的提交功能已關閉」。

    兩個模式是兩個 `gr.Tab`，各自持有自己的 `gr.State` 與輸出元件 ——
    狀態隔離因此是結構上的，不靠任何清空邏輯維持。
    """
    labels = [sample.label for sample in SAMPLES]
    empty_document = build_document([], LIMITS)
    empty_verdict = Verdict(
        scam_probability=None,
        confidence=0.0,
        scam_type=None,
        evidence=[],
        actions=[],
        checks=[],
        redacted=redact_document(empty_document),
    )
    initial_row = render_verdict_row(empty_verdict)
    initial_panel = render_panel(empty_verdict, empty_document, len(REGISTRY.enabled()), None)

    with gr.Blocks(title="scam-guard", analytics_enabled=False) as demo:
        gr.HTML(INTRO)
        gr.HTML(PRIVACY_NOTE)
        gr.HTML(samples_listing())

        with gr.Tabs():
            with gr.Tab("詐騙對練"):
                practice_row = gr.HTML(initial_row)
                gr.HTML(transcript_notice(TRANSCRIPT_LOGGER))
                with gr.Row():
                    with gr.Column(scale=3):
                        practice_conversation = gr.HTML('<div class="conversation"></div>')
                        practice_input = gr.Textbox(
                            label="你（扮演詐騙方）",
                            lines=3,
                            placeholder="打一句詐騙方會說的話，受害方會以判定結果回應你",
                        )
                        with gr.Row():
                            practice_send = gr.Button("送出", variant="primary")
                            practice_pick = gr.Dropdown(labels, label="範例訊息", value=None)
                            practice_fill = gr.Button("填入範例")
                        practice_status = gr.HTML(f'<div class="note">{POLISH_NOT_INJECTED}</div>')
                    with gr.Column(scale=2):
                        practice_panel = gr.HTML(initial_panel)
                        gr.HTML(EXTERNAL_BLOCK)
                practice_messages_state = gr.State([])
                practice_replies_state = gr.State([])

            with gr.Tab("這是詐騙嗎"):
                inquiry_row = gr.HTML(initial_row)
                with gr.Row():
                    with gr.Column(scale=3):
                        inquiry_input = gr.Textbox(
                            label="貼上你收到的訊息",
                            lines=8,
                            placeholder="一則就直接貼上；一段轉傳的多則對話請以空行分隔",
                            info="以空行分隔為多則。不解析時間戳行與暱稱行 ——"
                            "那些會被當成一般文字。",
                        )
                        with gr.Row():
                            inquiry_send = gr.Button("判定", variant="primary")
                            inquiry_pick = gr.Dropdown(labels, label="範例訊息", value=None)
                            inquiry_fill = gr.Button("填入範例")
                        inquiry_answer_html = gr.HTML('<div class="answer"></div>')
                        inquiry_echo = gr.HTML('<div class="conversation"></div>')
                    with gr.Column(scale=2):
                        inquiry_panel = gr.HTML(initial_panel)
                        gr.HTML(EXTERNAL_BLOCK)

        practice_send.click(
            practice_submit,
            inputs=[practice_input, practice_messages_state, practice_replies_state],
            outputs=[
                practice_messages_state,
                practice_replies_state,
                practice_conversation,
                practice_row,
                practice_panel,
                practice_status,
                practice_input,
            ],
        )
        practice_fill.click(fill_sample, inputs=[practice_pick], outputs=[practice_input])
        inquiry_send.click(
            inquiry_submit,
            inputs=[inquiry_input],
            outputs=[inquiry_echo, inquiry_row, inquiry_panel, inquiry_answer_html],
        )
        inquiry_fill.click(fill_sample, inputs=[inquiry_pick], outputs=[inquiry_input])

    return demo


if __name__ == "__main__":
    # Gradio 6 把 `css` 從 Blocks 的建構子移到 `launch()`。
    build_demo().launch(css=CSS)
