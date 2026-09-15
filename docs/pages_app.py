"""GitHub Pages 靜態 demo 的 Python 側 —— 在瀏覽器（Pyodide）裡執行。

**為什麼這裡有第二個介面層，而不是直接跑 `app.py`。**
原訂的載體是 Gradio-Lite（`@gradio/lite`，Pyodide 上的 Gradio），那條路今天走不通：
該套件在 npm 上最後一次發佈是 2025-09-10（同一天其餘 `@gradio/*` 都還在更新），
它內建的 Gradio 5.45.0 對上今天的 PyPI 會在啟動階段連續踩到三個相依性漂移的錯
（`huggingface-hub` 版本衝突、`filelock` 新版的 `os.link(follow_symlinks=...)`
探測撞上 Gradio-Lite 自己的 `os.link` mock、`anyio` / `typing-inspection` 找不到
純 Python wheel）。上游官方的 hello world 範例在瀏覽器裡同樣起不來，所以那不是
本專案的問題，也不是版本挑錯了 —— 那條路目前是壞的。

於是本頁不經過 Gradio：Pyodide 直接安裝 `scam_guard` 的 wheel（它的執行期依賴
是 `[]`，micropip 不需要向 PyPI 解析任何東西），介面由 `index.html` 與本檔組成。

**本檔 MUST NOT 自己寫任何對訊息的判斷文字。** 使用者讀到的「依據」與「建議動作」
一律來自 `Verdict.evidence` 與 `Verdict.actions`，那是 `scam_guard/render.py` 的
產物；本檔只負責把它們包成 HTML。這條界線與 `app.py` 的是同一條，理由也一樣：
一個會自己造句的呈現層可以寫出沒有任何地方能報告的錯誤。

**與 `app.py` 的關係：兩者平行，互不 import，也不共用程式碼。** 這是一筆已知的
債：判定列、面板、氣泡的標記各有一份。`app.py` 的交棒註記已經寫了正解 ——
確定性渲染層要移到兩個介面都能 import 的位置。那是對 `app.py` 的修改，屬另一個
change，本 change 不碰它一行。

**只有「這是詐騙嗎」一個模式。** `app.py` 的另一個模式（詐騙對練）的展示重點是
面板早於受害方台詞完成更新，也就是規則層與模型層的速度差 —— 而模型層在瀏覽器裡
根本不存在（`llama-cpp-python` 沒有 Pyodide 版本）。把一個少了對照組的對照實驗
搬上來，展示的東西就不是它原本要展示的東西了。
"""

import html
import json
import operator
import re

from scam_guard.check import CheckRegistry
from scam_guard.normalize import DEFAULT_LIMITS, Document, build_document
from scam_guard.pii import find_pii
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import CheckResult, Coord, Message, Request, Verdict
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import load_tables, register_url_checks
from scam_guard.weights import load_weights

LIMITS = DEFAULT_LIMITS
"""上限的單一來源。**同一個變數**同時傳給 `detect()` 與 `build_document()` ——
兩處用不同的 `Limits` 會產生不同的丟棄則數，於是同一個座標在兩邊指向不同的句子。"""

PSL_DIR = "/psl"
"""PSL 快照在 Pyodide 檔案系統中的位置，由 `index.html` 在啟動時寫入。"""

PSL_MAX_AGE_DAYS = 90
"""PSL 快照的新鮮度上限。`PublicSuffixList.load()` 的必填參數，沒有預設值。

靜態站台沒有「啟動」這件事：快照的取得時間就是**最後一次部署的時間**，所以這個
數字實際上是在說「這個站台超過 90 天沒有重新部署就不該再運作」。
`.github/workflows/pages.yml` 以每週一次的排程重新部署來滿足它。

超過上限時本檔在 import 階段就拋例外，頁面顯示那個例外的原文（含 `fetched_at`
與實際天數），**不會少一個訊號繼續跑**。一個安靜降級的頁面不會告訴任何人任何事。
"""

UNREGISTERED_CHECKS: tuple[tuple[str, str], ...] = (
    ("domain_age", "未注入 RDAP 解析器 —— 本頁不對任何外部服務發出請求"),
    (
        "url_blocklist",
        "未打包 165 涉詐網址快照 —— 15.9 MB、107,499 筆，在瀏覽器內載入後佔用約 66 MB 記憶體",
    ),
)
"""本部署**明確知道其存在、但選擇不註冊**的檢查，以及不註冊的理由。

只收錄已實作、且有規格定義「未註冊」狀態的檢查。少一個訊號要在畫面上看得見，
否則「沒有訊號」與「沒有資料」在結果裡長得一模一樣。
"""

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

UNDETERMINED_CELL = "未判定"

BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
"""多則轉傳的分隔符。**以空行分隔，不解析時間戳行與暱稱行** —— 那個格式隨
LINE 版本與語言設定改變，猜錯的後果是把一則訊息切成六則、每則半句話，座標系跟著錯。"""


def build_registry() -> CheckRegistry:
    """建立本介面唯一的檢查註冊表。每落地一項檢查，此處多一行。

    URL 層的四個檢查在這裡註冊得起來，是因為建置腳本把 PSL 快照一起部署了。
    `url_blocklist` 不註冊：`register_url_checks()` 的 `store` 不給就不註冊它，
    那是該函式本來就定義好的行為，不是這裡的特例處理。
    """
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    register_url_checks(
        registry,
        PublicSuffixList.load(PSL_DIR, max_age_days=PSL_MAX_AGE_DAYS),
        load_tables(),
    )
    return registry


REGISTRY: CheckRegistry = build_registry()

TABLE = load_weights()
"""權重表的單一來源。`detect()` 的必填參數，且在 import 時就載入並驗證整張表 ——
缺漏會在頁面啟動時炸，不是在第一次命中時。"""


def escaped(value: str) -> str:
    """使用者輸入進入標記之前的唯一入口。

    輸入是真實詐騙訊息，裡面有 `<`、`&`、完整網址；不逸出就是一個 XSS，
    而這是一個公開的頁面。
    """
    return html.escape(value).replace("\n", "<br>")


def anchor_id(coord: Coord) -> str:
    """證據座標對應的 HTML 錨點 id。純 HTML 與 CSS，不需要 JavaScript。"""
    message_index, sentence_index = coord
    return f"s-{message_index}-{sentence_index}"


def check_state(result: CheckResult) -> str:
    """把一筆 `CheckResult` 分類為四種狀態之一（未註冊不從這裡來）。

    判定依據是 `scam_guard.pipeline` 公開的 `NOT_HIT` 與 `SKIPPED` 兩個常數，
    不是字串字面值。無法分類即 `raise`，不猜 —— 猜錯會讓一個回報了奇怪 detail
    的檢查看起來只是狀態標籤不同。
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


def build_request(text: str) -> Request:
    """把輸入框的內容組成 `Request`。以**空行**切分為多則，單則即單則。

    `sender` 與 `sent_at` 一律為 `None`，因為系統真的不知道。
    """
    kept = [chunk.strip() for chunk in BLANK_LINE.split(text) if chunk.strip()]
    if not kept:
        raise ValueError("輸入為空：請貼上你收到的訊息內容")
    if len(kept) == 1:
        return Request.from_text(kept[0])
    return Request(messages=[Message(text=chunk) for chunk in kept])


def render_verdict_row(verdict: Verdict) -> str:
    """判定列：三個並列且各自標示名稱的量。

    不合併成單一指標、不用三級燈號。詐騙機率與信心值是兩個獨立的量：
    前者答「是不是詐騙」，後者答「有沒有足夠依據下判斷」。
    機率為 `None` 時顯示「未判定」且**不輸出任何數字** —— 0% 是一個答案，
    而正確的狀態是沒有答案。
    """
    if verdict.scam_probability is None:
        probability = UNDETERMINED_CELL
    else:
        probability = f"{verdict.scam_probability:.0%}"
    scam_type = UNDETERMINED_CELL if verdict.scam_type is None else escaped(verdict.scam_type.value)
    return (
        '<div class="verdict-row">'
        '<div class="cell"><div class="cell-label">詐騙機率</div>'
        f'<div class="cell-value">{probability}</div></div>'
        '<div class="cell"><div class="cell-label">信心值</div>'
        f'<div class="cell-value">{verdict.confidence:.2f}</div>'
        '<div class="cell-note">系統對本次判定的自我信心，不是命中率或準確率</div></div>'
        '<div class="cell"><div class="cell-label">詐騙類型</div>'
        f'<div class="cell-value">{scam_type}</div></div>'
        "</div>"
    )


def render_evidence(result: CheckResult, doc: Document) -> str:
    """把證據座標展開成指向輸入中該句的連結。

    **無效座標讓 `KeyError` 向上傳播，不吞。** 無效座標代表產生它的檢查算錯了；
    吞掉它會讓一個算錯座標的檢查看起來只是少一條依據。
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
    """偵測項目逐筆顯示，**不過濾未命中者**，最後補上明確不註冊的項目。

    行數由 `Verdict.checks` 決定，新增一項檢查不需要修改本函式。
    """
    rows = []
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


def pii_counts(sentence: str) -> list[tuple[str, int]]:
    """某個正規化句子命中的個資類型與筆數，依類型名稱排序。"""
    counts: dict[str, int] = {}
    for span in find_pii(sentence):
        counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
    return sorted(counts.items(), key=operator.itemgetter(0))


def render_pii_tags(sentence: str) -> str:
    """句子層級的個資標籤，掛在該句之後。**不改寫文字。**

    粒度只到句子是資料結構上的限制：`find_pii()` 的輸入是**正規化後**的句子，
    回報的區間在正規化座標系裡，而畫面顯示的是 `raw_at()` 的原文片段；要把區間
    映到原文需要句內的字元對映，而今天的 `Document` 沒有保留它。
    絕不以在原文中搜尋 PII 片段來補救 —— NFKC 有一對多與多對一，全形數字或
    零寬字元時必然搜不到，於是規避手法會同時關掉個資標註。
    """
    return "".join(
        f'<span class="pii-tag">{escaped(pii_type)} ×{count}</span>'
        for pii_type, count in pii_counts(sentence)
    )


def render_messages(messages: list[Message], doc: Document) -> str:
    """輸入回顯，逐句渲染並帶錨點，讓依據的座標連結有地方可去。

    座標不存在於 `Document` 的訊息標示為「未納入本次判定」—— 它還在畫面上，
    但系統其實沒讀到它。不顯示的話，使用者會以為系統看過全部。
    """
    blocks = []
    for message_index, message in enumerate(messages):
        positions = doc.message_range(message_index)
        if not positions:
            blocks.append(
                '<div class="bubble dropped">'
                f'<div class="who">第 {message_index + 1} 則 · 未納入本次判定</div>'
                f'<div class="lines">{escaped(message.text)}</div></div>'
            )
            continue
        parts = []
        for position in positions:
            coord = doc.coords[position]
            parts.append(
                f'<span class="sentence" id="{anchor_id(coord)}">'
                f"{escaped(doc.raw_sentences[position])}</span>"
                f"{render_pii_tags(doc.sentences[position])}"
            )
        blocks.append(
            '<div class="bubble">'
            f'<div class="who">第 {message_index + 1} 則</div>'
            f'<div class="lines">{"".join(parts)}</div></div>'
        )
    return "".join(blocks)


def render_answer(verdict: Verdict, check_count: int) -> str:
    """判定結果：依據與建議動作，兩段都**逐字取自 `Verdict`**。

    依據為空時陳述的是**組裝層的事實**（註冊了幾項檢查、全部未命中），
    不是對訊息的推測。這兩句都可驗證，而且每落地一項檢查就會自動改變。
    """
    if verdict.evidence:
        grounds = "".join(f"<li>{escaped(line)}</li>" for line in verdict.evidence)
        grounds_block = f"<h4>依據</h4><ul>{grounds}</ul>"
    else:
        grounds_block = (
            f"<h4>依據</h4><p>{check_count} 項檢查全部未命中，本次沒有產生任何依據。</p>"
        )
    if verdict.actions:
        actions = "".join(f"<li>{escaped(line)}</li>" for line in verdict.actions)
        actions_block = f"<h4>建議動作</h4><ul>{actions}</ul>"
    else:
        actions_block = ""
    return f'<div class="answer">{grounds_block}{actions_block}</div>'


def render_panel(verdict: Verdict, doc: Document, check_count: int) -> str:
    """偵測項目與理由。"""
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
        "</div>"
    )


def analyze(text: str) -> str:
    """頁面的唯一入口。回傳 JSON 字串，四個鍵各對應畫面上的一塊。

    回傳 JSON 而不是一整塊 HTML：四塊各有自己的位置與捲動行為，由 `index.html`
    決定；把版面塞進這裡會讓兩邊都要知道對方的結構。
    """
    request = build_request(text)
    verdict = detect(request, REGISTRY, TABLE, limits=LIMITS)
    document = build_document(request.messages, LIMITS)
    check_count = len(REGISTRY.enabled())
    return json.dumps(
        {
            "verdict_row": render_verdict_row(verdict),
            "answer": render_answer(verdict, check_count),
            "panel": render_panel(verdict, document, check_count),
            "echo": render_messages(list(request.messages), document),
        },
        ensure_ascii=False,
    )


def check_count() -> int:
    """已註冊的檢查數。頁面在就緒時顯示它，數字由註冊表產生而不是寫死在 HTML。"""
    return len(REGISTRY.enabled())
