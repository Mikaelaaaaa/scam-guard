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

**與 `app.py` 的關係：兩者是兩個載體，但畫面上的標記只有一份。**
判定卡、偵測細節、氣泡、逸出、狀態分類與樣式表全部來自 `demo_ui`，本檔不自己
組裝任何一塊。這是先前記在這裡的那筆債的還款 —— 兩份標記各改一次，第二次就是
它的利息。`demo_ui` 不 import gradio，那正是它能在這一側被使用的前提。

**只有「這是詐騙嗎」一個模式。** `app.py` 的另一個模式（詐騙對練）的展示重點是
判定卡早於受害方台詞完成更新，也就是規則層與模型層的速度差 —— 而模型層在瀏覽器
裡根本不存在（`llama-cpp-python` 沒有 Pyodide 版本）。把一個少了對照組的對照實驗
搬上來，展示的東西就不是它原本要展示的東西了。累積命中排行同理不搬：它是對練
模式的元件。
"""

import json
import re

import demo_ui
from scam_guard.check import CheckRegistry
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.pii import find_pii
from scam_guard.pipeline import detect
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import Message, Request
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

UNREGISTERED_CHECKS: tuple[demo_ui.UnregisteredCheck, ...] = (
    ("domain_age", "網域年齡查詢", "需要向網域註冊局查詢，本頁不對外連線"),
    ("url_blocklist", "涉詐網址名單比對", "名單有十萬筆，在瀏覽器裡載入太重"),
)
"""本部署**明確知道其存在、但選擇不註冊**的檢查：識別字、中文名與一行理由。

只收錄已實作、且有規格定義「未註冊」狀態的檢查。少一個訊號要在畫面上看得見，
否則「沒有訊號」與「沒有資料」在結果裡長得一模一樣。
"""

SENDER_LABEL = "你貼上的訊息"
REPLY_LABEL = "對方"

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


def recognize_pii(sentence: str) -> list[demo_ui.PiiSpan]:
    """把 `scam_guard.pii` 的輸出轉成標記層要的 `(start, end, type)`。

    標記層只認 tuple，不認 `scam_guard.pii.PiiSpan` —— 它同樣接受 `app.py` 那側
    注入的辨識器，而那個注入點的契約（`add-pii-recognizers`）本來就是三元組。
    """
    return [(span.start, span.end, span.entity_type) for span in find_pii(sentence)]


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


def analyze(text: str) -> str:
    """頁面的唯一入口。回傳 JSON 字串，兩個鍵各對應畫面上的一塊。

    回傳 JSON 而不是一整塊 HTML：兩塊各有自己的位置與捲動行為，由 `index.html`
    決定；把版面塞進這裡會讓兩邊都要知道對方的結構。
    """
    request = build_request(text)
    verdict = detect(request, REGISTRY, TABLE, limits=LIMITS)
    document = build_document(request.messages, LIMITS)
    return json.dumps(
        {
            "card": demo_ui.render_verdict_card(
                verdict, document, TABLE, UNREGISTERED_CHECKS, recognize_pii
            ),
            "echo": demo_ui.render_conversation(
                request.messages, (), document, SENDER_LABEL, REPLY_LABEL, recognize_pii
            ),
        },
        ensure_ascii=False,
    )


def styles() -> str:
    """共用樣式表。頁面在就緒後把它寫進一個 `<style>`。

    樣式表只有一份，跟著標記走 —— `index.html` 自己只定義主題變數的調色盤，
    不重複定義任何由 `demo_ui` 產生的 class。
    """
    return demo_ui.CSS


def check_count() -> int:
    """已註冊的檢查數。頁面在就緒時顯示它，數字由註冊表產生而不是寫死在 HTML。"""
    return len(REGISTRY.enabled())
