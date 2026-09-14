"""HTTP 服務本身 —— 組裝層、兩個路由，以及三個刻意不做的決定。

三層結構，由內而外：

    scam_guard.detect()      偵測核心（本檔不修改它一行）
        ↓
    api.schema               請求與回應的形狀（本檔不重新定義任何欄位）
        ↓
    本檔                     組裝、路由、狀態碼、log 政策

**組裝在 import 時執行一次**：`REGISTRY`、`TABLE` 是模組層級的常數，
跨請求共用同一個實例，不在每次請求時重建。

**啟動時多做一次 `validate_against()`，不只依賴 `detect()` 內建的那一次。**
`detect()` 每次呼叫都會驗證，理論上錯誤終究會浮現。但「終究」的意思是
**第一個打進來的真實請求** —— 對一個沒有認證、可能被監控系統定期
`GET /health` 但不會主動打 `POST /check` 的部署，一個註冊了但沒登錄於權重表的
檢查可以撐到真正有使用者送第一則訊息才炸。在 import 時多做這一次，
讓同一個錯誤在 `docker run` 或 CI 的 smoke test 階段就出現。**選會吵的那一個。**

**不註冊全域的 `except Exception` 例外處理器。** 多數 FastAPI 服務會加一個
`@app.exception_handler(Exception)`，好讓 500 的 body 也符合 `ErrorResponse`
的形狀。本專案的硬性規範是「禁止 `except Exception`、只能捕捉叫得出名字的
例外類型」，而且**沒有簡單任務的豁免**。那個裝飾器在語法上不是 `try/except`，
但它在效果上做的事完全一樣：註冊一個會攔下**任何**叫不出名字的例外、把它變成
一個看起來正常的回應的攔截器。

具體會發生什麼：`Check.__call__` 的協定已經要求「依賴外部服務的檢查 MUST
自行捕捉例外並記錄失敗原因，回傳空陣列，不得讓例外向上傳播」——一個例外真的
從 `detect()` 內部漏到這一層，代表**某處違反了它自己宣告的協定**，是一個 bug，
不是一個「本來就會發生、要優雅處理」的已知狀態。用 `ErrorResponse` 把它包起來
會讓呼叫端以為這是一種有文件的錯誤類型（畢竟它符合 schema），
而不是「這裡有一個違反不變式的 bug，需要有人去修」。

不寫 handler 的後果：Starlette 在 `debug=False`（正式環境的預設）時回一個
純文字的 500，不含 traceback、不含請求內容。這已經滿足「不外洩內部細節」——
**那個安全網不是本層寫的程式碼，是框架本來就有的行為。**
代價老實講：這條路徑上的回應不是 `ErrorResponse` 形狀，
`add-line-adapter` 須把「非 2xx 且非 `ErrorResponse` 形狀」當成一個獨立的
失敗分支，而不是假設全部錯誤都能解析。

**部署層要求（本層擋不住，寫在這裡讓它至少被讀到）：**
本服務**不做認證、不做速率限制、不做版本協商**。一個無認證的公開端點是一個
免費的「這則訊息是不是詐騙」神諭，詐騙方可以拿它逐句調整話術直到分數掉下門檻。
因此本服務 **MUST NOT 在沒有網路層存取控制的情況下直接公開暴露**。
在應用層發明 token 格式是「不做沒被要求的抽象」要擋的東西；
不假裝這一層擋住了，才是誠實的處置。
"""

import logging
import time

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from api.errors import validation_error_handler
from api.health import HealthResponse, build_health
from api.limits import BodySizeLimitMiddleware
from api.schema import CheckRequest, CheckResponse, verdict_to_response
from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.normalize import DEFAULT_LIMITS, Limits
from scam_guard.pipeline import detect
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import Message, Request
from scam_guard.weights import WeightTable, load_weights

LOGGER = logging.getLogger(__name__)

LIMITS: Limits = DEFAULT_LIMITS
"""上限的單一來源，與 `app.py` 同一個形狀。傳給 `detect()`，由它傳給
`build_document()` —— 座標系必須有唯一的產生者。"""

UNREGISTERED_CHECKS: tuple[tuple[str, str], ...] = (
    ("domain_age", "未注入外部查詢解析器"),
    ("url_blocklist", "未載入 165 涉詐網址黑名單快照"),
    ("url_shortener", "未載入 Public Suffix List 快照"),
    ("url_tld_risk", "未載入 Public Suffix List 快照"),
    ("url_host_shape", "未載入 Public Suffix List 快照"),
    ("url_brand", "未載入 Public Suffix List 快照"),
)
"""組裝層**明確知道其存在、但選擇不註冊**的檢查，以及不註冊的理由。
此清單經 `GET /health` 對外可讀 —— 一個沒有人看得到的顯式決定等於沒有決定。

`domain_age` 的理由需要單獨說明，因為它不是資料可用性問題。`app.py` 對
Gradio demo 的論證是：那個 Space 任何人都能開，注入 RDAP 之後每個訪客的每個
網域都從共用出口 IP 發出查詢，「顯式做的決定」在公開端點上換了承擔後果的主體
（從「開發者選擇要不要查」變成「任何路人都能觸發一次對外查詢」）。

這個 PR 面對的是同一種公開端點，但理由要重新檢驗而不是照抄 —— 檢驗完發現
**更強，不是更弱**：本服務不做認證，所以 `POST /check` 比 HF Spaces 的 Gradio
demo 還要公開一階（Gradio 至少要有人打開那個網頁）。共用出口 IP 的論證原封不動
成立，而觸發門檻更低。

失效條件：出現網路層存取控制，或本服務改為要求認證時，這個決定要重新評估。
但**不預先為它開一個設定項** —— 沒有認證機制就不該有「已認證時才啟用」的分支，
那是在為一個不存在的功能寫程式碼。

URL 層五個檢查的理由是另一回事：它們需要一份 `PublicSuffixList`
（`tools/fetch_psl.py` 的產物，落在版控排除的 `data/`），那是部署環境的資料
可用性問題。`register_url_checks()` 的注入點在 `build_registry()` 裡。
"""

BLOCKLIST_STORE: BlocklistStore | None = None
"""165 涉詐網址黑名單。以注入表達而非布林開關，形狀同 `app.py` 的可選依賴。

未注入時 `url_blocklist` 不註冊 —— 不註冊一個永遠不命中的空檢查，
否則「沒有訊號」與「沒有資料」在 `Verdict.checks` 裡看起來一模一樣。
"""

BLOCKLIST_MAX_AGE_DAYS = 60
"""黑名單的新鮮度門檻，**單一來源**：載入時傳給 `BlocklistStore.load()`，
健康檢查每次呼叫時拿同一個數字重算。兩處各自寫一個數字會在其中一個被調整時
安靜地不同步。

`BlocklistStore.load()` 刻意不給 `max_age_days` 預設值，理由是三個資料集的
宣告更新頻率都是「不定期更新」，沒有任何一個數字可以從公告推出來 ——
所以這個值是**部署參數**，由組裝層決定，不是核心可以替我們決定的事。
"""


def build_registry() -> CheckRegistry:
    """建立本服務唯一的檢查註冊表。每落地一項檢查，此處多一行。

    顯式列出要註冊的檢查，比照 `app.py`。不註冊的檢查與理由在
    `UNREGISTERED_CHECKS` —— 兩張表合起來才是完整的決定。
    """
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    return registry


REGISTRY: CheckRegistry = build_registry()
TABLE: WeightTable = load_weights()
TABLE.validate_against(REGISTRY)

app = FastAPI(
    title="scam-guard",
    description=(
        "台灣中文詐騙訊息偵測。`POST /check` 一律回 200，**包含拒答** —— "
        "拒答是系統正確地判斷自己沒有足夠依據，那是一次成功的推論不是一次失敗。"
        "請讀 `abstained` 欄位，不要對 `scam_probability` 做數值比較。"
    ),
    debug=False,
)
app.add_middleware(BodySizeLimitMiddleware)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@app.post("/check", response_model=CheckResponse)
def check(payload: CheckRequest) -> CheckResponse:
    """一次判定。合法請求一律 200，拒答亦然。

    log 政策：**MUST NOT 記錄請求 body 或回應的 `evidence` / `actions`**
    （後兩者依 `add-api-schema` 的決定可能含原文片段）。記的是延遲、
    `abstained`、`confidence` 與類型的**成員名** —— 不記中文 `label`，
    兩者揭露的資訊量相同，但成員名是我們自己的詞彙表而中文值是 165 的官方名稱，
    沒有理由選後者。

    記 `abstained` 的理由是 `add-score-compute` 定的「FPR MUST NOT 在沒有
    棄權率的情況下被引用」：本服務不留存任何 `Verdict`，記下這一個布林值，
    讓日後可以從 log 統計實際流量的棄權率而不必重跑偵測器。
    它不含任何訊息內容。
    """
    started = time.perf_counter()
    request = Request(
        messages=[
            Message(text=item.text, sender=item.sender, sent_at=item.sent_at)
            for item in payload.messages
        ]
    )
    verdict = detect(request, REGISTRY, TABLE, limits=LIMITS)
    response = verdict_to_response(verdict)
    LOGGER.info(
        "check abstained=%s confidence=%s scam_type=%s latency_ms=%.1f",
        response.abstained,
        response.confidence,
        None if response.scam_type is None else response.scam_type.code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """行程現在還能不能給出有意義的判定。狀態碼恆為 200。"""
    return build_health(
        REGISTRY,
        UNREGISTERED_CHECKS,
        BLOCKLIST_STORE,
        BLOCKLIST_MAX_AGE_DAYS,
        TABLE,
    )
