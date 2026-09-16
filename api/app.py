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

import importlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from api.errors import validation_error_handler
from api.health import HealthResponse, build_health
from api.limits import BodySizeLimitMiddleware
from api.schema import CheckRequest, CheckResponse, verdict_to_response
from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.llm.check import LlmCheck
from scam_guard.llm.prompt import DEFAULT_BUDGET
from scam_guard.llm.validate import SCAM_SIGNAL, LlmOutcomeCounter
from scam_guard.ngram import load_model, register_ngram_check
from scam_guard.normalize import DEFAULT_LIMITS, Limits
from scam_guard.pipeline import detect
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import Message, Request
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import load_tables, register_url_checks
from scam_guard.weights import WeightTable, load_weights

LOGGER = logging.getLogger(__name__)

LIMITS: Limits = DEFAULT_LIMITS
"""上限的單一來源，與 `app.py` 同一個形狀。傳給 `detect()`，由它傳給
`build_document()` —— 座標系必須有唯一的產生者。"""

# ---------------------------------------------------------------------------
# URL 層、分類器與必備 LLM 層的組裝 —— 全部 module-level，import 時一次性解析
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PSL_DIR = DATA_DIR / "psl"
BLOCKLIST_DIR = DATA_DIR / "blocklist"
ALLOWLIST_DIR = DATA_DIR / "allowlist"

PSL_MAX_AGE_DAYS = 90
"""PSL 快照新鮮度上限。取自 `docs/pages_app.py` 的 `PSL_MAX_AGE_DAYS` ——
同一份資料在三個部署上不該有三個門檻。"""

BLOCKLIST_MAX_AGE_DAYS = {"176455": 60, "165027": 60}
"""165 涉詐網址黑名單新鮮度上限，**逐 source 對應表**（取代原本的裸 int 60）。

`BlocklistStore.load()` 現在收的是逐 source 的門檻：176455（粒度為月的政府名單）
與 165027 各自 60 天。160055 標記為 `retired`，`_check_freshness` 會跳過它、
不列入對應表。載入時傳給 `BlocklistStore.load()`，`GET /health` 每次呼叫時拿
**同一份對應表**重算 —— 兩處各寫一份會在其一被調整時安靜地不同步。
"""

ALLOWLIST_MAX_AGE_DAYS = 30
"""Tranco 白名單新鮮度上限。取自 `RankAllowlist.load()` 的 docstring。"""

LLM_DEADLINE_S = 45.0
"""LLM 判讀期限。取自 `llm_runtime/__init__.py` 的組裝範例。涵蓋不了 prefill
（見 `LlmRuntime.generate` 的 docstring），對 prefill 設限的是 `PromptBudget`。"""

GGUF_ENV = "SCAM_GUARD_GGUF"
"""本機模型來源的環境變數名，指向本機 GGUF 檔。與 `app.py`、
`tests/test_llm_grammar.py`、`.github/workflows/ci.yml` 既有的約定同名。"""

TABLE: WeightTable = load_weights()
NGRAM_MODEL = load_model()

BLOCKLIST_STORE: BlocklistStore | None = None
"""165 涉詐網址黑名單，供 `GET /health` 讀。以注入表達而非布林開關。

模組層在 import 時由 `load_url_snapshots()` 賦上實際載入成功的 store（載入失敗
時維持 `None`）。未載入時 `url_blocklist` 不註冊 —— 不註冊一個永遠不命中的空
檢查，否則「沒有訊號」與「沒有資料」在 `Verdict.checks` 裡看起來一模一樣。
"""


def load_psl() -> PublicSuffixList | None:
    """從 `data/psl` 載入 PSL 快照。缺席或載入失敗即回 `None`，整個 URL 層不註冊。

    `data/` 是 `.gitignore` 排除的 operator-local 目錄，剛 clone 的工作區沒有它，
    缺席是**正常首次狀態**、不致命 —— 與 `docs/pages_app.py`（PSL 缺席在 import
    階段致命）不同，因為那邊的 PSL 是建置產物、缺席代表建置壞了。

    **MUST NOT 以空 `PublicSuffixList` 代替。** 以空 PSL 註冊的 URL 層會對每個網址
    算錯可註冊網域、產出看似正常的錯誤判定 —— 那是把大聲的缺席換成安靜的錯答。
    只捕捉 `(FileNotFoundError, ValueError)`，不 `except Exception`、不裸 `except`。
    """
    try:
        return PublicSuffixList.load(PSL_DIR, max_age_days=PSL_MAX_AGE_DAYS)
    except (FileNotFoundError, ValueError) as error:
        print(
            f"[scam-guard] URL 層未註冊：PSL 快照無法載入（{error}）。"
            "請執行 `python -m tools.fetch_psl` 取得快照。",
            file=sys.stderr,
        )
        return None


def load_url_snapshots(
    psl: PublicSuffixList,
) -> tuple[BlocklistStore | None, RankAllowlist | None, str]:
    """成對載入黑名單與白名單，回傳 `(store, allowlist, 未註冊理由)`。

    形狀與 `docs/pages_app.load_snapshots` / `app.py` 相同（但各自一份、不共用）。
    任一載入失敗即 `(None, None, 例外訊息原文)`，`url_blocklist` 不註冊，其餘四個
    URL 檢查照常（它們不需要黑白名單）。只捕捉 `(FileNotFoundError, ValueError)`；
    `require_redistributable` 用預設 `True`（`POST /check` 是無認證的公開端點）。

    **MUST NOT 建立空的 `BlocklistStore`。**「這個網域不在名單上」與「這裡沒有名單」
    在 `Verdict.checks` 裡長得一模一樣，前者是一次成功的推論、後者是一次缺席。
    """
    try:
        store = BlocklistStore.load(BLOCKLIST_DIR, psl, max_age_days=BLOCKLIST_MAX_AGE_DAYS)
    except (FileNotFoundError, ValueError) as error:
        print(
            f"[scam-guard] url_blocklist 未註冊：165 涉詐網址名單快照無法載入（{error}）。"
            "請執行 `python -m tools.fetch_blocklist` 取得快照。",
            file=sys.stderr,
        )
        return None, None, str(error)
    try:
        allowlist = RankAllowlist.load(ALLOWLIST_DIR, max_age_days=ALLOWLIST_MAX_AGE_DAYS)
    except (FileNotFoundError, ValueError) as error:
        print(
            f"[scam-guard] url_blocklist 未註冊：Tranco 白名單快照無法載入（{error}）。"
            "請執行 `python -m tools.fetch_tranco` 取得快照。",
            file=sys.stderr,
        )
        return None, None, str(error)
    return store, allowlist, ""


def build_llm_check() -> LlmCheck | None:
    """建必備 LLM 語意層的 `LlmCheck`；未裝好時回 `None`（誠實降級）。

    掛載的兩個條件皆須滿足：`SCAM_GUARD_GGUF` 指向存在的檔案、`llm` extra 已安裝。
    處置（與 `app.py` 同一套，各自一份）：

    - `SCAM_GUARD_GGUF` 未設定（缺席）→ `None`，`checks_unregistered` 申報一列，印提示。
    - `find_spec("llama_cpp")` 為 `None`（extra 未裝）→ `None`，印提示。extra 探測擋在
      `import_module` 之前，殘留的 `SCAM_GUARD_GGUF` 不會讓沒裝 extra 的服務崩潰。
    - `SCAM_GUARD_GGUF` 已設定但檔案不存在（設定錯誤）→ `raise FileNotFoundError` 指名該路徑。

    extra 是否安裝以 `importlib.util.find_spec` 布林探測，**不用 `try`/`except import`**；
    確認 spec 存在後的 `import_module` 因此不可能拋 `ModuleNotFoundError`。
    **不呼叫 `ensure_model()`**（不下載模型）。同步呼叫 `llama.cpp`，直接建 `LlmCheck`。
    """
    model_path = os.environ.get(GGUF_ENV)
    if not model_path:
        print(
            "[scam-guard] 語意層（LLM）未載入。LLM 為必備層：請下載 GGUF 模型並設定 "
            f"{GGUF_ENV}，且安裝 llm extra。服務以規則版繼續（誠實降級）。",
            file=sys.stderr,
        )
        return None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{GGUF_ENV} 指向的 GGUF 模型檔不存在：{path}。"
            "這是設定錯誤（已設定但檔案不在），請確認路徑或重新下載模型。"
        )
    if importlib.util.find_spec("llama_cpp") is None:
        print(
            "[scam-guard] 語意層（LLM）未載入：未安裝 llm extra（llama-cpp-python）。"
            "LLM 為必備層，請安裝 llm extra。服務以規則版繼續（誠實降級）。",
            file=sys.stderr,
        )
        return None
    module = importlib.import_module("llm_runtime.llama_cpp_runtime")
    runtime = module.LlamaCppRuntime(path)
    return LlmCheck(
        runtime=runtime,
        counter=LlmOutcomeCounter(),
        table=TABLE,
        budget=DEFAULT_BUDGET,
        deadline_s=LLM_DEADLINE_S,
    )


def build_registry(
    psl: PublicSuffixList | None,
    store: BlocklistStore | None,
    allowlist: RankAllowlist | None,
    llm_check: LlmCheck | None,
) -> CheckRegistry:
    """建立本服務唯一的檢查註冊表。四個注入點各自的缺席行為：

    - **規則三層 + n-gram 分類器**：無條件註冊（分類器模型 `ngram_model.json` 進了
      版控，沒有缺席分支）。
    - **URL 層**：`psl` 非 `None` 時以 `register_url_checks()` 註冊五個 URL 檢查
      （`store` 為 `None` 時該函式本來就不註冊 `url_blocklist`）；`psl` 為 `None`
      時完全不呼叫它，整層不註冊。
    - **黑白名單**：成對，任一缺席即 `store`／`allowlist` 皆 `None`，`url_blocklist`
      不註冊，其餘四個 URL 檢查照常。
    - **LLM 語意層**：`llm_check` 非 `None` 時註冊進**同一個** `REGISTRY`，短路、
      `prior` 傳遞全由既有機制驅動（本機同步呼叫 `llama.cpp`，不需 `TwoPass`）；
      為 `None` 時不註冊（誠實降級為純規則+分類器+URL 版）。
    """
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    register_ngram_check(registry, TABLE, model=NGRAM_MODEL)
    if psl is not None:
        register_url_checks(registry, psl, load_tables(), store=store, allowlist=allowlist)
    if llm_check is not None:
        registry.register(llm_check)
    return registry


def build_unregistered(blocklist_reason: str, llm_loaded: bool) -> tuple[tuple[str, str], ...]:
    """依載入結果組出未註冊清單（`(name, reason)` 二元組，餵 `GET /health`）。

    基底恆含 `domain_age`；黑白名單失敗時多一列 `url_blocklist`（理由為例外訊息
    原文）；LLM 未載入時多一列 `llm_scam`，措辭傳達「必備但未載入」（你少了一個
    必要的東西，這樣補），而非「可選、沒開」——LLM 必備之後，把「沒裝模型」說成
    像「無需動用」會誤導。

    `domain_age` 的理由（不做外部查詢）與資料可用性無關，是這個公開端點的性質：
    `POST /check` 不做認證，注入 RDAP 會讓任何路人都能從共用出口 IP 觸發對外查詢。

    PSL 缺席、整個 URL 層不註冊時**不**為五個 URL 檢查各塞一列：`checks_registered`
    少五項加上 stderr 的 `tools.fetch_psl` 提示已讓這個狀態可見且可行動。
    """
    rows: list[tuple[str, str]] = [("domain_age", "未注入外部查詢解析器")]
    if blocklist_reason:
        rows.append(("url_blocklist", blocklist_reason))
    if not llm_loaded:
        rows.append(
            (
                SCAM_SIGNAL,
                f"LLM 為必備層，尚未載入：請下載 GGUF 模型並設定 {GGUF_ENV}，並安裝 llm extra。",
            )
        )
    return tuple(rows)


PSL = load_psl()
if PSL is not None:
    BLOCKLIST_STORE, ALLOWLIST, BLOCKLIST_UNREGISTERED_REASON = load_url_snapshots(PSL)
else:
    ALLOWLIST, BLOCKLIST_UNREGISTERED_REASON = None, ""
LLM_CHECK = build_llm_check()

REGISTRY: CheckRegistry = build_registry(PSL, BLOCKLIST_STORE, ALLOWLIST, LLM_CHECK)
TABLE.validate_against(REGISTRY)

UNREGISTERED_CHECKS: tuple[tuple[str, str], ...] = build_unregistered(
    BLOCKLIST_UNREGISTERED_REASON, LLM_CHECK is not None
)
"""組裝層**明確知道其存在、但選擇不註冊**的檢查與理由，依載入結果在 import 時算出。
此清單經 `GET /health` 對外可讀 —— 一個沒有人看得到的顯式決定等於沒有決定。"""

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
