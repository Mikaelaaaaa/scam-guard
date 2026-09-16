"""Gradio 介面 —— 偵測核心的第一個看得見的消費者。

本檔是 Gradio 相關程式碼的**唯一**容身處，位置由兩個既有約束指定：
`pyproject.toml` 的 `banned-api` 對 `gradio` 的訊息明寫「Gradio 相關程式碼請放在
`app.py`」、`per-file-ignores` 只豁免 `"app.py"`；HuggingFace Spaces 亦固定執行
repo 根目錄的 `app.py`。

三層結構，由內而外：

    scam_guard.detect()      偵測核心（本檔不修改它一行）
        ↓
    demo_ui                  標記層（純 Python，無 gradio，兩份介面層共用）
        ↓
    ├─ 模式二「這是詐騙嗎」：直接顯示，MUST NOT 經過模型
    └─ 模式一「詐騙對練」：可選的善良市民 persona 生成層，受三條前綴條件約束

交棒事項：文案與標記現在的中繼站是 `demo_ui.py`（`docs/pages_app.py` 也 import
它）。終點不變 —— `add-verdict-render` 在 `scam_guard/` 內持有文案層，
`Verdict.evidence` 與 `Verdict.actions` 本來就是它產的；`demo_ui` 留下的是標記，
那一層不該進核心。

更新順序是 requirement 而非實作細節：判定卡與排行 MUST 在 persona 產生任何字元之前
完成更新，受害方的回應 MUST 以 streaming 呈現。因此送出的 handler 是 generator
function，第一次 `yield` 已含完整的判定卡。
"""

import importlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr

import demo_ui
from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.llm.check import LlmCheck
from scam_guard.llm.prompt import DEFAULT_BUDGET
from scam_guard.llm.validate import SCAM_SIGNAL, LlmOutcomeCounter
from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document
from scam_guard.pii import find_pii
from scam_guard.pipeline import detect
from scam_guard.redact import RedactedText
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.ngram import load_model, register_ngram_check
from scam_guard.types import Message, Request, Verdict
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import load_tables, register_url_checks
from scam_guard.weights import load_weights

# ---------------------------------------------------------------------------
# 組裝層：註冊、上限、可選依賴的注入點
# ---------------------------------------------------------------------------

LIMITS: Limits = DEFAULT_LIMITS
"""上限的單一來源。**同一個變數**同時傳給 `detect()` 與 `build_document()`。

兩處用不同的 `Limits` 會產生不同的丟棄則數，於是同一個座標 `(3, 0)` 在兩邊
指向不同的句子 —— `detect()` 的 docstring 為此警告過「座標系必須有唯一的產生者」。
"""

SENDER_THEM = "them"
"""模式一的發送者標示。`project.md` 的輸入契約範例寫的就是 `"from": "them"`。

這是**偵測語義**（被檢查的那一方），與版面的左右無關 —— 見 `demo_ui.CSS` 裡
氣泡對齊那一段。
"""

SAMPLES_PATH = Path(__file__).with_name("demo_samples.json")
ASSETS_PATH = Path(__file__).with_name("assets")
SCAMMER_AVATAR = "/gradio_api/file=assets/2.png"
PERSONA_AVATAR = "/gradio_api/file=assets/3.png"
gr.set_static_paths(paths=[ASSETS_PATH])

PiiRecognizer = demo_ui.PiiRecognizer
PiiSpan = demo_ui.PiiSpan
Polisher = Callable[[Sequence[str]], Iterator[str]]
TranscriptLogger = Callable[[RedactedText], None]
"""記錄器的簽章。**參數型別就是許可** —— `RedactedText` 是系統裡唯一可以寫進
log 的東西（`scam_guard/redact.py`：「外層拿得到這個型別的實例，就等於拿到許可」），
於是「把原文寫進 log」在型別上不可表達，不必靠一句叮嚀。"""


def recognize_pii(sentence: str) -> list[demo_ui.PiiSpan]:
    """把 `scam_guard.pii` 的輸出轉成標記層要的 `(start, end, type)`。

    標記層只認 tuple，不認 `scam_guard.pii.PiiSpan` —— 它同樣接受
    `docs/pages_app.py` 那側掛上的辨識器，而那個注入點的契約
    （`add-pii-recognizers`）本來就是三元組。

    **與 `docs/pages_app.py` 的同名函式逐字相同，而且刻意不搬進 `demo_ui`。**
    搬進去會讓標記層直接相依 `scam_guard.pii`，於是「有沒有開啟個資標註」
    從**部署的選擇**變成**標記層的預設值**，而下面那個 `None` 的狀態就再也
    表達不出來了。兩行重複換的是一個不可能被誤設成「總是開啟」的結構。
    """
    return [(span.start, span.end, span.entity_type) for span in find_pii(sentence)]


PII_RECOGNIZER: PiiRecognizer | None = recognize_pii
"""個資辨識器。四條辨識器是純標準庫、就在同一個 wheel 裡，所以這一側掛得上 ——
`docs/pages_app.py` 早就掛著同一個 `find_pii`，兩份介面層對同一份能力
不該給出兩種答案。

未掛載（`None`）時畫面顯示「沒有開啟」，MUST NOT 顯示「沒有找到」：
前者是沒有人在看，後者是看過了沒有。"""

POLISHER: Polisher | None = None
"""模式一的 persona 生成層。輸入只有偵測結果 XML，**簽章中沒有訊息原文**。"""

TRANSCRIPT_LOGGER: TranscriptLogger | None = None
"""**兩個模式共用**的記錄器。以掛載與否表達而非布林開關 —— 一個預設為 False
的布林是一個可以被貼進設定檔、看起來像是有人想過的值；而「沒有掛上」是不可能
被誤解的狀態。**公開部署預設不掛。** 掛上時 UI 才顯示「本模式的輸入會被記錄」，
且隱私說明跟著改口（見 `privacy_note()`）。"""


def log_transcript(verdict: Verdict) -> None:
    """兩個模式的**唯一**記錄點。未掛記錄器時什麼都不做。

    寫進去的是 `Verdict.redacted`，而且只有它。記錄發生在 `detect()` **之後**，
    副作用是被 `Limits` 丟棄的最舊訊息不進 log —— 那是正確的：
    沒有遮蔽投影的文字沒有合法的記錄形式。

    ⚠️ 遮蔽的保證範圍是四個辨識類型，**不等於「不含個資」**：姓名、地址、
    銀行帳號、護照號碼仍然原樣留在裡面。文案 MUST NOT 把它說成已去識別化。
    """
    if TRANSCRIPT_LOGGER is not None:
        TRANSCRIPT_LOGGER(verdict.redacted)


TABLE = load_weights()
NGRAM_MODEL = load_model()


# ---------------------------------------------------------------------------
# URL 層與必備 LLM 層的組裝 —— 全部 module-level，一次性解析，不進 per-request 路徑
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).with_name("data")
PSL_DIR = DATA_DIR / "psl"
BLOCKLIST_DIR = DATA_DIR / "blocklist"
ALLOWLIST_DIR = DATA_DIR / "allowlist"

PSL_MAX_AGE_DAYS = 90
"""PSL 快照新鮮度上限。取自 `docs/pages_app.py` 的 `PSL_MAX_AGE_DAYS` ——
同一份資料在本機與瀏覽器兩個部署上不該有兩個門檻。"""

BLOCKLIST_MAX_AGE_DAYS = {"176455": 60, "165027": 60}
"""165 涉詐網址黑名單新鮮度上限，逐 source。取自 `api/app.py` 的
`BLOCKLIST_MAX_AGE_DAYS`。160055 標記為 `retired`，`_check_freshness` 會跳過它，
不列入對應表。"""

ALLOWLIST_MAX_AGE_DAYS = 30
"""Tranco 白名單新鮮度上限。取自 `RankAllowlist.load()` 的 docstring。"""

LLM_DEADLINE_S = 45.0
"""LLM 判讀期限。取自 `llm_runtime/__init__.py` 的組裝範例。涵蓋不了 prefill
（見 `LlmRuntime.generate` 的 docstring），對 prefill 設限的是 `PromptBudget`。"""

GGUF_ENV = "SCAM_GUARD_GGUF"
"""本機模型來源的環境變數名，指向本機 GGUF 檔。與 `tests/test_llm_grammar.py`、
`.github/workflows/ci.yml` 既有的約定同名。"""

DOMAIN_AGE_UNREGISTERED: demo_ui.UnregisteredCheck = (
    "domain_age",
    "網域年齡查詢",
    "需要向網域註冊局查詢，本服務不對外連線",
)
"""唯一一筆與載入結果無關的未註冊紀錄：本服務不對外連線是這個載體的性質。"""

LLM_UNREGISTERED: demo_ui.UnregisteredCheck = (
    SCAM_SIGNAL,
    "語意判讀",
    f"LLM 為必備層，尚未載入：請下載 GGUF 模型並設定 {GGUF_ENV}，並安裝 llm extra。",
)
"""LLM 未載入時的未註冊紀錄。措辭傳達「必備但未載入」（你少了一個必要的東西，
這樣補），而非「可選、沒開」——LLM 必備之後，把「沒裝模型」說成像「無需動用」會誤導。"""


def load_psl() -> PublicSuffixList | None:
    """從 `data/psl` 載入 PSL 快照。缺席或載入失敗即回 `None`，整個 URL 層不註冊。

    `data/` 是 `.gitignore` 排除的 operator-local 目錄，剛 clone 的工作區沒有它，
    缺席是**正常首次狀態**、不致命——與 `docs/pages_app.py`（PSL 缺席在 import
    階段致命）不同，因為那邊的 PSL 是建置產物、缺席代表建置壞了。

    **MUST NOT 以空 `PublicSuffixList` 代替。** 以空 PSL 註冊的 URL 層會對每個網址
    算錯可註冊網域、產出看似正常的錯誤判定——那是把大聲的缺席換成安靜的錯答。
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

    形狀與 `docs/pages_app.load_snapshots` 相同（但**不共用同一個函式**，理由見
    design：本機與瀏覽器的快照路徑、PSL 致命性、LLM 掛法皆不同）。任一載入失敗即
    `(None, None, 例外訊息原文)`，`url_blocklist` 不註冊，其餘四個 URL 檢查照常
    （它們不需要黑白名單）。只捕捉 `(FileNotFoundError, ValueError)`；
    `require_redistributable` 用預設 `True`（`app.py` 也跑在公開的 HuggingFace Spaces）。

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
        return None, None, f"165 涉詐網址名單快照無法載入：{error}"
    try:
        allowlist = RankAllowlist.load(ALLOWLIST_DIR, max_age_days=ALLOWLIST_MAX_AGE_DAYS)
    except (FileNotFoundError, ValueError) as error:
        print(
            f"[scam-guard] url_blocklist 未註冊：Tranco 白名單快照無法載入（{error}）。"
            "請執行 `python -m tools.fetch_tranco` 取得快照。",
            file=sys.stderr,
        )
        return (
            None,
            None,
            f"Tranco 排名白名單快照無法載入，涉詐網址名單因此一併不啟用"
            f"（沒有白名單時它會重現一筆已知的硬證據偽陽性，而畫面上無處申報）：{error}",
        )
    return store, allowlist, ""


def build_llm_check() -> LlmCheck | None:
    """建必備 LLM 語意層的 `LlmCheck`；未裝好時回 `None`（誠實降級）。

    掛載的兩個條件皆須滿足：`SCAM_GUARD_GGUF` 指向存在的檔案、`llm` extra 已安裝。
    處置（次序見 design）：

    - `SCAM_GUARD_GGUF` 未設定（缺席）→ `None`，語意格「未載入」，印必備提示。
    - `find_spec("llama_cpp")` 為 `None`（extra 未裝）→ `None`，印必備提示。extra 探測
      擋在 `import_module` 之前，一個殘留在環境裡的 `SCAM_GUARD_GGUF` 不會讓沒裝
      extra 的 demo 崩潰。
    - `SCAM_GUARD_GGUF` 已設定但檔案不存在（設定錯誤）→ `raise FileNotFoundError`
      指名該路徑。缺席與設定錯誤是兩個狀態，前者略過、後者 raise。

    extra 是否安裝以 `importlib.util.find_spec` 布林探測，**不用 `try`/`except import`**；
    確認 spec 存在後的 `import_module` 因此不可能拋 `ModuleNotFoundError`。
    **不呼叫 `ensure_model()`**（本機不下載 806 MB，demo 啟動不連網）。
    本機同步呼叫 `llama.cpp`，直接建 `LlmCheck`，不需 `TwoPass` / `ReplayRuntime`。
    """
    model_path = os.environ.get(GGUF_ENV)
    if not model_path:
        print(
            "[scam-guard] 語意層（LLM）未載入。LLM 為必備層：請下載 GGUF 模型並設定 "
            f"{GGUF_ENV}，且安裝 llm extra。demo 以規則版繼續（誠實降級）。",
            file=sys.stderr,
        )
        return None
    if importlib.util.find_spec("llama_cpp") is None:
        print(
            "[scam-guard] 語意層（LLM）未載入：未安裝 llm extra（llama-cpp-python）。"
            "LLM 為必備層，請安裝 llm extra。demo 以規則版繼續（誠實降級）。",
            file=sys.stderr,
        )
        return None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{GGUF_ENV} 指向的 GGUF 模型檔不存在：{path}。"
            "這是設定錯誤（已設定但檔案不在），請確認路徑或重新下載模型。"
        )
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
    """建立本介面唯一的檢查註冊表。三個既有注入點各自的缺席行為：

    - **URL 層**：`psl` 非 `None` 時以 `register_url_checks()` 註冊五個 URL 檢查
      （`store` 為 `None` 時該函式本來就不註冊 `url_blocklist`）；`psl` 為 `None`
      時完全不呼叫它，整層不註冊。
    - **黑白名單**：成對，任一缺席即 `store`／`allowlist` 皆 `None`，`url_blocklist`
      不註冊，其餘四個 URL 檢查照常。
    - **LLM 語意層**：`llm_check` 非 `None` 時註冊進**同一個** `REGISTRY`，短路、
      `prior` 傳遞、消融的 `disable()` 全由既有機制驅動（本機同步呼叫 `llama.cpp`，
      不需 `TwoPass` / `ReplayRuntime`）；為 `None` 時不註冊（誠實降級為純規則版）。
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


def build_unregistered(
    blocklist_reason: str, llm_loaded: bool
) -> tuple[demo_ui.UnregisteredCheck, ...]:
    """依載入結果組出未註冊清單（比照 `pages_app.unregistered_checks`），而非寫死。

    基底恆含 `domain_age`；黑白名單失敗時多一列 `url_blocklist`（理由為例外訊息
    原文）；LLM 未載入時多一列語意層（識別字 `llm_scam`，措辭傳達「必備但未載入」），
    讓四源面板語意格顯示「未載入」且細節欄告訴使用者為什麼、怎麼補。

    PSL 缺席、整個 URL 層不註冊時**不**為五個 URL 檢查各塞一列：網址格的「未載入」
    加上 stderr 的 `tools.fetch_psl` 提示已讓這個狀態可見且可行動。
    """
    rows: list[demo_ui.UnregisteredCheck] = [DOMAIN_AGE_UNREGISTERED]
    if blocklist_reason:
        rows.append(("url_blocklist", "涉詐網址名單比對", blocklist_reason))
    if not llm_loaded:
        rows.append(LLM_UNREGISTERED)
    return tuple(rows)


PSL = load_psl()
if PSL is not None:
    STORE, ALLOWLIST, BLOCKLIST_UNREGISTERED_REASON = load_url_snapshots(PSL)
else:
    STORE, ALLOWLIST, BLOCKLIST_UNREGISTERED_REASON = None, None, ""
LLM_CHECK = build_llm_check()

REGISTRY: CheckRegistry = build_registry(PSL, STORE, ALLOWLIST, LLM_CHECK)
"""權重表的單一來源。`detect()` 的必填參數 —— 一個有預設表的 `detect()`
會讓呼叫端在沒有表的情況下跑出一個看起來正常的結果。

模組層載入而非每次請求載入：`load_weights()` 會讀檔並驗證整張表，
放進請求路徑等於每則訊息都重讀一次 TOML。它同時在 import 時就驗證
`REGISTRY` 的每個檢查都登記在表中 —— 缺漏會在啟動時炸，不是在第一次命中時。
"""

UNREGISTERED_CHECKS: tuple[demo_ui.UnregisteredCheck, ...] = build_unregistered(
    BLOCKLIST_UNREGISTERED_REASON, LLM_CHECK is not None
)
"""組裝層**明確知道其存在、但選擇不註冊**的檢查：識別字、中文名與一行理由，
依載入結果在 import 時算出。清單要留著：少一個訊號要在畫面上看得見，否則
「沒有訊號」與「沒有資料」長得一模一樣。"""


# ---------------------------------------------------------------------------
# 範例庫
# ---------------------------------------------------------------------------

SHORT_LABEL_MAX = 6
"""`short_label` 的長度上限。標籤是一顆按鈕上的字，八顆要能換行排下。"""


@dataclass(frozen=True)
class Sample:
    """一則進版控的合成範例訊息。

    `short_label` 是畫面上那顆按鈕的字，**不從 `label` 推導**：八筆的 `label`
    是「類型：說明」的形式，在 `：` 切開會得到兩筆都叫「假借補助金」的標籤，
    而兩個一模一樣的按鈕比一個長按鈕更糟。
    """

    label: str
    short_label: str
    text: str


def load_samples(path: Path = SAMPLES_PATH) -> list[Sample]:
    """載入範例庫。

    **缺檔即 `raise`，不降級。** 這個檔案進版控，缺少它代表安裝壞了，不是一個
    正常狀態。不讀 `data/`、不內建預設清單、不回傳空清單 —— 「有 `data/` 就讀
    `data/`、沒有就用內嵌」會讓 HF Spaces 永遠走其中一條、本機永遠走另一條，
    於是兩條路徑的差異不會被任何人發現。

    `short_label` 的三個條件（存在、不超過上限、跨全部範例唯一）皆於此處驗證並
    指名該筆：重複的短標籤會產生兩顆一模一樣的按鈕，而那是一個沒有任何地方會
    報告的錯。
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
    seen: set[str] = set()
    for entry in loaded["samples"]:
        for field_name in ("label", "short_label", "text"):
            if field_name not in entry:
                raise ValueError(
                    f"範例庫的一筆資料缺少 {field_name} 欄位：{path.name}，該筆為 {entry!r}"
                )
        short_label = entry["short_label"]
        if len(short_label) > SHORT_LABEL_MAX:
            raise ValueError(
                f"範例庫的 short_label 超過 {SHORT_LABEL_MAX} 個字：{path.name}，"
                f"該值為 {short_label!r}（{len(short_label)} 個字）"
            )
        if short_label in seen:
            raise ValueError(f"範例庫的 short_label 重複：{path.name}，該值為 {short_label!r}")
        seen.add(short_label)
        samples.append(
            Sample(
                label=entry["label"],
                short_label=short_label,
                text=entry["text"],
            )
        )
    return samples


SAMPLES: list[Sample] = load_samples()


def sample_text(short_label: str) -> str:
    """以短標籤取範例的內文。未知的短標籤 `raise` 並指名該值。

    不回傳空字串、不退回第一筆 —— 標籤與範例庫出自同一個 `SAMPLES`，對不上
    代表其中一邊被改壞了。
    """
    for sample in SAMPLES:
        if sample.short_label == short_label:
            return sample.text
    raise ValueError(f"範例庫中沒有這個短標籤：{short_label!r}")


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
# Event handlers —— 全部定義於模組層，狀態以顯式參數傳入與傳出
# ---------------------------------------------------------------------------

PRACTICE_SENDER = "邪惡詐騙犯"
PRACTICE_REPLY = "善良市民"
INQUIRY_SENDER = "你貼上的訊息"


def render_practice_conversation(
    messages: Sequence[Message], replies: Sequence[str], document: Document
) -> str:
    """以兩個固定角色頭像畫出對練對話；頭像不進入台詞或判定資料。"""
    return demo_ui.render_conversation(
        messages,
        replies,
        document,
        PRACTICE_SENDER,
        PRACTICE_REPLY,
        PII_RECOGNIZER,
        PERSONA_AVATAR,
        SCAMMER_AVATAR,
    )


def practice_submit(
    text: str,
    messages: list[Message],
    replies: list[str],
    spoken: list[str],
) -> Iterator[tuple[list[Message], list[str], list[str], str, str, str, str, str]]:
    """模式一的送出處理，**generator function**。

    第一次 `yield` 帶完整的判定卡、排行與受害方那一句；其後每次 `yield` 追加
    persona 的新片段。判定卡 MUST 在模型產生任何字元之前完成更新 —— 若實作成
    「等模型跑完再一起更新」，畫面仍然正確，只是慢，而**沒有任何測試會報告
    展示效果消失**。因此測試驗證的是第一次產出的內容，不是最終畫面。

    `spoken` 是跨輪的狀態，與 `messages` / `replies` 同一個機制：它記住哪幾條
    依據已經被說過，使受害方每輪至多說一條**新**的。
    """
    if not text.strip():
        raise gr.Error("輸入為空：請輸入一則詐騙方會說的話")

    updated = practice_messages(messages, text)
    request = Request(messages=updated)
    verdict = detect(request, REGISTRY, TABLE, limits=LIMITS)
    document = build_document(request.messages, LIMITS)
    log_transcript(verdict)

    baseline, updated_spoken = demo_ui.victim_reply(verdict, spoken)
    updated_replies = [*replies, baseline]
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED_CHECKS)
    ranking = demo_ui.render_ranking(verdict.checks, len(updated))
    conversation = render_practice_conversation(updated, updated_replies, document)
    status = demo_ui.POLISH_NOT_INJECTED if POLISHER is None else demo_ui.POLISH_STREAMING
    yield updated, updated_replies, updated_spoken, conversation, card, ranking, status, ""

    if POLISHER is None:
        return

    validator = demo_ui.PolishValidator(demo_ui.verdict_segments(verdict), verdict)
    try:
        generated = iter(POLISHER([demo_ui.practice_prompt(verdict, TABLE)]))
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
        updated_replies[-1] = baseline
        yield (
            updated,
            updated_replies,
            updated_spoken,
            render_practice_conversation(updated, updated_replies, document),
            card,
            ranking,
            demo_ui.POLISH_FAILED,
            "",
        )
        return
    while True:
        try:
            chunk = next(generated)
        except StopIteration:
            break
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            updated_replies[-1] = baseline
            yield (
                updated,
                updated_replies,
                updated_spoken,
                render_practice_conversation(updated, updated_replies, document),
                card,
                ranking,
                demo_ui.POLISH_FAILED,
                "",
            )
            return
        if not validator.feed(chunk):
            updated_replies[-1] = baseline
            yield (
                updated,
                updated_replies,
                updated_spoken,
                render_practice_conversation(updated, updated_replies, document),
                card,
                ranking,
                demo_ui.POLISH_DISCARDED,
                "",
            )
            return
        updated_replies[-1] = validator.text
        yield (
            updated,
            updated_replies,
            updated_spoken,
            render_practice_conversation(updated, updated_replies, document),
            card,
            ranking,
            demo_ui.POLISH_STREAMING,
            "",
        )
    yield (
        updated,
        updated_replies,
        updated_spoken,
        render_practice_conversation(updated, updated_replies, document),
        card,
        ranking,
        demo_ui.POLISH_ACCEPTED,
        "",
    )


def practice_from_sample(
    short_label: str,
    messages: list[Message],
    replies: list[str],
    spoken: list[str],
) -> Iterator[tuple[list[Message], list[str], list[str], str, str, str, str, str]]:
    """點一顆範例標籤就送出那一則，一步完成。

    標籤的身分走**資料**進 handler（`gr.State` 常數），不走捕獲：
    以 lambda 或巢狀 `def` 捕獲迴圈變數，八顆按鈕會全部送出最後一筆，
    而那個錯在每顆按鈕上看起來都「有反應」。
    """
    yield from practice_submit(sample_text(short_label), messages, replies, spoken)


def inquiry_submit(text: str) -> tuple[str, str]:
    """模式二的送出處理。

    **不呼叫 persona 生成層，即使已注入。** 這是產品路徑，使用者在問一個關於自己安危的
    問題，模型在這條路徑上連措辭都不經手 —— 這是「LLM 不能有最終話語權」最直接
    的實作。而且模式二未來要給 LINE 用，那裡沒有串流可以展示速度差。

    **記錄與模式一走同一個點**（`log_transcript()`），寫入的只有
    `Verdict.redacted` 的 `sentences` / `coords` / `counts`，
    MUST NOT 寫入 `Request`、`Message.text` 或 `Verdict.evidence`。
    公開部署沒有掛記錄器，那時這一行什麼都不做。
    """
    request = build_inquiry_request(text)
    verdict = detect(request, REGISTRY, TABLE, limits=LIMITS)
    document = build_document(request.messages, LIMITS)
    log_transcript(verdict)
    return (
        demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED_CHECKS),
        demo_ui.render_conversation(
            request.messages, (), document, INQUIRY_SENDER, PRACTICE_REPLY, PII_RECOGNIZER
        ),
    )


def inquiry_from_sample(short_label: str) -> tuple[str, str, str]:
    """點一顆範例標籤就填入並判定，一步完成。標籤的身分以參數傳入，不捕獲。"""
    text = sample_text(short_label)
    card, echo = inquiry_submit(text)
    return text, card, echo


def inquiry_layout_submit(text: str) -> tuple[str, str, str]:
    """Gradio 版面用輸出：同一份共用標記分別投影到分析與細節欄。"""
    card, echo = inquiry_submit(text)
    panel, details = demo_ui.split_result(card)
    return panel, details, echo


def inquiry_layout_from_sample(short_label: str) -> tuple[str, str, str, str]:
    """範例標籤的一步操作，另帶右欄所需的同源標記。"""
    text, card, echo = inquiry_from_sample(short_label)
    panel, details = demo_ui.split_result(card)
    return text, panel, details, echo


def practice_layout_submit(
    text: str,
    messages: list[Message],
    replies: list[str],
    spoken: list[str],
) -> Iterator[tuple[list[Message], list[str], list[str], str, str, str, str, str, str]]:
    """對練版面用串流：保留既有輸出順序，末端加上同源的右欄標記。"""
    for output in practice_submit(text, messages, replies, spoken):
        panel, details = demo_ui.split_result(output[4])
        yield (*output[:4], panel, *output[5:], details)


def practice_layout_from_sample(
    short_label: str,
    messages: list[Message],
    replies: list[str],
    spoken: list[str],
) -> Iterator[tuple[list[Message], list[str], list[str], str, str, str, str, str, str]]:
    """對練範例的版面投影。"""
    yield from practice_layout_submit(sample_text(short_label), messages, replies, spoken)


# ---------------------------------------------------------------------------
# 介面組裝
# ---------------------------------------------------------------------------

HEADER = """
<div class="sg-head">
<h1>這是詐騙嗎</h1>
<p>貼上你收到的可疑訊息，看看它像不像詐騙、依據是什麼，以及你現在可以做什麼。</p>
</div>
"""

PRACTICE_NOTE = (
    '<div class="note">你扮演「邪惡詐騙犯」打字，「善良市民」由系統扮演。'
    "對方每一輪只會講一條新看到的訊號，<b>聊不起來是正常的</b>。</div>"
)

SYNTHETIC_SAMPLE_NOTE = (
    '<p class="note">這些是示範用的合成範例，用來展示系統怎麼判讀；'
    "系統對真實訊息的實際表現見評估報告。</p>"
)


def transcript_notice(logger: TranscriptLogger | None) -> str:
    """記錄器已掛上時的顯著標示，**兩個模式都掛**。沒掛時不顯示。

    沒掛時回空字串而不是一句「本模式不會記錄」：那兩個狀態的差別要在畫面上
    看得見，而「什麼都沒說」與「說了會記錄」已經是兩個不同的畫面。

    ⚠️ 文案 MUST NOT 說成「已去識別化」：寫進記錄之前只蓋掉四個辨識類型，
    姓名與地址原樣留著。
    """
    if logger is None:
        return ""
    return (
        '<div class="notice">本模式的輸入會被記錄：'
        "你在這裡打的每一則訊息都會被寫進伺服器的記錄檔，寫進去之前會先蓋掉"
        "身分證字號、手機號碼、市話與信用卡號四種。"
        "<b>姓名、地址與銀行帳號不在這四種裡面</b>，會原樣留在記錄裡。"
        "不要在這裡貼上真實的個人資料。</div>"
    )


def build_demo() -> gr.Blocks:
    """組裝介面。

    `gr.Blocks` 而非 `gr.Interface`：flagging（使用者按下 flag 會把原文連同輸出
    寫進 `.gradio/flagged/dataset.csv`，一個落地的檔案）只存在於 `gr.Interface`，
    Blocks 沒有這條路徑。這是顯式的選擇而不是預設值，連同 `analytics_enabled=False`
    一起構成「框架內建的提交功能已關閉」。

    兩個模式是兩個 `gr.Tab`，各自持有自己的 `gr.State` 與輸出元件 ——
    狀態隔離因此是結構上的，不靠任何清空邏輯維持。

    **不嘗試鎖定明暗模式。** `gr.Blocks` 沒有 theme 參數，`launch(theme=)` 接的是
    調色盤不是明暗模式，而使用者可以在頁尾的設定面板自己切，設定 persist 在
    瀏覽器。唯一的解法是不寫死顏色 —— 見 `demo_ui.CSS`。
    """
    with gr.Blocks(title="這是詐騙嗎 · scam-guard", analytics_enabled=False) as demo:
        gr.HTML(HEADER)

        with gr.Tabs(elem_classes="demo-tabs"):
            with gr.Tab("這是詐騙嗎"):
                gr.HTML(transcript_notice(TRANSCRIPT_LOGGER))
                inquiry_card = gr.HTML(elem_classes="analysis-slot")
                with gr.Row(elem_classes="demo-columns"):
                    with gr.Column(elem_classes="input-column"):
                        inquiry_input = gr.Textbox(
                            label="貼上你收到的訊息",
                            lines=5,
                            placeholder="把整則訊息貼進來。若是一段轉傳的對話，請在每則之間空一行。",
                        )
                        with gr.Row(elem_classes="chips"):
                            inquiry_chips = [
                                gr.Button(sample.short_label, size="sm", scale=0)
                                for sample in SAMPLES
                            ]
                        gr.HTML(SYNTHETIC_SAMPLE_NOTE)
                        inquiry_send = gr.Button("看看這是不是詐騙", variant="primary")
                        inquiry_echo = gr.HTML()
                    with gr.Column(elem_classes="detail-column"):
                        inquiry_details = gr.HTML()

            with gr.Tab("詐騙對練"):
                gr.HTML(PRACTICE_NOTE)
                gr.HTML(transcript_notice(TRANSCRIPT_LOGGER))
                practice_card = gr.HTML(elem_classes="analysis-slot")
                with gr.Row(elem_classes="demo-columns"):
                    with gr.Column(elem_classes="input-column"):
                        practice_conversation = gr.HTML()
                        practice_input = gr.Textbox(
                            label="邪惡詐騙犯（你）",
                            lines=2,
                            placeholder="打一句詐騙方會說的話",
                        )
                        with gr.Row(elem_classes="chips"):
                            practice_chips = [
                                gr.Button(sample.short_label, size="sm", scale=0)
                                for sample in SAMPLES
                            ]
                        gr.HTML(SYNTHETIC_SAMPLE_NOTE)
                        practice_send = gr.Button("送出", variant="primary")
                        practice_status = gr.HTML(
                            f'<div class="note">{demo_ui.POLISH_NOT_INJECTED}</div>'
                        )
                    with gr.Column(elem_classes="detail-column"):
                        practice_details = gr.HTML()
                        practice_ranking = gr.HTML()
                practice_messages_state = gr.State([])
                practice_replies_state = gr.State([])
                practice_spoken_state = gr.State([])

        practice_inputs = [practice_messages_state, practice_replies_state, practice_spoken_state]
        practice_outputs = [
            practice_messages_state,
            practice_replies_state,
            practice_spoken_state,
            practice_conversation,
            practice_card,
            practice_ranking,
            practice_status,
            practice_input,
            practice_details,
        ]
        inquiry_outputs = [inquiry_card, inquiry_details, inquiry_echo]

        inquiry_send.click(inquiry_layout_submit, inputs=[inquiry_input], outputs=inquiry_outputs)
        practice_send.click(
            practice_layout_submit,
            inputs=[practice_input, *practice_inputs],
            outputs=practice_outputs,
        )
        # 標籤的身分以一個常數 `gr.State` 進 handler。迴圈裡不定義任何函式 ——
        # 捕獲迴圈變數的寫法會讓八顆按鈕全部送出最後一筆。
        for sample, chip in zip(SAMPLES, inquiry_chips):
            chip.click(
                inquiry_layout_from_sample,
                inputs=[gr.State(sample.short_label)],
                outputs=[inquiry_input, *inquiry_outputs],
            )
        for sample, chip in zip(SAMPLES, practice_chips):
            chip.click(
                practice_layout_from_sample,
                inputs=[gr.State(sample.short_label), *practice_inputs],
                outputs=practice_outputs,
            )

    return demo


if __name__ == "__main__":
    # Gradio 6 把 `css` 從 Blocks 的建構子移到 `launch()`。
    build_demo().launch(css=demo_ui.CSS)
