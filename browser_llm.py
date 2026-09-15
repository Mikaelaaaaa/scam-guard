"""線上 demo 的 LLM 這一半 —— 兩趟 `detect()`、重放式 runtime，與兩個模式的一輪。

**這個模組存在的唯一理由是一個時序問題。**
`scam_guard.llm.check.LlmCheck.__call__` 是同步的，`LlmRuntime.generate()` 是同步的，
而瀏覽器裡的推論（transformers.js 的 `model.generate()`）回傳一個 Promise。
一個同步的 Python 函式等不了一個 JS Promise，除非用 `SharedArrayBuffer` +
`Atomics.wait`（要兩個 GitHub Pages 設不了的回應標頭）或 JSPI（把整個線上 demo
的可用性綁在一個引擎功能上）。兩條都否決，理由見 design。

**採用的作法是把一次判讀切成兩趟同步呼叫，中間夾一段非同步生成：**

```
第一趟（Python，同步）  只含規則層的註冊表跑 detect() → 判定卡立刻上畫面
                       build_prompt() 產出這次判讀的 prompt
                       回傳一次性 token，Request 留在 Python 側
      ↓ token + prompt 過界
生成（JavaScript）      model.generate() → 一段 raw 字串（逾時則為 TIMEOUT_SENTINEL）
      ↓ token + raw 過界
第二趟（Python，同步）  規則層 + LlmCheck(ReplayRuntime(raw)) 再跑一次 detect()
```

**`ReplayRuntime` 是全部的技巧。** 它滿足同一個 `LlmRuntime` Protocol，
`generate()` 逐字回傳建構時收到的字串。於是 `scam_guard/llm/` 的四個模組
**完全不知道生成發生在別的語言裡** —— 對它們而言那是一次普通的同步呼叫。
這正是 `LlmRuntime` 這個 Protocol 原本要換到的東西。

**待判讀的文字 MUST NOT 第二次過界。** 第一趟回傳一次性 token，`Request` 留在
Python 側；第二趟的入口簽章裡沒有訊息文字。這消除一整類錯誤：使用者在生成期間
改了輸入框，第二趟不可能拿著「對前一段文字的答案」去判讀新的文字。

**本模組不 import `js`、不 import `pyodide`、不 import `llm_runtime`、
不 import `llama_cpp`。** 它是一個純 Python 模組，在一個沒有瀏覽器、沒有 GPU、
沒有下載過任何模型的環境裡 `pytest` 直接 import 得起來 ——
`tests/test_browser_llm.py` 就是在那樣的環境裡跑的。JavaScript 那一側的知識
（WebGPU 的三個前提、dtype、revision、快取）全部留在 `docs/index.html`。

**它 import `demo_ui`**：畫面上的每一段標記只有一份實作，而那一份在那裡。
本模組不自己組任何 HTML。
"""

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import demo_ui
from scam_guard.check import CheckRegistry, Stage
from scam_guard.llm.check import LlmCheck
from scam_guard.llm.prompt import DEFAULT_BUDGET, Prompt, PromptBudget, build_prompt
from scam_guard.llm.schema import FIELD_NAMES
from scam_guard.llm.validate import (
    SCAM_SIGNAL,
    SUSPICIOUS_SIGNAL,
    LlmOutcome,
    LlmOutcomeCounter,
)
from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document
from scam_guard.pipeline import detect
from scam_guard.types import Coord, Message, Request, Verdict
from scam_guard.weights import WeightTable

# ---------------------------------------------------------------------------
# 過界的常數 —— JavaScript 那一側 MUST 向這裡要，MUST NOT 自己寫一份
# ---------------------------------------------------------------------------

ASSISTANT_PREFILL = '{"' + FIELD_NAMES[0] + '":"'
"""助理回合的預填前綴，由 `FIELD_NAMES` 產生。

transformers.js 與它下面的 ONNX Runtime Web 都沒有文法約束解碼（見
`ReplayRuntime.generate()` 的註解），所以結構保證退回「產生後驗證」。預填關掉的
是 1B 模型最常見的那一種結構失敗：在 JSON 前面先寫一段「好的，以下是我的分析：」。
它關不掉括號沒收尾、鍵名寫錯、多寫第二個物件 —— 那些照常 `json.loads` 失敗。

**欄位名由常數產生而不是手寫。** 手寫的欄位名與 `scam_guard.llm.schema` 不同步時
沒有任何機制會報告，而 JavaScript 那一側更是連 lint 都看不到這個關聯。
"""

DEADLINE_S = 90.0
"""一次判讀生成的期限，單位為秒。**這個數字由 JavaScript 那一側執行。**

`ReplayRuntime` 收到它之後忽略它（生成已經發生完畢），所以它在這裡的唯一作用是
「只有一個地方寫著這個數字」。本機版在 2 vCPU 上是 45 秒；瀏覽器裡的 WebGPU
比它快，但第一次呼叫要付 shader 編譯的成本，所以放寬一倍。**沒有實測依據。**
"""

PRACTICE_SENDER_FIELD = "them"
"""對練模式裡每一則訊息的 `sender`。

這是**偵測語義**（被檢查的那一方），與版面的左右無關 —— 與 `app.py` 的
`SENDER_THEM` 同一個值、同一個理由，`project.md` 的輸入契約範例寫的就是它。
"""

MAX_PENDING = 8
"""同時保存的判讀上限。介面在生成期間停用按鈕，實際在飛的數量遠小於此值。"""


# ---------------------------------------------------------------------------
# 重放式 runtime
# ---------------------------------------------------------------------------


class ReplayRuntime:
    """滿足 `scam_guard.llm.check.LlmRuntime`，逐字回傳建構時收到的字串。

    它是這整個設計的樞紐：有了它，`LlmCheck` 照常 `build_prompt()`、照常呼叫
    `generate()`、照常 `parse_and_validate()`，而那次「生成」其實在幾秒前發生於
    另一個語言裡。核心不需要知道這件事，也因此 `scam_guard/llm/` 一行都不用改。

    **只回答一次。** 第二次呼叫拋例外而不是重播同一個字串：一次判讀對應一次生成，
    而一個會重播的 runtime 讓「同一段輸出被算進兩次判讀」變成可能，
    那會讓 `LlmOutcomeCounter` 的分母失真。
    """

    def __init__(self, raw: str) -> None:
        self._raw = raw
        self._answered = False

    def generate(self, prompt: str, grammar: str, *, deadline_s: float) -> str:
        # `grammar` 收下就丟掉：transformers.js 的 14 個 logits processor 沒有一個是
        # 文法或 JSON schema 約束，它下面的 ONNX Runtime Web 也沒有，所以這一側
        # **沒有文法約束解碼**，結構保證退回「產生後驗證」（見 design 的 Decisions 3）。
        # 難看是刻意的：任何人讀到這一行都會問「為什麼丟掉」。MUST NOT 為了好看而
        # 把它從 Protocol 上拿掉 —— 那是對 `scam_guard/llm/check.py` 的修改。
        #
        # `deadline_s` 同樣忽略：生成已經發生完畢，一個事後的期限不是關於那次生成的
        # 事實。逾時由 JavaScript 側判定，並以 `TIMEOUT_SENTINEL` 作為 `raw` 傳進來。
        if self._answered:
            raise RuntimeError(
                "重放式 runtime 只回答一次，這是第二次呼叫："
                "一次判讀對應一次生成，重播會讓同一段輸出被算進兩次判讀"
            )
        self._answered = True
        return self._raw


# ---------------------------------------------------------------------------
# 模型層當下的狀態
# ---------------------------------------------------------------------------


class LlmState(Enum):
    """模型層在**這一次判定發生的當下**是什麼狀態。由 JavaScript 側告知。

    四個成員對應 design 的四個狀態。它們 MUST 在畫面上是四段不同的文字，
    而且沒有任何一段可以被讀成「模型檢查過且沒有發現」——
    那是 `STATUS_NO_SIGNAL` 的意思，它是第五段文字。
    """

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOAD_FAILED = "load_failed"
    READY = "ready"


STATUS_NOT_LOADED = "語意判讀尚未完成初始化：這次的判定只有規則層。"
STATUS_LOADING = "模型還在下載，這次的判定只有規則層。下載完成後再送出一次就會多一層語意判讀。"
STATUS_LOAD_FAILED = "模型載入失敗，這次的判定只有規則層。失敗的原文寫在上方的模型那一列。"
STATUS_RUNNING = "規則層已經判完（下方就是結果），語意判讀還在跑，跑完會再更新一次。"
STATUS_STRUCTURE = "這次的語意判讀沒有產生一個完整的 JSON 物件，整筆作廢，判定維持規則層的結果。"
STATUS_SEMANTIC = (
    "這次的語意判讀沒有通過語意驗證（例如指到一句不存在的句子），整筆作廢，判定維持規則層的結果。"
)
STATUS_TIMEOUT = "這次的語意判讀在期限內沒有寫完，已中止，判定維持規則層的結果。"
STATUS_SKIPPED = "規則層已經有硬證據，昂貴階段依設計短路，這次沒有用到語意判讀。"
STATUS_NO_SIGNAL = "語意判讀完成：模型讀完整段訊息，判為沒有詐騙話術，因此沒有新增任何訊號。"
STATUS_HIT = "語意判讀完成：模型判出話術，判定卡上多了一項語意訊號。"
"""畫面上關於模型層的文字，一個狀態一段。

**沒有任何一段可以被讀成「模型說沒問題」，除了 `STATUS_NO_SIGNAL`** ——
而那一段的前提是模型真的跑完、輸出真的通過了七條語意驗證。三種失敗與三種
「沒有跑」各自說自己的事，理由是它們指向不同的處置（見
`scam_guard.llm.validate.LlmOutcome` 的對照表）。
"""

_OUTCOME_STATUS: Mapping[LlmOutcome, str] = {
    LlmOutcome.STRUCTURE: STATUS_STRUCTURE,
    LlmOutcome.SEMANTIC: STATUS_SEMANTIC,
    LlmOutcome.TIMEOUT: STATUS_TIMEOUT,
}

LLM_CHECK_LABEL = "語意判讀"

_UNREGISTERED_REASON: Mapping[LlmState, str] = {
    LlmState.NOT_LOADED: "模型沒有載入，這一層在這次判定裡不存在",
    LlmState.LOADING: "模型還在下載，這次判定沒有用到它",
    LlmState.LOAD_FAILED: "模型載入失敗，這次判定沒有用到它",
    LlmState.READY: "這次的語意判讀還在跑",
}
"""模型層出現在「未提供的檢查」清單裡時的一行理由。

沿用 `docs/pages_app.py` 那個常數自己的 docstring 寫的理由：少一個訊號要在畫面上
看得見，否則「沒有訊號」與「沒有資料」在結果裡長得一模一樣。
"""

_FAILED_REASON: Mapping[LlmOutcome, str] = {
    LlmOutcome.STRUCTURE: "這次的輸出不是一個完整的 JSON 物件，整筆作廢",
    LlmOutcome.SEMANTIC: "這次的輸出沒有通過語意驗證，整筆作廢",
    LlmOutcome.TIMEOUT: "這次的生成在期限內沒有完成",
}

_SKIPPED_REASON = "規則層已有硬證據，昂貴階段依設計短路"


# ---------------------------------------------------------------------------
# outcome 計數 —— 模組層一份，兩個模式共用
# ---------------------------------------------------------------------------

COUNTER = LlmOutcomeCounter()
"""本頁面連續運作期間的四種 outcome 累計。

**模組層建立一次**，兩個模式共用：它回答的是「這一層有多少比例的時間不存在」，
而那個問題的對象是這個分頁，不是某一次判讀。
per-page 而非 per-process 的差別在這裡不存在 —— 重新整理就歸零，
與 `LlmOutcomeCounter` 自己 docstring 說的 HF Spaces 休眠是同一件事。
"""


def outcome_counts() -> dict[str, int]:
    """四種 outcome 的當下次數，鍵為 `LlmOutcome` 的值。給畫面用。

    不回傳 `failure_rate()`：那個函式在零次判讀上會拋例外（0/0 沒有意義），
    而畫面在第一次判讀之前就要能顯示這一塊。要失敗率的是量測，不是畫面，
    而量測拿得到這四個數字自己算。
    """
    return {outcome.value: count for outcome, count in COUNTER.counts().items()}


def _recorded_outcome(
    before: Mapping[LlmOutcome, int], after: Mapping[LlmOutcome, int]
) -> LlmOutcome | None:
    """這一趟記了哪一種 outcome。一種都沒記時回傳 `None`（`LlmCheck` 被短路了）。

    以差值推得而不是另設一個「最後一次的 outcome」欄位：計數器是那件事唯一的
    記錄者，第二個記錄點會有第二個真相。同時記到兩種是不可能的 ——
    `LlmCheck.__call__` 一次只 `record()` 一次 —— 所以那個情形拋例外而不是挑一個。
    """
    increased = [outcome for outcome in LlmOutcome if after[outcome] > before[outcome]]
    if not increased:
        return None
    if len(increased) > 1:
        raise RuntimeError(f"一趟判讀記了不只一種 outcome：{[o.value for o in increased]}")
    return increased[0]


# ---------------------------------------------------------------------------
# 兩趟流程
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstPass:
    """第一趟的產物。`request` 不在裡面 —— 它留在 Python 側，見模組 docstring。

    `window_coords` 是這次 prompt 實際涵蓋的句子座標。它在這裡是為了讓
    「兩趟看到的是同一個視窗」這件事可以被斷言 —— 第二趟的 `LlmCheck` 會自己
    再算一次視窗，而模型回報的證據座標是拿那一份驗的。`prompt` 為空字串時它也是空的。
    """

    token: str
    prompt: str
    window_coords: tuple[Coord, ...]
    verdict: Verdict
    document: Document


@dataclass(frozen=True)
class SecondPass:
    """第二趟的產物。

    `outcome` 為 `None` 代表 `LlmCheck` 根本沒有被執行：規則層已有硬證據，
    pipeline 依設計短路了昂貴階段。那次生成的成本因此白付 —— 這是本設計已知的
    代價，換到的是不在這裡重寫一份 `_should_short_circuit()`。
    """

    verdict: Verdict
    document: Document
    outcome: LlmOutcome | None


class TwoPass:
    """一次線上判讀的兩趟，以及那個一次性 token 的生命週期。

    **這個類別不產生任何 HTML。** 它的產物是 `Verdict` 與 `Document`，
    標記由 `demo_ui` 產生，組合由下面兩個模式類別負責。
    """

    def __init__(
        self,
        *,
        registry: CheckRegistry,
        table: WeightTable,
        limits: Limits = DEFAULT_LIMITS,
        budget: PromptBudget = DEFAULT_BUDGET,
        deadline_s: float = DEADLINE_S,
        counter: LlmOutcomeCounter = COUNTER,
    ) -> None:
        expensive = [check.name for check in registry.enabled() if check.stage is not Stage.LOCAL]
        if expensive:
            # 第一趟 MUST 不呼叫模型，而「不呼叫」要由註冊表的形狀保證，不由順序保證。
            raise ValueError(
                f"第一趟的註冊表只能含 Stage.LOCAL 的檢查，但它含有 {expensive}："
                "昂貴階段的檢查會讓第一趟等一個它不該等的東西"
            )
        self._registry = registry
        self._table = table
        self._limits = limits
        self._budget = budget
        self._deadline_s = deadline_s
        self._counter = counter
        self._pending: dict[str, Request] = {}
        self._expired: dict[str, str] = {}

    def first_pass(self, request: Request) -> FirstPass:
        """規則層那一趟。回傳判定、文件、這次判讀的 prompt 與一次性 token。

        `prompt` 為空字串代表這則訊息正規化之後一句都不剩（貼圖、純空白）——
        `build_prompt()` 對空 `Document` 拋例外，而那個例外的意思是「呼叫端算錯了」，
        不是這裡的狀態。呼叫端看到空字串就不要生成。

        每筆 request 以自己的 token 索引；另一個模式或另一輪送出不會覆蓋它。
        超過上限時淘汰最舊的一筆，避免中途離頁的 first pass 永久累積。
        """
        verdict = detect(request, self._registry, self._table, limits=self._limits)
        document = build_document(request.messages, self._limits)
        token = secrets.token_hex(8)
        if len(self._pending) >= MAX_PENDING:
            evicted = next(iter(self._pending))
            self._pending.pop(evicted)
            self._remember_expired(evicted, "已因過舊被淘汰")
        self._pending[token] = request
        prompt = self._prompt_for(verdict, document)
        return FirstPass(
            token=token,
            prompt="" if prompt is None else prompt.text,
            window_coords=() if prompt is None else tuple(prompt.window.coords),
            verdict=verdict,
            document=document,
        )

    def second_pass(self, token: str, raw: str) -> SecondPass:
        """含 `LlmCheck` 的那一趟。**簽章裡沒有訊息文字。**

        `raw` 逐字元照原樣交給 `ReplayRuntime`，由 `LlmCheck` 內部的
        `parse_and_validate()` 處置 —— 本層不 strip、不補括號、不做任何修補。
        `scam_guard.llm.check.TIMEOUT_SENTINEL` 是合法的 `raw`，它會被記成 `TIMEOUT`。
        """
        request = self._consume(token)
        registry = CheckRegistry()
        for check in self._registry.enabled():
            registry.register(check)
        registry.register(
            LlmCheck(
                runtime=ReplayRuntime(raw),
                counter=self._counter,
                table=self._table,
                budget=self._budget,
                deadline_s=self._deadline_s,
            )
        )
        before = self._counter.counts()
        verdict = detect(request, registry, self._table, limits=self._limits)
        after = self._counter.counts()
        return SecondPass(
            verdict=verdict,
            document=build_document(request.messages, self._limits),
            outcome=_recorded_outcome(before, after),
        )

    def without_model(self, token: str) -> SecondPass:
        """沒有生成發生時收掉這次判讀。**簽章裡一樣沒有訊息文字。**

        這不是「第二趟的降級版」：`LlmCheck` 根本沒有被註冊，所以 `Verdict` 與
        第一趟逐欄位相同，計數器一次都不記 —— 一次沒有發生的判讀不該出現在
        失敗率的分母裡。`outcome` 為 `None`，與被短路那一種同形（兩者都是
        「這次沒有用到模型」），畫面上的文字由 `LlmState` 分辨。
        """
        request = self._consume(token)
        return SecondPass(
            verdict=detect(request, self._registry, self._table, limits=self._limits),
            document=build_document(request.messages, self._limits),
            outcome=None,
        )

    def _prompt_for(self, verdict: Verdict, document: Document) -> Prompt | None:
        """這次判讀要送給模型的 prompt。空 `Document` 時為 `None`。

        `hit_groups` 以 `verdict.checks` 算得，而那與 `LlmCheck` 拿到的 `prior`
        是同一批結果：本註冊表只有 `Stage.LOCAL` 的檢查（`__init__` 擋住其餘），
        而 `detect()` 的 `prior` 就是全部 `LOCAL` 結果。

        ⚠️ 第二趟的 `LlmCheck` 會**再組一次** prompt，nonce 不同，所以兩段文字不同。
        這是刻意的：驗證證據座標用的是 `PromptWindow`，而視窗只由 `Document` 與
        `PromptBudget` 決定，與 nonce 無關 —— 兩趟的 `coords` 因此逐項相同。
        """
        if not document.coords:
            return None
        hit_groups = sorted(
            {self._table.group_of(result.name) for result in verdict.checks if result.hit}
        )
        return build_prompt(
            document,
            hit_groups=hit_groups,
            budget=self._budget,
            nonce=secrets.token_hex(4),
        )

    def _consume(self, token: str) -> Request:
        """取出並作廢一個 token。不認得的一律拋例外，**不回傳規則層的結果冒充成功**。"""
        if token not in self._pending:
            reason = self._expired.get(token, "從未發出、已被重送取代，或失效原因已過期")
            raise ValueError(f"判讀識別字 {token!r} 已經失效：{reason}")
        request = self._pending.pop(token)
        self._remember_expired(token, "已被消費")
        return request

    def _remember_expired(self, token: str, reason: str) -> None:
        """有界保存失效原因，讓近期 token 大聲說明是已消費或被淘汰。"""
        if len(self._expired) >= MAX_PENDING:
            self._expired.pop(next(iter(self._expired)))
        self._expired[token] = reason


# ---------------------------------------------------------------------------
# 畫面需要、而本模組不知道的東西
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Presentation:
    """由部署的組裝層提供的標記參數。

    `unregistered` 只含**與模型無關**的那幾筆；語意判讀那一筆由本模組依當下狀態
    接上去，因為只有這裡知道這次判讀發生了什麼。
    """

    table: WeightTable
    unregistered: tuple[demo_ui.UnregisteredCheck, ...]
    sender_label: str
    reply_label: str
    recognizer: demo_ui.PiiRecognizer | None = None
    reply_avatar: str | None = None
    sender_avatar: str | None = None


def _llm_hit(verdict: Verdict) -> bool:
    return any(
        result.hit and result.name in (SCAM_SIGNAL, SUSPICIOUS_SIGNAL) for result in verdict.checks
    )


def first_pass_status(state: LlmState) -> str:
    """第一趟之後關於模型層的一段文字。"""
    if state is LlmState.READY:
        return STATUS_RUNNING
    if state is LlmState.LOADING:
        return STATUS_LOADING
    if state is LlmState.LOAD_FAILED:
        return STATUS_LOAD_FAILED
    return STATUS_NOT_LOADED


def second_pass_status(result: SecondPass) -> str:
    """第二趟之後關於模型層的一段文字。"""
    if result.outcome is None:
        return STATUS_SKIPPED
    if result.outcome is LlmOutcome.OK:
        return STATUS_HIT if _llm_hit(result.verdict) else STATUS_NO_SIGNAL
    return _OUTCOME_STATUS[result.outcome]


def first_pass_unregistered(
    presentation: Presentation, state: LlmState
) -> tuple[demo_ui.UnregisteredCheck, ...]:
    """第一趟的未提供清單：語意判讀這一筆一定在，因為它還沒跑。"""
    return (*presentation.unregistered, (SCAM_SIGNAL, LLM_CHECK_LABEL, _UNREGISTERED_REASON[state]))


def second_pass_unregistered(
    presentation: Presentation, state: LlmState, result: SecondPass
) -> tuple[demo_ui.UnregisteredCheck, ...]:
    """第二趟的未提供清單。

    判讀成功時語意判讀**不在**清單裡 —— 它跑了，而「跑了而沒有命中」與
    「沒有跑」是兩件事，前者由 `Verdict.checks` 自己說。
    """
    if result.outcome is LlmOutcome.OK:
        return presentation.unregistered
    if result.outcome is None:
        reason = _SKIPPED_REASON if state is LlmState.READY else _UNREGISTERED_REASON[state]
    else:
        reason = _FAILED_REASON[result.outcome]
    return (*presentation.unregistered, (SCAM_SIGNAL, LLM_CHECK_LABEL, reason))


# ---------------------------------------------------------------------------
# 模式一：這是詐騙嗎
# ---------------------------------------------------------------------------


class InquiryMode:
    """一則訊息、一次判定。兩趟之間畫面上已經有一張完整的規則層判定卡。"""

    def __init__(self, two_pass: TwoPass, presentation: Presentation) -> None:
        self._two_pass = two_pass
        self._presentation = presentation

    def first(self, request: Request, state: LlmState) -> dict[str, object]:
        result = self._two_pass.first_pass(request)
        return {
            "token": result.token,
            "prompt": result.prompt,
            "card": self._card(
                result.verdict,
                result.document,
                first_pass_unregistered(self._presentation, state),
            ),
            "echo": self._echo(request.messages, result.document),
            "status": first_pass_status(state),
            "counts": outcome_counts(),
        }

    def second(self, token: str, raw: str) -> dict[str, object]:
        return self._render(self._two_pass.second_pass(token, raw), LlmState.READY)

    def without_model(self, token: str, state: LlmState) -> dict[str, object]:
        return self._render(self._two_pass.without_model(token), state)

    def _render(self, result: SecondPass, state: LlmState) -> dict[str, object]:
        return {
            "card": self._card(
                result.verdict,
                result.document,
                second_pass_unregistered(self._presentation, state, result),
            ),
            "status": second_pass_status(result) if state is LlmState.READY else "",
            "counts": outcome_counts(),
        }

    def _card(
        self,
        verdict: Verdict,
        document: Document,
        unregistered: Sequence[demo_ui.UnregisteredCheck],
    ) -> str:
        return demo_ui.render_verdict_card(
            verdict, document, self._presentation.table, unregistered, self._presentation.recognizer
        )

    def _echo(self, messages: Sequence[Message], document: Document) -> str:
        return demo_ui.render_conversation(
            messages,
            (),
            document,
            self._presentation.sender_label,
            self._presentation.reply_label,
            self._presentation.recognizer,
        )


# ---------------------------------------------------------------------------
# 模式二：詐騙對練
# ---------------------------------------------------------------------------


@dataclass
class _Polish:
    """一輪的 persona 生成狀態。`document` 留著是為了每一塊都能重畫對話。"""

    validator: demo_ui.PolishValidator
    baseline: str
    document: Document


class PracticeMode:
    """使用者扮演詐騙方，系統以累積的對話計算 `Verdict` 並產出受害方的一句話。

    **一輪的順序，以及它為什麼是這個順序：**

    ```
    1  first()            規則層那一趟 → 判定卡 + 排行立即更新，還沒有受害方台詞
    2  （JavaScript）      判讀生成
    3  second()           含 LlmCheck 的那一趟 → 判定卡 + 排行再更新
                          → victim_reply(最終 Verdict) → baseline 氣泡出現
    4  （JavaScript）      persona 生成，逐塊回來
    5  polish_feed()      逐塊驗證，違規即回報 False（呼叫端 MUST 中止生成）
    6  polish_end()       整句通過
    ```

    **受害方台詞取自第二趟的 `Verdict`。** `victim_reply()` 與 `PolishValidator`
    吃的是**同一個** `Verdict` 物件 —— 用兩個不同的會讓「判定卡說是假借包裹招領、
    而 persona 驗證層禁止提到假借包裹招領」這種矛盾成為可能。
    代價是台詞比規則層晚一次生成的時間出現，而那正是這個模式要展示的東西。

    **`first()` 的回傳值裡沒有受害方台詞，這是刻意的。** 更新順序是 requirement
    不是實作細節，而把它寫成「第一趟的產物裡不可能有那句話」比寫一條計時測試牢靠。
    """

    def __init__(self, two_pass: TwoPass, presentation: Presentation) -> None:
        self._two_pass = two_pass
        self._presentation = presentation
        self._messages: list[Message] = []
        self._replies: list[str] = []
        self._spoken: list[str] = []
        self._polish: _Polish | None = None

    def first(self, text: str, state: LlmState) -> dict[str, object]:
        """把這一輪的話接到對話後面，跑規則層那一趟。

        `sender` 填 `"them"`（人扮演詐騙方），`sent_at` 填**實際送出時間**而不是
        `None` —— `None` 的語意是「不知道」，而這個模式知道。
        """
        if not text.strip():
            raise ValueError("輸入為空：請輸入一則詐騙方會說的話")
        self._messages.append(
            Message(text=text, sender=PRACTICE_SENDER_FIELD, sent_at=datetime.now(tz=UTC))
        )
        self._polish = None
        result = self._two_pass.first_pass(Request(messages=list(self._messages)))
        return {
            "token": result.token,
            "prompt": result.prompt,
            "card": self._card(
                result.verdict,
                result.document,
                first_pass_unregistered(self._presentation, state),
            ),
            "ranking": demo_ui.render_ranking(result.verdict.checks, len(self._messages)),
            "conversation": self._conversation(result.document),
            "status": first_pass_status(state),
            "counts": outcome_counts(),
        }

    def second(self, token: str, raw: str) -> dict[str, object]:
        """含 `LlmCheck` 的那一趟，並由它的 `Verdict` 產出受害方台詞。"""
        return self._finish(self._two_pass.second_pass(token, raw), LlmState.READY, polished=True)

    def without_model(self, token: str, state: LlmState) -> dict[str, object]:
        """沒有模型時收掉這一輪。台詞與驗證器仍然吃同一個 `Verdict`（規則層那個）。"""
        return self._finish(self._two_pass.without_model(token), state, polished=False)

    def polish_feed(self, chunk: str) -> dict[str, object]:
        """餵入 persona 生成的一塊。`ok` 為 `False` 時呼叫端 MUST 立刻中止生成。

        違規時整句退回 `baseline`，**不只丟棄違規的那一塊** —— 前面通過的部分是
        同一次改寫的一部分，留著它等於留下半句被判定為在唬爛的話。
        """
        polish = self._require_polish()
        if not polish.validator.feed(chunk):
            self._replies[-1] = polish.baseline
            self._polish = None
            return {
                "ok": False,
                "conversation": self._conversation(polish.document),
                "polish_status": demo_ui.POLISH_DISCARDED,
            }
        self._replies[-1] = polish.validator.text
        return {
            "ok": True,
            "conversation": self._conversation(polish.document),
            "polish_status": demo_ui.POLISH_STREAMING,
        }

    def polish_end(self) -> dict[str, object]:
        """persona 生成正常結束。整句已逐字元通過三條條件。"""
        polish = self._require_polish()
        self._replies[-1] = polish.validator.text
        self._polish = None
        return {
            "conversation": self._conversation(polish.document),
            "polish_status": demo_ui.POLISH_ACCEPTED,
        }

    def polish_fail(self) -> dict[str, object]:
        """persona 生成失敗時整句退回確定性底稿，並清掉串流狀態。"""
        polish = self._require_polish()
        self._replies[-1] = polish.baseline
        self._polish = None
        return {
            "conversation": self._conversation(polish.document),
            "polish_status": demo_ui.POLISH_FAILED,
        }

    def _finish(self, result: SecondPass, state: LlmState, *, polished: bool) -> dict[str, object]:
        baseline, self._spoken = demo_ui.victim_reply(result.verdict, self._spoken)
        self._replies.append(baseline)
        self._polish = _Polish(
            validator=demo_ui.PolishValidator(
                demo_ui.verdict_segments(result.verdict), result.verdict
            ),
            baseline=baseline,
            document=result.document,
        )
        return {
            "card": self._card(
                result.verdict,
                result.document,
                second_pass_unregistered(self._presentation, state, result),
            ),
            "ranking": demo_ui.render_ranking(result.verdict.checks, len(self._messages)),
            "conversation": self._conversation(result.document),
            "polish_prompt": demo_ui.practice_prompt(result.verdict, self._presentation.table),
            "status": second_pass_status(result) if state is LlmState.READY else "",
            "polish_status": (
                demo_ui.POLISH_STREAMING if polished else demo_ui.POLISH_NOT_INJECTED
            ),
            "counts": outcome_counts(),
        }

    def _require_polish(self) -> _Polish:
        if self._polish is None:
            raise ValueError("這一輪沒有進行中的 persona 生成：違規已丟棄，或生成已結束")
        return self._polish

    def _card(
        self,
        verdict: Verdict,
        document: Document,
        unregistered: Sequence[demo_ui.UnregisteredCheck],
    ) -> str:
        return demo_ui.render_verdict_card(
            verdict, document, self._presentation.table, unregistered, self._presentation.recognizer
        )

    def _conversation(self, document: Document) -> str:
        return demo_ui.render_conversation(
            self._messages,
            self._replies,
            document,
            self._presentation.sender_label,
            self._presentation.reply_label,
            self._presentation.recognizer,
            self._presentation.reply_avatar,
            self._presentation.sender_avatar,
        )
