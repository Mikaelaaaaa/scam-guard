"""LLM 判讀在偵測核心這一側的全部內容：一個 Protocol 與一個 `Check`。

**核心不知道有 llama.cpp、不知道有 GGUF、不知道有量化。** 它只呼叫
`LlmRuntime.generate()`，而實作在頂層的 `llm_runtime/`，由組裝層注入 ——
與 `DomainAgeLookup` 是同一個形狀。

**可選掛載以「不註冊」表達。** 掛載就是建一個 runtime、建一個 counter、
建一個 `LlmCheck` 再 `register()`；不掛載就是**什麼都不做**。
本模組因此 MUST NOT 提供空實作的 runtime、MUST NOT 提供 `enabled: bool`、
MUST NOT 讓 `runtime` 有預設值 —— 三者都是語法合法、可以被貼進 production 設定、
而且看起來像是有人想過的值，而「沒有注入」是不可能被誤解的狀態。

未掛載時系統照常完整運作，降級為純規則版；而**純規則版是預設路徑，
不是失效狀態** —— `Verdict.checks` 裡沒有這一行，信心也不因此降低
（那是系統配置的事實，不是這則訊息的事實）。

**本模組不寫任何 `try`/`except`。** 「哪些檢查會失敗」是檢查自己的知識，
但推論引擎會拋什麼是 `llm_runtime/` 的知識 —— 由它捕捉可列舉的具體型別
並轉成一個有名字的結果，與 `net/rdap.py` 對 `DomainAgeCheck` 做的事相同。
"""

import secrets
from collections.abc import Callable, Sequence
from typing import Protocol

from scam_guard.check import Stage
from scam_guard.llm.prompt import PromptBudget, build_prompt
from scam_guard.llm.schema import build_grammar
from scam_guard.llm.validate import (
    SCAM_SIGNAL,
    LlmOutcome,
    LlmOutcomeCounter,
    parse_and_validate,
    to_check_results,
)
from scam_guard.normalize import Document
from scam_guard.types import CheckResult, Request
from scam_guard.weights import WeightTable

TIMEOUT_SENTINEL = "<timeout>"
"""runtime 回傳這個字串表示生成在期限前未完成。

`LlmRuntime.generate()` 的回傳型別是 `str`，所以「逾時」必須以一個值表達 ——
而它不能是例外，因為本模組不寫 `try`/`except`。

選這個字串的理由不是它好看，是它**不是合法 JSON**：即使有人把下面那個
`if` 拿掉，它也會走 `json.loads` 失敗那條路而被記成 `STRUCTURE`，
不可能被誤判成一次成功的判讀。fail-closed 在這裡有兩層。
"""


class LlmRuntime(Protocol):
    """核心與推論引擎之間的全部介面。

    `grammar` 是一段 GBNF 文字（`schema.build_grammar()` 的輸出），由 runtime
    交給解碼層。`deadline_s` 是期限。

    ⚠️ **`deadline_s` 涵蓋不了 prefill。** 逐 token 的停止條件要等第一個 token
    產生之後才開始被呼叫，而 prefill 發生在那之前。所以它保證的是
    「prefill 完成之後的 wall-clock 上界」，不是整次呼叫的上界。
    對 prefill 設限的唯一手段是 `PromptBudget.max_chars`，
    也就是那個預算同時是成本控制與逾時控制。**不假裝 deadline 是完整的上界。**
    """

    def generate(self, prompt: str, grammar: str, *, deadline_s: float) -> str: ...


class LlmCheck:
    """把一次 LLM 判讀變成一個與其他 34 個訊號同形的 `Check`。

    **它是一個 `Check` 而不是 pipeline 的特例**，兩個理由：消融實驗的逐層關閉
    靠 `registry.disable(name)`，不是 `Check` 就要另建一套開關；而短路邏輯
    已經為它設計好了（`Stage` 的 docstring 直接點名「`EXPENSIVE` —— RDAP、
    LLM 等外部呼叫」）。

    `name` 為 `llm_scam`。registry 的名稱必須是權重表中登錄的訊號名
    （`WeightTable.validate_against()` 在組裝階段驗證這件事），而本檢查依
    `label` 產出 `llm_scam` 或 `llm_suspicious` 兩種 `CheckResult.name` ——
    兩者是不同的東西：registry 的名稱是「怎麼指稱這個檢查」，
    `CheckResult.name` 是「這一筆訊號叫什麼」。已知的代價：短路或未命中時
    pipeline 補的佔位記錄一律掛在 `llm_scam` 名下，即使模型原本會說
    `llm_suspicious`。那是一個命名上的假象，不影響計分（計分只讀 `hit=True`）。

    **建構參數除 `notes_sink` 外全部必填且無預設值。** 一個預設為 `None` 的
    計數器是一個可以被忘記的東西，而忘記它的後果恰好是這一層要防的那件事。

    `notes_sink` 是唯一的可選者，且以**注入**表達而不是布林開關：
    未注入時 `analysis_notes` 被丟棄。它 MUST NOT 進入 `Verdict` 或
    `CheckResult.detail`（見 `schema.LlmOutput`）。
    """

    name = SCAM_SIGNAL
    stage = Stage.EXPENSIVE
    wants_prior = True

    def __init__(
        self,
        *,
        runtime: LlmRuntime,
        counter: LlmOutcomeCounter,
        table: WeightTable,
        budget: PromptBudget,
        deadline_s: float,
        notes_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._runtime = runtime
        self._counter = counter
        self._table = table
        self._budget = budget
        self._deadline_s = deadline_s
        self._notes_sink = notes_sink

    def __call__(
        self, req: Request, doc: Document, *, prior: Sequence[CheckResult]
    ) -> list[CheckResult]:
        """一次判讀：一次呼叫，一個結果，四種 outcome 之一。

        **不重試。** `temperature = 0` 使同一個 prompt 加同一個 grammar 產生
        同一個輸出，所以重試在數學上不會改變結果；要讓重試有價值就必須提高溫度，
        也就是**為了通過驗證而故意讓判定更隨機**，而那在一個判定系統裡說不通，
        並且會破壞逐層消融的可歸因性。
        「重試多次仍失敗」與「模型判為無訊號」的可分辨性由計數器保證，
        不由重試機制保證 —— 前者記 `STRUCTURE` 或 `SEMANTIC`，
        後者記 `OK` 且 `label` 為最低級。

        **空文件在組裝 prompt 之前先判斷，不以捕捉例外實作。** 空文件不是失敗，
        它是「沒有東西可以判讀」，而 `build_prompt()` 的例外代表呼叫端算錯了。
        先判斷比 try/except 乾淨，而且它讓那個例外保持原本的意思。
        """
        if not doc.coords:
            return []
        # 未登錄的訊號名讓 `KeyError` 傳播，不以略過或預設值處理 ——
        # 一個查不到群組的訊號悄悄消失，等於 prompt 少一段而沒有人會說。
        hit_groups = sorted({self._table.group_of(result.name) for result in prior if result.hit})
        nonce = secrets.token_hex(4)
        prompt = build_prompt(doc, hit_groups=hit_groups, budget=self._budget, nonce=nonce)
        raw = self._runtime.generate(prompt.text, build_grammar(), deadline_s=self._deadline_s)
        if raw == TIMEOUT_SENTINEL:
            outcome: LlmOutcome = LlmOutcome.TIMEOUT
            output = None
        else:
            outcome, output = parse_and_validate(raw, prompt.window)
        self._counter.record(outcome)
        if self._notes_sink is not None and output is not None:
            self._notes_sink(output.analysis_notes)
        return to_check_results(outcome, output)
