"""`LlmRuntime` 的 llama.cpp 實作。本檔在**頂端** import 推論引擎。

模型於**建構時**載入，不做延遲載入。三個理由：第一次請求載入會讓那一次多等
數十秒而使用者不知道原因；載入失敗會從「啟動失敗」這個大聲的狀態變成
「檢查安靜地回傳空陣列」；而「載了沒」的旗標在多執行緒的介面層是一個可以
race 的狀態，最壞情況下會同時載入兩份模型。
"""

import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama, LlamaGrammar, StoppingCriteriaList

from scam_guard.llm.check import TIMEOUT_SENTINEL

DEFAULT_N_CTX = 4096
"""prompt 視窗 4,000 字元約 2,989 token（實測）+ 指令段約 400 + 輸出上限 320。

**不設更大**：KV cache 隨 `n_ctx` 線性成長，而用不到的 context 是白付的記憶體。
"""

DEFAULT_N_THREADS = 2
"""目標環境是 2 vCPU。設更多會讓執行緒互相搶。"""

DEFAULT_SEED = 20260915
"""固定 seed。與 `temperature=0` 一起保證可重現 —— 逐層消融分析在隨機輸出上
無法歸因。代價是一則會失敗的訊息**每次都會失敗**，而那比「有時失敗」好偵錯。
"""

DEFAULT_MAX_TOKENS = 320
"""輸出 token 數上限，與 deadline 構成**雙重上界**。

`max_tokens` 保證最壞情況的 token 數有上界（也就是 decode 時間有上界）；
deadline 保證 prefill 之後的 wall-clock 有上界（涵蓋機器忙碌、其他請求搶 CPU）。
320 略大於 schema 的最大輸出（200 字元的 notes 加上四個鍵約 250 token），
使「生成長度用盡」不會在正常輸出上發生。
"""


class DeadlineStop:
    """逾時即中止生成。

    **以類別而非 closure 表達，因為 deadline 是狀態** —— 專案規範禁止巢狀
    `def` 與 closure，而回呼最自然的寫法正是 closure（捕捉 `deadline`）。

    `stopped` 記錄它有沒有真的中止過。少了它，逾時產生的截斷輸出會被記成
    `STRUCTURE`（合法前綴、`json.loads` 失敗）而不是 `TIMEOUT` ——
    兩者指向完全不同的修正動作，而三種失敗 MUST NOT 互相代替。
    """

    def __init__(self, deadline: float) -> None:
        self._deadline = deadline
        self.stopped = False

    def __call__(self, input_ids: Any, logits: Any) -> bool:
        if time.monotonic() >= self._deadline:
            self.stopped = True
        return self.stopped


class LlamaCppRuntime:
    """持有一個 `Llama` 實例，把一段 prompt 與一段 GBNF 變成一段輸出。

    ⚠️ **`add-gradio-chat` 的措辭潤飾層 MUST 使用同一個實例，MUST NOT 另行
    載入模型。** 不是記憶體問題（兩份 Q4 約 1.6 GB）—— 是**兩個 `Llama` 實例
    各自持有自己的 `n_threads`，在 2 vCPU 上會互相搶執行緒**，
    而兩者是序列呼叫的，第二份在第一份跑的時候完全閒置。
    潤飾呼叫 MUST NOT 帶 grammar（潤飾是自由文字，grammar 幫不上忙）。

    `verbose=False` 關掉 llama.cpp 的載入資訊 —— Space 的 stdout/stderr 進
    logs 分頁。⚠️ **部分輸出走 C 層的 stderr，Python 的設定管不到**：
    2026-09-15 實際觀察，`verbose=False` 之下仍會印出一行
    `llama_kv_cache_iswa: using full-size SWA cache (...)`。
    外洩的是效能與組態數字，**不含 prompt 內容**（已實際確認），
    但這一句是觀察不是保證，換一個 llama.cpp 版本要重看。
    """

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int = DEFAULT_N_THREADS,
        seed: int = DEFAULT_SEED,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._max_tokens = max_tokens
        self._grammars: dict[str, LlamaGrammar] = {}
        self._llama = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            seed=seed,
            logits_all=False,
            verbose=False,
        )

    def _grammar(self, grammar: str) -> LlamaGrammar:
        """快取解析結果 —— grammar 每次請求都相同，重複解析是白付的成本。"""
        if grammar not in self._grammars:
            self._grammars[grammar] = LlamaGrammar.from_string(grammar, verbose=False)
        return self._grammars[grammar]

    def generate(self, prompt: str, grammar: str, *, deadline_s: float) -> str:
        """一次呼叫，一段輸出。**不重試**（論證見 `scam_guard.llm.check`）。

        例外**逐一列舉**捕捉並轉成一個有名字的結果，使 `LlmCheck` 不需要寫
        `try`/`except` —— 與 `net/rdap.py` 對 `DomainAgeCheck` 做的事相同。
        不在清單上的例外照常向上傳播：那代表出現了我們沒想到的失敗，
        應該大聲壞掉。**MUST NOT `except Exception`。**

        - `TimeoutError` —— 引擎自己報的逾時，與 deadline 同義，記 `TIMEOUT`
        - `ValueError` —— grammar 解析失敗。回傳空字串，驗證層記 `STRUCTURE`
          （「我們的參數不對」，而 grammar 確實是我們的）
        - `MemoryError` —— `n_ctx` 或批次太大。同樣是我們的參數，同樣記 `STRUCTURE`
        """
        stop = DeadlineStop(time.monotonic() + deadline_s)
        try:
            completion = self._llama.create_completion(
                prompt,
                grammar=self._grammar(grammar),
                max_tokens=self._max_tokens,
                temperature=0.0,
                stopping_criteria=StoppingCriteriaList([stop]),
            )
        except TimeoutError:
            return TIMEOUT_SENTINEL
        except (ValueError, MemoryError):
            return ""
        if stop.stopped:
            return TIMEOUT_SENTINEL
        return completion["choices"][0]["text"]
