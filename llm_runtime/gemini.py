"""Gemini API 的 `LlmRuntime` 實作 —— 給伺服器端介面（LINE / API / 本機）用。

與 `llama_cpp_runtime.LlamaCppRuntime` 同一個 Protocol（`scam_guard.llm.check.LlmRuntime`），
所以偵測核心不知道判讀是本機 llama.cpp 還是雲端 Gemini。差別只在這一個檔案。

**為什麼在 `llm_runtime/` 而不在 `scam_guard/`：** 它做網路呼叫，而核心的 banned-api
禁 `urllib` / `requests` / `socket`。核心只產生 prompt 字串、驗證回傳字串，網路那一段
在這裡。

**隱私：** Gemini 是外部 API，送出去的是待判讀的訊息文字。伺服器端介面（LINE、API）
訊息本來就到伺服器了；瀏覽器版**不用**這個 runtime，走 transformers.js 留在裝置上。

**Gemini 沒有 GBNF。** `grammar` 參數收下但不使用 —— 改用 `responseMimeType:
application/json` 要求 JSON，格式由 prompt 內文約束，回傳再交給 `parse_and_validate`。
輸出仍可能不合格（座標指到視窗外等），那時 `LlmCheck` 走既有降級，與其他 runtime 一致。
"""

import json
import os
import time
import urllib.error
import urllib.request

from scam_guard.llm.schema import FIELD_NAMES, LABELS
from scam_guard.types import ScamType

MAX_ATTEMPTS = 2
"""一次 generate 最多打幾次。免費層常回 429，重試一次多半就過；再多會拖過 reply_token。"""

RETRY_BACKOFF_S = 1.5
"""兩次嘗試之間的等待。"""

_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
"""可重試的 HTTP 狀態：額度（429）與伺服器忙碌（5xx）。其餘（400/401/404）是設定錯誤，
不重試、直接 raise。"""

DEFAULT_MODEL = "gemini-3.6-flash"
"""預設模型。`gemini-2.5-flash` 已對新用戶關閉，改用 3.6。可由 `GEMINI_MODEL` 覆寫。"""

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
"""`generateContent` 端點。API key 以 query string `?key=` 帶（Gemini 的 API key 認證方式，
不是 Bearer；實測 `AQ.` 開頭的 key 走這條）。"""


def _response_schema() -> dict[str, object]:
    """Gemini structured output 的 responseSchema，強制輸出四個欄位的正確型別與順序。

    **這個 schema 修的是一個具體的 bug：** 不帶 schema 時 Gemini 把座標回成字串
    `"[0,0]"`，而 `parse_and_validate` 要巢狀整數陣列 `[[0,0]]` —— 於是每次都是
    `STRUCTURE` 失敗。`evidence_sentence_ids` 宣告成 array-of-array-of-integer 就消掉它。

    欄位順序沿用 `FIELD_NAMES`（`label` 在最後，理由見 `schema.py`：結論以說明與證據
    為條件）。`category_165` 帶 `ScamType` 的 18 個值當 enum 引導 Gemini 選合法值，
    `nullable` 讓「無詐騙話術」時可為 null。值域是否合法仍由 `parse_and_validate` 把關，
    schema 只是把 Gemini 往正確格式推。
    """
    return {
        "type": "OBJECT",
        "properties": {
            "analysis_notes": {"type": "STRING"},
            "evidence_sentence_ids": {
                "type": "ARRAY",
                "items": {"type": "ARRAY", "items": {"type": "INTEGER"}},
            },
            "category_165": {
                "type": "STRING",
                "nullable": True,
                "enum": [scam_type.value for scam_type in ScamType],
            },
            "label": {"type": "STRING", "enum": list(LABELS)},
        },
        "propertyOrdering": list(FIELD_NAMES),
        "required": list(FIELD_NAMES),
    }


class GeminiUnavailable(RuntimeError):
    """`GEMINI_API_KEY` 未設時建構即拋 —— 呼叫端據此決定不掛這個 runtime，
    而不是掛了之後每次呼叫才失敗。與 llama.cpp 缺 GGUF 的處置對稱。"""


class GeminiRuntime:
    """呼叫 Gemini `generateContent`，回傳模型產生的原始文字。

    `api_key` 與 `model` 預設從環境變數讀（`GEMINI_API_KEY` / `GEMINI_MODEL`），
    也可顯式傳入（測試用）。key 缺席時建構就拋 `GeminiUnavailable`。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise GeminiUnavailable(
                "GEMINI_API_KEY 未設。Gemini runtime 需要一把 Gemini API key；"
                "請設定環境變數，或改用其他 runtime。"
            )
        self._api_key = resolved_key
        self._model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self._timeout_s = timeout_s

    def generate(self, prompt: str, grammar: str, *, deadline_s: float) -> str:
        """呼叫 Gemini，回傳模型輸出的文字（預期是一段 JSON）。

        `grammar` 收下不用（Gemini 無 GBNF）。`deadline_s` 併入 HTTP timeout ——
        Gemini 沒有 llama.cpp 那種逐 token 停止條件，所以期限只能是整次呼叫的上界，
        取 `deadline_s` 與建構時 `timeout_s` 的較小者。

        **失敗就 raise `GeminiCallFailed`，不優雅降級。** 使用者的決定：LLM 是必備層，
        它失敗就是整個判定失敗，前端要顯示「失敗」，不給只有三層的降級結果。免費層常回
        429，先重試 `MAX_ATTEMPTS` 次（transient）；用盡仍失敗、或設定錯誤（401/404）就
        raise，由介面層接住並寫「語意判讀失敗」。
        """
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "responseSchema": _response_schema(),
                    "maxOutputTokens": 512,
                },
            }
        ).encode("utf-8")
        url = ENDPOINT.format(model=self._model) + "?key=" + self._api_key
        timeout = min(self._timeout_s, deadline_s) if deadline_s > 0 else self._timeout_s
        reason = ""
        for attempt in range(MAX_ATTEMPTS):
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return _extract_text(payload)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:150]
                if error.code not in _TRANSIENT_STATUS:
                    raise GeminiCallFailed(
                        f"Gemini 回 HTTP {error.code}（設定錯誤，不重試）：{detail}"
                    ) from error
                reason = f"HTTP {error.code}：{detail}"
            except urllib.error.URLError as error:
                reason = f"連線失敗：{error.reason}"
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S)
        raise GeminiCallFailed(f"Gemini 呼叫失敗（已重試 {MAX_ATTEMPTS} 次）：{reason}")


class GeminiCallFailed(RuntimeError):
    """一次呼叫失敗（HTTP 4xx/5xx 或連線問題）。`LlmCheck` 接到後走既有降級 ——
    這一層沒貢獻，不編造判定。與 llama.cpp 逾時的處置對稱。"""


def _extract_text(payload: dict[str, object]) -> str:
    """從 Gemini 回應取出模型產生的文字。

    形狀是 `candidates[0].content.parts[0].text`。任一層缺席就是一次沒有可用輸出的
    回應（安全過濾、空回答等），拋 `GeminiCallFailed` 讓上層降級 —— 不回空字串冒充
    成功，因為空字串會被 `parse_and_validate` 當成 `STRUCTURE` 失敗、把「API 沒給東西」
    誤記成「模型格式錯」。
    """
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise GeminiCallFailed(f"Gemini 回應沒有 candidates：{json.dumps(payload)[:200]}")
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise GeminiCallFailed(f"Gemini 回應沒有 parts：{json.dumps(payload)[:200]}")
    text = parts[0].get("text") if isinstance(parts[0], dict) else None
    if not isinstance(text, str) or not text:
        raise GeminiCallFailed("Gemini 回應的 text 為空")
    return text
