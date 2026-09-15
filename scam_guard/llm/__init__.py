"""LLM 層的**核心側** —— 純函式、無外部依賴、不做 I/O。

本套件裡的每一個模組都只做字串與資料的轉換：組一段 prompt、產一段 grammar、
把一段模型輸出解析並驗證成 `CheckResult`。**它們都不知道模型怎麼跑。**

推論引擎的實作在**頂層**的 `llm_runtime/` 套件（與 `net/`、`pii_nlp/`、`api/`
同層），由組裝層注入。兩者不可混淆：

| 套件 | 內容 | 依賴 |
|---|---|---|
| `scam_guard/llm/`（本套件） | schema、prompt、驗證、`LlmCheck` | 只有標準庫 |
| `llm_runtime/`（頂層） | 推論引擎的實際呼叫、模型檔的取得 | `llm` extra 的兩個套件 |

本套件 MUST NOT import `llm_runtime`，也 MUST NOT import 任何推論引擎 ——
`scam_guard/` 的 import 圖裡沒有那個節點，所以未安裝 `llm` extra 時
核心的 import 不可能失敗，而系統中因此也 MUST NOT 出現任何 `try: import`。
"""
