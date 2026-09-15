"""推論引擎的實作 —— 偵測核心之外，與 `net/`、`tools/`、`pii_nlp/`、`api/` 同層。

**為什麼是一個新的頂層套件而不是放進既有的三個：**

- **`net/`** 的定義是「在**每次請求中**執行的網路 I/O」。推論不做網路 I/O；
  模型檔的下載做，但那發生在啟動時，不在請求路徑上。放進去會讓
  「`net/` 是請求路徑上的對外連線」這個有用的說法變模糊。
- **`tools/`** 由人在部署前執行一次，產物是一個檔案。推論在請求路徑上。
- **`pii_nlp/`** 是選配的 NLP 個資辨識，與 LLM 無關。
- **`app.py`** 是介面層。潤飾層與未來的 `api/` 都要用同一個 runtime 實例，
  放進 `app.py` 會讓 `api/` 需要 import 它，方向反了。

**為什麼叫 `llm_runtime/` 而不是 `llm/`：** 核心側已經有 `scam_guard/llm/`，
兩個同名的套件在 import 敘述裡看起來一樣（`from llm import ...` 與
`from scam_guard.llm import ...`），而那是一個純粹由命名造成的困惑。

**沒有 `try: import`。** `scam_guard/` 的 import 圖裡沒有本套件，所以未安裝
`llm` extra 時核心的 import 不可能失敗；而 `import llm_runtime.llama_cpp_runtime`
會在**模組頂端**拋 `ModuleNotFoundError`，大聲、在模組邊界、指名缺哪個套件。

## 安裝

```
pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \\
    -e ".[dev,llm]"
```

⚠️ PyPI 上的 `llama-cpp-python` 0.3.35 **只有 sdist**（74.9 MB），少了
`--extra-index-url` 會觸發數分鐘的 C++ 編譯。那個 index 是作者以 GitHub Pages
託管的，不是 PyPI —— 它掛掉或停更時安裝會落回編譯。
**2026-09-15 實測：`manylinux2014_x86_64` 的 wheel 正常（zip 驗證通過），
但 `macosx_11_0_arm64` 的 0.3.35 wheel 是壞的**（下載完整 18.2 MB，
`lib/libggml-base.0.20.0.dylib` 的 CRC-32 不符，`pip install` 失敗，重試兩次相同），
本機因此走 sdist 自行編譯。這正是上面那句「第三方託管」風險的實例。

## 組裝

```python
from llm_runtime.llama_cpp_runtime import LlamaCppRuntime
from llm_runtime.model import ensure_model
from scam_guard.llm.check import LlmCheck
from scam_guard.llm.prompt import DEFAULT_BUDGET
from scam_guard.llm.validate import LlmOutcomeCounter

runtime = LlamaCppRuntime(ensure_model())
counter = LlmOutcomeCounter()
registry.register(
    LlmCheck(
        runtime=runtime,
        counter=counter,
        table=table,
        budget=DEFAULT_BUDGET,
        deadline_s=45.0,
    )
)
```

**「不掛載」的作法是什麼都不做** —— 不建 runtime、不建 `LlmCheck`、不 `register()`。
此時 `Verdict.checks` 中沒有這一行（pipeline 只為註冊過的檢查補記錄），
信心也不因此降低，系統降級為純規則版。
**純規則版是預設路徑，不是失效狀態。**
"""
