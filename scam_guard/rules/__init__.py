"""規則層 —— 文字訊號的產生者，系統中唯一「規則本身」的所在。

**import 方向是單向的**：`rules/ → normalize.py → types.py`，
另外向 `check.py` 取 `Stage`。本套件 MUST NOT import `scam_guard.pipeline` ——
`pipeline` 是規則的消費端（它執行規則、讀它們的 `CheckResult`），
反向 import 會形成循環，而且會讓「規則知道自己被怎麼跑」這件事成真。
需要跨層辨識規則時走**具名常數**（如 `RELATIONSHIP_RULE`），
由消費端 import 規則層，不由規則層 import 消費端。

同樣 MUST NOT import 任何介面層模組（gradio / fastapi / linebot / telegram）
或任何網路函式庫：規則層是純標準庫的文字比對，
`pyproject.toml` 的 `[tool.ruff.lint.flake8-tidy-imports.banned-api]` 擋住前者。
"""
