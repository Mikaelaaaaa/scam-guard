"""資料取得（網路 I/O）—— 與 `api/`、`adapters/` 同級的介面層目錄。

分工只有一條，但它是硬的：

- `scam_guard/` 做**資料查詢**：純函式、讀本機檔案，不知道有網路。
- `tools/` 做**資料取得**：對外發出網路請求，把結果落地成本機檔案。

方向是單向的 —— `tools/` 可以 import `scam_guard/`（它是消費端），
`scam_guard/` import `tools/` 則是違規。後者 lint 擋不到（同屬專案內模組），
只能靠 review，與 `openspec/project.md` 既有的說法一致。

前者由 `pyproject.toml` 的 ruff `flake8-tidy-imports.banned-api` 強制：
`urllib.request` / `requests` / `httpx` 一律禁止，`tools/*` 豁免。
那份清單擋的是最順手的三個入口，擋不住 `socket`、`http.client` 或
`subprocess` 叫 curl —— 它是提醒機制不是沙箱，繞過它需要刻意。
"""
