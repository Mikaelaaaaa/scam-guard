"""HTTP 介面層。

`scam_guard/` 是偵測核心，不得 import `fastapi`（由 `pyproject.toml` 的
banned-api 強制）；本套件是那條界線的另一側，`per-file-ignores` 對
`"api/*"` 豁免 TID251。
"""
