"""評估側 —— 測試集的建立、指標的計算與消融實驗。

**import 方向是單向的：`tools/eval/` → `tools/` → `scam_guard/`，反向皆違規。**
偵測核心不知道評估存在，也不知道測試集存在；一旦反向依賴出現，`scam_guard/`
的「無 I/O、無外部依賴」就守不住，而那條界線是本專案四份 spec 的共同前提。

本套件的任何模組 MUST NOT 被 `scam_guard/` import，`scam_guard/` 的模組
MUST NOT 在此被修改 —— 評估只讀不寫核心。
"""
