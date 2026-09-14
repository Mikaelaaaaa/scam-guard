## 1. 模組骨架與型別

- [x] 1.1 建立 `scam_guard/normalize.py`，docstring 註明此模組不 import 專案內任何模組
- [x] 1.2 定義 `NormalizedText` frozen dataclass：`raw`、`text`、`offsets`
- [x] 1.3 `__post_init__` 驗證 `len(offsets) == len(text) + 1`、單調不減、
      首元素為 0（`text` 非空時）、末元素為 `len(raw)`，不符拋 `ValueError`
- [x] 1.4 `NormalizedText.raw_span(a, b) -> str` 回傳 `raw[offsets[a]:offsets[b]]`

## 2. 字元映射表

- [x] 2.1 定義 `INVISIBLE` 常數：零寬（U+200B–U+200D、U+2060、U+FEFF）、
      軟連字號（U+00AD）、雙向控制（U+200E、U+200F、U+202A–U+202E、U+2066–U+2069）
- [x] 2.2 定義 `WHITESPACE` 常數：需統一為半形空格的空白字元，排除換行
- [x] 2.3 定義 `PUNCT` 常數：NFKC 未涵蓋的標點等價寫法對映表
- [x] 2.4 以註解記錄「ASCII 句點不映到句號、頓號不映到逗號」及其理由

## 3. normalize_text 實作

- [x] 3.1 `normalize_text(raw: str) -> NormalizedText`
- [x] 3.2 逐字元處理，同時累積輸出字元與其來源索引
- [x] 3.3 每個字元依序套用：不可見字元判定（刪除）→ NFKC → 空白統一 → 標點對映
- [x] 3.4 換行統一：`\r\n` 與 `\r` 收斂為 `\n`
- [x] 3.5 尾端補哨兵 `len(raw)`，建構 `NormalizedText`

## 4. 測試

- [x] 4.1 `tests/test_normalize.py`：全形英數與全形標點轉換
- [x] 4.2 中文句號與頓號不被轉成 ASCII
- [x] 4.3 零寬字元與雙向覆寫字元被移除
- [x] 4.4 全為不可見字元的輸入產出空字串，不拋例外
- [x] 4.5 網址中的 ASCII 句點與小數點原樣保留
- [x] 4.6 頓號與逗號正規化後仍可分辨
- [x] 4.7 全形空格轉半形；連續空白不被壓縮；換行保留
- [x] 4.8 區間映回原文正確，且被移除的零寬字元出現在映回的片段中
- [x] 4.9 全文區間映回等於原文（對多組輸入）
- [x] 4.10 簡體字與繁簡混用原樣保留
- [x] 4.11 全大寫英文保留
- [x] 4.12 冪等性：`normalize_text(normalize_text(x).text).text == normalize_text(x).text`
- [x] 4.13 性質測試：對隨機字串驗證 `offsets` 單調不減、長度正確、
      全文映回等於原文
- [x] 4.14 `NormalizedText` 不變式違反時拋 `ValueError`

## 5. 驗證

- [x] 5.1 `pytest tests/test_normalize.py` 全數通過
- [x] 5.2 `ruff check .` 通過
- [x] 5.3 確認 `normalize.py` 的 import 僅有標準庫
