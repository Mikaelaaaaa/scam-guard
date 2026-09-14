## 1. Limits 型別

- [x] 1.1 於 `scam_guard/normalize.py` 定義 `Limits` frozen dataclass：
      `max_messages: int = 100`、`max_chars: int = 50_000`
- [x] 1.2 `__post_init__` 驗證兩者皆大於等於 1，不符拋 `ValueError` 並指出欄位與值
- [x] 1.3 定義模組常數 `DEFAULT_LIMITS = Limits()`
- [x] 1.4 docstring 記錄數字來源（Cofacts 長度分佈：中位數 83、p90 315、
      p99 1008、最長 2438）與「這是起點不是調過的值」

## 2. 丟棄邏輯

- [x] 2.1 `build_document(messages, limits=DEFAULT_LIMITS)` 接受上限參數
- [x] 2.2 先對每則訊息正規化並切句，記錄各則的正規化字元數與原始序號
- [x] 2.3 從最舊往新丟棄，直到同時滿足兩個上限
- [x] 2.4 最後一則永遠不丟；若其單則即超過字元上限，前文全部丟棄
- [x] 2.5 丟棄以整則為單位，不對任何訊息做部分截斷
- [x] 2.6 保留的訊息維持原始序號，座標不因丟棄而位移
- [x] 2.7 填入 `truncated` 與 `dropped_messages`，並以斷言維持
      `truncated == (dropped_messages > 0)`

## 3. 測試

- [x] 3.1 `tests/test_context_limits.py`：150 則短訊息 → 保留 100 則
- [x] 3.2 20 則長訊息超過字元上限 → 字元總數不超過上限
- [x] 3.3 兩者皆未超過 → 全部保留，`truncated` 為 False
- [x] 3.4 兩者皆超過 → 結果同時滿足兩個上限
- [x] 3.5 自訂則數上限 5、字元上限較小值，各自生效
- [x] 3.6 `Limits(max_messages=0)` 與負字元上限皆拋 `ValueError`
- [x] 3.7 10 則、上限 3 則 → 保留第 8、9、10 則且順序不變
- [x] 3.8 保留的每則句子與未套用上限時完全相同（不做部分截斷）
- [x] 3.9 最後一則單則超過字元上限 → 該則完整保留、其餘全丟
- [x] 3.10 則數上限 1 → 只保留最後一則
- [x] 3.11 訊息中插入大量零寬字元 → 不計入字元總數
- [x] 3.12 丟棄 37 則 → `truncated` 為 True、`dropped_messages` 為 37
- [x] 3.13 丟棄最舊 3 則後，保留的第一則座標訊息序號為 3
- [x] 3.14 性質測試：對隨機長度的訊息序列，結果恆滿足兩個上限且
      `truncated == (dropped_messages > 0)`

## 4. 驗證

- [x] 4.1 `pytest` 全數通過
- [x] 4.2 `ruff check .` 通過
- [x] 4.3 於 `add-confidence` 的 openspec 筆記中留下待辦：
      `truncated` 為 True 時須降低信心值，否則此痕跡無人消費
