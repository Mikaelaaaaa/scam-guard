## 1. 座標型別

- [x] 1.1 於 `scam_guard/types.py` 定義 `Coord: TypeAlias = tuple[int, int]`
- [x] 1.2 docstring 明示語意為 `(原始訊息序號, 訊息內句子序號)`，兩者皆從 0 起算，
      並說明為何第二個分量不是全域序號
- [x] 1.3 確認 `types.py` 仍不 import 專案內任何模組

## 2. Document 型別

- [x] 2.1 於 `scam_guard/normalize.py` 定義 `Document` frozen dataclass
- [x] 2.2 欄位：`sentences`、`raw_sentences`、`coords`、`truncated=False`、
      `dropped_messages=0`
- [x] 2.3 三個序列以 `tuple` 儲存，防止檢查就地修改
- [x] 2.4 `__post_init__` 驗證三者等長，不符拋 `ValueError` 並指出長度
- [x] 2.5 `__post_init__` 驗證座標遞增（訊息序號不減；同訊息內句子序號連續自 0 起）
- [x] 2.6 允許空 `Document`，不得因空陣列拋例外

## 3. 座標查詢方法

- [x] 3.1 建構時預先建立 `coord -> 扁平索引` 的對照 dict
- [x] 3.2 `index_of(coord) -> int`，無效座標拋 `KeyError` 並在訊息中印出該座標
- [x] 3.3 `text_at(coord) -> str`
- [x] 3.4 `raw_at(coord) -> str`
- [x] 3.5 `message_range(m) -> range`，無句子的訊息回傳空 range
- [x] 3.6 docstring 寫明：呈現用 `raw_at` / `raw_sentences`，
      規則與 LLM 用 `text_at` / `sentences`

## 4. 組裝函式

- [x] 4.1 `build_document(messages) -> Document`：對每則訊息呼叫 `normalize_text`
      與 `split_sentences`，扁平累積並產生座標
- [x] 4.2 訊息序號採用輸入序列中的位置
- [x] 4.3 不產生句子的訊息略過，但不影響後續訊息的序號
- [x] 4.4 此階段不套用任何上限，`truncated` 恆為 False（上限屬 `add-context-limits`）

## 5. 測試

- [x] 5.1 `tests/test_document.py`：三則訊息各兩句 → 六個句子
- [x] 5.2 兩則各兩句時座標為 (0,0)、(0,1)、(1,0)、(1,1)
- [x] 5.3 正規化句子與原文片段等長，且原文片段保留零寬字元
- [x] 5.4 全形數字：正規化句子為半形、原文片段為全形
- [x] 5.5 `index_of` / `text_at` / `raw_at` 對有效座標回傳正確結果
- [x] 5.6 無效座標拋 `KeyError`
- [x] 5.7 `message_range` 回傳正確範圍；無句子的訊息回傳空 range
- [x] 5.8 未指定截斷資訊時 `truncated` 為 False、`dropped_messages` 為 0
- [x] 5.9 欄位重新賦值拋例外；序列追加元素拋例外
- [x] 5.10 長度不一致、座標數不符、座標未遞增皆拋 `ValueError`
- [x] 5.11 全部訊息皆無文字時建構成功且三個序列皆空
- [x] 5.12 座標可回指 `req.messages` 的正確元素（含 `sent_at` 取用）

## 6. 驗證

- [x] 6.1 `pytest` 全數通過
- [x] 6.2 `ruff check .` 通過
- [x] 6.3 確認 `normalize.py` 僅 import 標準庫與 `scam_guard.types`
