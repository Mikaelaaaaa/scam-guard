## 1. 型別定義

- [x] 1.1 建立 `scam_guard/types.py`
- [x] 1.2 定義 `Message` dataclass，`frozen=True`，欄位 `text: str`、
      `sender: str | None = None`、`sent_at: datetime | None = None`
- [x] 1.3 定義 `Request` dataclass，`frozen=True`，欄位 `messages: list[Message]`
- [x] 1.4 `Request.__post_init__` 於空陣列時拋 `ValueError`，訊息需可讀
- [x] 1.5 `Request.latest` property 回傳 `messages[-1]`
- [x] 1.6 `Request.context` property 回傳 `messages[:-1]`
- [x] 1.7 `Request.from_text()` classmethod

## 2. 測試

- [x] 2.1 `tests/test_types.py`：僅含 text 的 `Message` 可建構，另兩欄為 None
- [x] 2.2 三則訊息時 `latest` 為第三則、`context` 為前兩則且順序正確
- [x] 2.3 單則訊息時 `context` 為空陣列
- [x] 2.4 空陣列建構拋 `ValueError`
- [x] 2.5 `from_text()` 產出含一則訊息的 Request
- [x] 2.6 對 `Message.text` 重新賦值時拋例外

## 3. 驗證

- [x] 3.1 `pytest` 全數通過
- [x] 3.2 `ruff check .` 通過 —— 確認 `types.py` 未 import 任何介面層模組
