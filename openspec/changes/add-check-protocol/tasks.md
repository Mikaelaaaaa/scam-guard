## 1. Check 協定

- [x] 1.1 建立 `scam_guard/check.py`
- [x] 1.2 定義 `Check` Protocol：`name: str` 屬性與
      `__call__(self, req: Request, doc: Document) -> list[CheckResult]`
- [x] 1.3 docstring 說明未命中須回傳空陣列、外部失敗須自行吞例外並記錄
- [x] 1.4 `Document` 此時尚未實作（屬 `add-text-normalize`），
      先以 `typing.TYPE_CHECKING` 或暫時型別別名處理，不可為此引入循環依賴

## 2. CheckRegistry

- [x] 2.1 定義 `CheckRegistry` 類別，內部以 dict 保存 name → check
- [x] 2.2 `register(check)`：驗證具 `name` 屬性且 callable，否則拋例外
- [x] 2.3 `register()` 對重複名稱的行為需明確定義（覆蓋或拋例外，擇一並記錄於 docstring）
- [x] 2.4 `disable(name)` 與 `enable(name)`
- [x] 2.5 `enabled()` 回傳已啟用的檢查，順序穩定（依註冊順序）
- [x] 2.6 停用不存在的名稱時的行為需明確定義

## 3. 測試

- [x] 3.1 `tests/test_check_protocol.py`：函式形式的檢查可註冊並列舉
- [x] 3.2 類別形式的檢查可註冊
- [x] 3.3 兩個 registry 實例互相隔離
- [x] 3.4 註冊缺少 `name` 的物件時拋例外
- [x] 3.5 註冊不可呼叫的物件時拋例外
- [x] 3.6 停用後不被列舉，重新啟用後又出現
- [x] 3.7 停用不存在的名稱，行為符合 docstring 所述
- [x] 3.8 一個檢查回傳三筆結果時，陣列長度為 3

## 4. 驗證

- [x] 4.1 `pytest` 全數通過
- [x] 4.2 `ruff check .` 通過
