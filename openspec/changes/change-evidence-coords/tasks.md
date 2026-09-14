## 1. 契約變更

- [x] 1.1 於 `scam_guard/types.py` 確認 `Coord` 別名已存在
      （由 `add-document-type` 引入），否則先補上
- [x] 1.2 `CheckResult.evidence` 型別由 `list[int]` 改為 `list[Coord]`
- [x] 1.3 更新 `CheckResult` 的 docstring：說明座標為
      `(原始訊息序號, 訊息內句子序號)`、兩者皆從 0 起算、
      訊息序號不因截斷位移、句子序號非全域序號
- [x] 1.4 於 docstring 保留原本「存位置而非文字片段」的理由，
      並註明此次變更的原因是 `Document` 擴及全部訊息
- [x] 1.5 確認 `Verdict.evidence: list[str]` 未被改動

## 2. 既有程式碼盤點

- [x] 2.1 搜尋所有 `evidence` 的使用處，確認除 `types.py` 外無生產者
- [x] 2.2 確認 `pipeline.py` 的佔位 `CheckResult` 使用預設空陣列，無需修改
- [x] 2.3 更新 `tests/test_result_types.py` 中與 `evidence` 相關的斷言
- [x] 2.4 更新 `tests/test_check_protocol.py` 的假檢查 `evidence=[i]` 與斷言 `[[0], [1], [2]]`，改為座標形式

## 3. 測試

- [x] 3.1 以 `[(0, 3), (0, 4)]` 建構 `CheckResult`，欄位可讀取
- [x] 3.2 跨訊息證據 `[(0, 2), (4, 0)]` 可建構
- [x] 3.3 未提供 `evidence` 時為空陣列
- [x] 3.4 `hit=True` 且 `evidence` 為空（整體特徵訊號）為合法組合
- [x] 3.5 以 `Document.index_of()` 解析有效座標得到對應句子
- [x] 3.6 無效座標解析時拋例外，不回傳空值
- [x] 3.7 座標指向的正規化內容與檢查所見一致（往返測試）
- [x] 3.8 截斷情境：丟棄最舊 3 則後，證據座標的訊息序號仍為 3

## 4. 驗證

- [x] 4.1 `pytest` 全數通過
- [x] 4.2 `ruff check .` 通過
- [x] 4.3 確認無任何殘留的 `list[int]` 型別標註或舊格式測試資料
- [x] 4.4 commit 訊息使用 `refactor!:` 前綴，內文說明前提變更而非實作錯誤
