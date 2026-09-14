## 1. 前置確認

- [x] 1.1 確認 `add-normalize-text`、`add-split-sentences`、`add-document-type`、
      `add-context-limits`、`change-evidence-coords` 五者的任務皆已完成
- [x] 1.2 確認 `build_document(messages, limits)` 與 `DEFAULT_LIMITS` 可用

## 2. 移除型別佔位

- [x] 2.1 於 `scam_guard/check.py` 刪除 `Document: TypeAlias = Any` 與其佔位註解
- [x] 2.2 改為 `from scam_guard.normalize import Document`（直接 import，
      不用 `TYPE_CHECKING`；理由見 design）
- [x] 2.3 更新 `Check` Protocol 的 docstring，移除「尚未實作」的敘述
- [x] 2.4 確認 import 方向仍為 `check.py` → `normalize.py` → `types.py`，無循環

## 3. detect 接線

- [x] 3.1 於 `scam_guard/pipeline.py` import `Document`、`Limits`、
      `DEFAULT_LIMITS`、`build_document`
- [x] 3.2 `detect()` 簽章加入 `limits: Limits = DEFAULT_LIMITS`
- [x] 3.3 以 `doc = build_document(req.messages, limits)` 取代 `doc = None`
- [x] 3.4 刪除「`Document` 屬 `add-text-normalize`，此處傳 `None`」的佔位註解
- [x] 3.5 確認 `doc` 只建構一次，且原樣傳給 `_run()` 中的每個檢查
- [x] 3.6 更新 `detect()` 的 docstring：說明正規化在檢查之前執行且僅一次、
      `limits` 的用途、`Document` 座標與 `req.messages` 索引對齊

## 4. 測試

- [x] 4.1 檢查收到的第二個參數為 `Document` 而非 `None`
- [x] 4.2 註冊三個檢查，確認三者收到的是同一個 `Document` 實例
- [x] 4.3 兩個檢查回報同一座標時，解析結果為同一句子
- [x] 4.4 未傳 `limits` 時套用預設值
- [x] 4.5 傳入較小的 `limits` 時，檢查收到的 `Document` 已截斷且 `truncated` 為 True
- [x] 4.6 所有訊息皆為空白時，檢查仍全數被呼叫，`Verdict.checks` 完整
- [x] 4.7 空文件且無命中時 `scam_probability` 為 `None`
- [x] 4.8 檢查可由 `req.messages[m]` 取得座標對應訊息的 `sent_at`
- [x] 4.9 端到端：註冊一個以座標回報 `evidence` 的假檢查，
      經 `detect()` 後該座標可由 `Document` 解析出正確原文片段

## 5. 迴歸

- [x] 5.1 `tests/test_pipeline.py` 既有的短路測試**不需修改**即通過
      （若需修改，代表接線改變了短路行為，視為 bug）
- [x] 5.2 `tests/test_check_protocol.py` 全數通過
- [x] 5.3 `grep` 確認 repo 中不再有 `TypeAlias = Any` 的 `Document` 佔位
      與 `doc: Document = None`

## 6. 驗證

- [x] 6.1 `pytest` 全數通過
- [x] 6.2 `ruff check .` 通過，確認 `scam_guard/` 未 import 任何介面層模組
- [x] 6.3 `text-normalize` PR 的六個 change 全數完成，可端到端執行
