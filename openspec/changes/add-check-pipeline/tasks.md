## 1. 階段標記

- [x] 1.1 於 `scam_guard/check.py` 定義 `Stage` 列舉：`LOCAL`、`EXPENSIVE`
- [x] 1.2 `Check` Protocol 加入 `stage: Stage` 屬性
- [x] 1.3 `CheckRegistry.register()` 驗證 `stage` 存在且為合法值

## 2. detect 主流程

- [x] 2.1 建立 `scam_guard/pipeline.py`
- [x] 2.2 `detect(req, registry, *, short_circuit=True) -> Verdict`
- [x] 2.3 先執行所有 `LOCAL` 檢查，收集結果
- [x] 2.4 判斷是否短路：有 `hit and hard` 且無 `quotation` 命中
- [x] 2.5 未短路時執行 `EXPENSIVE` 檢查
- [x] 2.6 為未回傳結果的檢查補 `hit=False` 記錄
- [x] 2.7 為被跳過的檢查補記錄，`detail` 需與「未命中」可區分
- [x] 2.8 組裝 `Verdict`，`scam_probability=None`，
      docstring 註明計分屬 `scoring` PR

## 3. 測試

- [x] 3.1 `tests/test_pipeline.py`：空 registry 不拋例外
- [x] 3.2 未命中的檢查在 `checks` 中有記錄
- [x] 3.3 硬證據命中時昂貴檢查被跳過，且跳過記錄可辨識
- [x] 3.4 本機檢查不受短路影響，全部執行
- [x] 3.5 弱訊號命中不觸發短路
- [x] 3.6 引述偵測命中時，即使有硬證據仍執行昂貴檢查
- [x] 3.7 `short_circuit=False` 時全部執行
- [x] 3.8 `scam_probability` 為 `None`
- [x] 3.9 端到端：註冊兩個假檢查（一本機一昂貴），確認流程跑通

## 4. 驗證

- [x] 4.1 `pytest` 全數通過
- [x] 4.2 `ruff check .` 通過，確認 `pipeline.py` 未 import 介面層模組
- [x] 4.3 `detector-core` PR 的五個 change 全數完成，可端到端執行
