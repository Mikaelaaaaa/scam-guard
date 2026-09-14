## 1. CheckResult

- [x] 1.1 於 `scam_guard/types.py` 定義 `CheckResult` dataclass，`frozen=True`
- [x] 1.2 欄位：`name: str`、`hit: bool`、`weight: float`、`detail: str`
- [x] 1.3 欄位：`evidence: list[int] = field(default_factory=list)`
- [x] 1.4 欄位：`scam_types: list[str] = field(default_factory=list)`
- [x] 1.5 欄位：`hard: bool = False`
- [x] 1.6 docstring 說明 `hard` 的判定標準（黑名單命中、Tier-A 規則為 True）
      與 `evidence` 為句子編號而非文字

## 2. Verdict

- [x] 2.1 定義 `Verdict` dataclass，`frozen=True`
- [x] 2.2 欄位：`scam_probability: float | None`、`confidence: float`
- [x] 2.3 欄位：`scam_type: str | None`
- [x] 2.4 欄位：`evidence: list[str]`（人類可讀的依據陳述）
- [x] 2.5 欄位：`actions: list[str]`（建議動作）
- [x] 2.6 欄位：`checks: list[CheckResult]`
- [x] 2.7 docstring 說明 `scam_probability` 為 `None` 代表信心不足，
      呼叫端應顯示「無法判定」而非數字

## 3. 測試

- [x] 3.1 `tests/test_result_types.py`：完整欄位的 `CheckResult` 可建構
- [x] 3.2 未提供 evidence 與 scam_types 時為空陣列，且兩個實例不共用同一個 list
- [x] 3.3 `Verdict` 的 `scam_probability` 可為 `None`
- [x] 3.4 `Verdict.checks` 保留未命中的檢查
- [x] 3.5 對 `CheckResult.hit` 重新賦值時拋例外

## 4. 驗證

- [x] 4.1 `pytest` 全數通過
- [x] 4.2 `ruff check .` 通過
