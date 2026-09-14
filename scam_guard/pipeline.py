"""檢查執行流程與短路規則 —— 系統的單一入口。"""

from scam_guard.check import Check, CheckRegistry, Document, Stage
from scam_guard.types import CheckResult, Request, Verdict

QUOTATION_CHECK = "quotation"
"""引述偵測的檢查名稱。命中時否決短路 —— 見 `detect()` 的說明。"""

NOT_HIT = "未命中"
SKIPPED = "因短路未執行"


def _run(check: Check, req: Request, doc: Document) -> list[CheckResult]:
    """執行單一檢查，未回傳結果時補上 `hit=False` 記錄。

    檢查未命中時回傳空陣列（見 `add-check-protocol`），但 `Verdict.checks`
    須保留全部執行記錄，因此由 pipeline 補 —— pipeline 知道執行過哪些檢查。
    """
    results = check(req, doc)
    if results:
        return results
    return [CheckResult(name=check.name, hit=False, weight=0.0, detail=NOT_HIT)]


def _skipped(check: Check) -> CheckResult:
    """為因短路未執行的檢查補記錄，`detail` 與「執行了但未命中」可區分。

    消融實驗需要這個區別 —— 「跑了沒訊號」與「根本沒跑」是兩件事。
    """
    return CheckResult(name=check.name, hit=False, weight=0.0, detail=SKIPPED)


def detect(req: Request, registry: CheckRegistry, *, short_circuit: bool = True) -> Verdict:
    """系統唯一的偵測入口。依序執行已啟用的檢查，回傳 `Verdict`。

    registry 由呼叫端傳入而非用模組層級的全域 —— 全域讓測試要操作全域狀態，
    且無法同時跑兩組不同設定，而消融實驗正需要這個。

    執行順序為先全部 `LOCAL`、再視短路結果決定是否執行 `EXPENSIVE`。
    短路的三條規則：

    1. 只有**硬證據命中**（`hit and hard`）才短路。弱訊號（Tier-B）命中
       不足以短路 —— 那正是需要 LLM 釐清的情況。
    2. **引述偵測命中時強制不短路**，即使同時存在硬證據。防詐宣導文含有比
       真詐騙更多的詐騙關鍵字（「本行絕不會要求您至 ATM 操作」一句同時命中
       三條 Tier-A 規則），照一般短路邏輯會直接誤判；此類訊息正需要 LLM
       判斷它是宣導還是實施。以顯式的否決旗標表達而非以執行順序表達 ——
       讓順序承載語意的話，順序一改就出 bug。
    3. `short_circuit=False` 時全部執行，供消融實驗分析「LLM 在黑名單
       已命中的情況下會說什麼」這類問題。

    ⚠️ 此階段**不做計分**：`scam_probability` 固定為 `None`、`confidence`
    為 0.0、`scam_type` / `evidence` / `actions` 為空，皆為佔位值。
    實際計算屬 `scoring` PR（`add-score-compute`、`add-confidence`、
    `add-type-resolve`、`add-verdict-render`）。在那之前接上 API 只會得到
    「無法判定」，不會得到無意義的數字。
    """
    # `Document` 屬 `add-text-normalize`，正規化尚未實作，此處傳 `None`。
    # 該 change 落地後改為在此呼叫 normalize(req)。
    doc: Document = None

    checks = registry.enabled()
    local = [c for c in checks if c.stage is Stage.LOCAL]
    expensive = [c for c in checks if c.stage is Stage.EXPENSIVE]

    results: list[CheckResult] = []
    for check in local:
        results.extend(_run(check, req, doc))

    if short_circuit and _should_short_circuit(results):
        results.extend(_skipped(check) for check in expensive)
    else:
        for check in expensive:
            results.extend(_run(check, req, doc))

    return Verdict(
        scam_probability=None,
        confidence=0.0,
        scam_type=None,
        evidence=[],
        actions=[],
        checks=results,
    )


def _should_short_circuit(results: list[CheckResult]) -> bool:
    """硬證據命中即短路，但引述偵測命中時否決之。"""
    if any(r.name == QUOTATION_CHECK and r.hit for r in results):
        return False
    return any(r.hit and r.hard for r in results)
