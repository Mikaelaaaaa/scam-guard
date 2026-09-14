"""檢查執行流程與短路規則 —— 系統的單一入口。"""

from scam_guard.check import Check, CheckRegistry, Stage
from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document
from scam_guard.redact import redact_document
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
    return [CheckResult(name=check.name, hit=False, detail=NOT_HIT)]


def _skipped(check: Check) -> CheckResult:
    """為因短路未執行的檢查補記錄，`detail` 與「執行了但未命中」可區分。

    消融實驗需要這個區別 —— 「跑了沒訊號」與「根本沒跑」是兩件事。
    """
    return CheckResult(name=check.name, hit=False, detail=SKIPPED)


def detect(
    req: Request,
    registry: CheckRegistry,
    *,
    short_circuit: bool = True,
    limits: Limits = DEFAULT_LIMITS,
) -> Verdict:
    """系統唯一的偵測入口。依序執行已啟用的檢查，回傳 `Verdict`。

    **正規化在任何檢查之前執行，且一次呼叫只執行一次**，所有檢查共用同一個
    `Document` 實例。由 `detect()` 而非各檢查自行呼叫 `build_document()`：
    成本不必乘上 N 是次要理由，致命的理由是座標會不一致 —— 若某個檢查用不同的
    `limits` 正規化，它的訊息序號與丟棄則數就與其他檢查不同，而 `evidence`
    座標是跨檢查共用的語言。座標系必須有唯一的產生者。

    `Document` 的座標與 `req.messages` 的索引對齊：座標為 `(m, s)` 時
    `req.messages[m]` 就是該句所屬的訊息，需要 `sent_at` 的軌跡檢查因此
    不需要另一張對照表。

    registry 與 `limits` 皆由呼叫端傳入而非用模組層級的全域 —— 全域讓測試要
    操作全域狀態，且無法同時跑兩組不同設定，而消融實驗正需要這個
    （`add-ablation` 要掃「前文長度對準確率的影響」，掃的就是 `limits`）。

    正規化結果為空（貼圖、純圖片、只有空白的訊息）**不中斷流程**：檢查照常
    全部執行、`Verdict.checks` 照常有完整記錄、輸出仍為「無法判定」。
    提前回傳會讓 `Verdict.checks` 變空，而消融實驗依賴「每個檢查都有記錄」
    這個不變式。

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

    **可記錄投影（`Verdict.redacted`）於全部檢查之後產出，一次呼叫只產出一次。**
    這個順序不是效能考量，是「規則層與 URL 層讀未遮蔽文字」的結構性保證：
    遮蔽結果在最後一個 `Check` 回傳之前**根本不存在**，檢查在時間上不可能讀到它，
    也不可能把它當成輸入。短路時也照常產出 —— 少跑幾個檢查不影響投影的完整性。

    ⚠️ 此階段**不做計分**：`scam_probability` 固定為 `None`、`confidence`
    為 0.0、`scam_type` / `evidence` / `actions` 為空，皆為佔位值。
    實際計算屬 `scoring` PR（`add-score-compute`、`add-confidence`、
    `add-type-resolve`、`add-verdict-render`）。在那之前接上 API 只會得到
    「無法判定」，不會得到無意義的數字。
    """
    doc: Document = build_document(req.messages, limits)

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
        redacted=redact_document(doc),
    )


def _should_short_circuit(results: list[CheckResult]) -> bool:
    """硬證據命中即短路，但引述偵測命中時否決之。"""
    if any(r.name == QUOTATION_CHECK and r.hit for r in results):
        return False
    return any(r.hit and r.hard for r in results)
