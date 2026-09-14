"""類型判定：候選、兩條覆寫規則、三層排序、衝突與規則優先。"""

import ast
import copy
import random
from pathlib import Path

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document, build_document
from scam_guard.pipeline import detect
from scam_guard.type_resolve import REFINEMENTS, is_conflict, resolve_type
from scam_guard.types import CheckResult, Coord, Message, Request, ScamType, Verdict
from scam_guard.weights import Signal, SignalWeight, WeightTable, load_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE = load_weights()

LLM_SIGNALS = frozenset({"llm_category"})
"""測試用的來源標記。本模組不知道任何 LLM 檢查的名稱，標記由組裝層提供。"""

LLM_SIGNAL = Signal(
    name="llm_category",
    group="llm_category",
    hard_capable=False,
    weight_soft=SignalWeight(
        value=0.6,
        basis="placeholder",
        inherited_from="測試：llm-layer 落地時 LLM 訊號會登錄進表",
        blocked_on="llm-layer",
    ),
)
TABLE_WITH_LLM = WeightTable(
    path=TABLE.path,
    signals={**TABLE.signals, LLM_SIGNAL.name: LLM_SIGNAL},
    roles=TABLE.roles,
    thresholds=TABLE.thresholds,
    type_priority=TABLE.type_priority,
)
"""LLM 掛載之後的表。單調升級的條件是 LLM 訊號的權重非負 —— 這裡是 0.6。"""


def hit(
    name: str,
    *types: ScamType,
    hard: bool = False,
    evidence: tuple[Coord, ...] = (),
) -> CheckResult:
    return CheckResult(
        name=name,
        hit=True,
        detail="命中",
        evidence=list(evidence),
        scam_types=list(types),
        hard=hard,
    )


def miss(name: str, *types: ScamType) -> CheckResult:
    return CheckResult(name=name, hit=False, detail="未命中", scam_types=list(types))


# --- 候選集合 -----------------------------------------------------------


def test_unhit_result_produces_no_candidate() -> None:
    resolution = resolve_type([miss("safe_account", ScamType.FAKE_AUTHORITY)], TABLE)

    assert resolution.scam_type is None


def test_same_type_from_two_results_keeps_the_stronger() -> None:
    results = [
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True, evidence=((0, 0),)),
        hit("identity_reset", ScamType.FAKE_AUTHORITY, evidence=((0, 1),)),
    ]

    resolution = resolve_type(results, TABLE)

    assert resolution.scam_type is ScamType.FAKE_AUTHORITY
    assert resolution.evidence == ((0, 0),)


# --- ROMANCE_INVESTMENT 的合成 -----------------------------------------


def test_investment_plus_relationship_synthesises_romance_investment() -> None:
    results = [
        hit("guaranteed_return", ScamType.FAKE_INVESTMENT, evidence=((0, 0),)),
        hit("relationship_building", evidence=((1, 2),)),
    ]

    assert resolve_type(results, TABLE).scam_type is ScamType.ROMANCE_INVESTMENT


def test_investment_alone_stays_fake_investment() -> None:
    results = [hit("guaranteed_return", ScamType.FAKE_INVESTMENT)]

    assert resolve_type(results, TABLE).scam_type is ScamType.FAKE_INVESTMENT


def test_relationship_alone_produces_no_candidate() -> None:
    """關係經營規則的 `scam_types` 是空的 ——「這段對話在經營關係」不是一種類型。"""
    assert resolve_type([hit("relationship_building")], TABLE).scam_type is None


def test_single_forwarded_message_degrades_to_fake_investment() -> None:
    """單則轉傳時關係經營幾乎不可能命中。這是正確的降級而非誤判。"""
    results = [hit("guaranteed_return", ScamType.FAKE_INVESTMENT), miss("relationship_building")]

    assert resolve_type(results, TABLE).scam_type is ScamType.FAKE_INVESTMENT


def test_synthesis_evidence_points_at_both_signals() -> None:
    doc = build_document(
        [Message(text="跟著老師操作保證獲利"), Message(text="我一個人在國外。要不要換 LINE 聊")]
    )
    results = [
        hit("guaranteed_return", ScamType.FAKE_INVESTMENT, evidence=((0, 0),)),
        hit("relationship_building", evidence=((1, 0), (1, 1))),
    ]

    resolution = resolve_type(results, TABLE)

    assert resolution.evidence == ((0, 0), (1, 0), (1, 1))
    assert all(isinstance(doc.raw_at(coord), str) for coord in resolution.evidence)


# --- PHISHING_LINK 讓位 -------------------------------------------------


def test_phishing_link_yields_to_a_pretext_type() -> None:
    """假檢警訊息附惡意連結：兩者 hard 與權重相同，件數排序會選錯。"""
    results = [
        hit("url_blocklist", ScamType.PHISHING_LINK, hard=True),
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True),
    ]

    assert resolve_type(results, TABLE).scam_type is ScamType.FAKE_AUTHORITY


def test_phishing_link_alone_does_not_yield() -> None:
    results = [hit("url_blocklist", ScamType.PHISHING_LINK, hard=True)]

    assert resolve_type(results, TABLE).scam_type is ScamType.PHISHING_LINK


def test_yielding_leaves_the_result_in_checks_and_in_the_score() -> None:
    registry = CheckRegistry()
    registry.register(
        StaticCheck("url_blocklist", [hit("url_blocklist", ScamType.PHISHING_LINK, hard=True)])
    )
    registry.register(
        StaticCheck("safe_account", [hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True)])
    )

    verdict = detect(Request.from_text("請匯入監管帳戶 https://evil.com"), registry, TABLE)

    assert verdict.scam_type is ScamType.FAKE_AUTHORITY
    assert [r.name for r in verdict.checks if r.hit] == ["url_blocklist", "safe_account"]
    assert verdict.scam_probability is not None


# --- 三層排序 -----------------------------------------------------------


def test_hard_beats_soft_at_equal_weight() -> None:
    """權重相同時硬證據優先 —— 件數排序會選 FAKE_INVESTMENT（30,053 件）。"""
    table = TABLE.with_overrides(weights={("guaranteed_return", False): 2.5})
    results = [
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True),
        hit("guaranteed_return", ScamType.FAKE_INVESTMENT),
    ]

    assert resolve_type(results, table).scam_type is ScamType.FAKE_AUTHORITY


def test_weight_decides_when_hard_is_equal() -> None:
    results = [
        hit("guaranteed_return", ScamType.FAKE_INVESTMENT),
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True),
    ]

    assert resolve_type(results, TABLE).scam_type is ScamType.FAKE_AUTHORITY


def test_case_count_decides_when_hard_and_weight_are_equal() -> None:
    """`prepay_to_receive` 的四個候選 hard 與權重完全相同 —— 由件數決定。"""
    results = [
        hit(
            "prepay_to_receive",
            ScamType.FAKE_PRIZE,
            ScamType.FAKE_LOAN,
            ScamType.FAKE_INVESTMENT,
            ScamType.FAKE_JOB,
            hard=True,
        )
    ]

    assert resolve_type(results, TABLE).scam_type is ScamType.FAKE_INVESTMENT


def test_multi_type_signal_does_not_dilute() -> None:
    """四個候選各帶完整權重，不依成員數量折減。"""
    four = hit(
        "prepay_to_receive",
        ScamType.FAKE_PRIZE,
        ScamType.FAKE_LOAN,
        ScamType.FAKE_INVESTMENT,
        ScamType.FAKE_JOB,
        hard=True,
    )
    one = hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True)

    assert resolve_type([four, one], TABLE).scam_type is ScamType.FAKE_INVESTMENT


# --- 決定性 -------------------------------------------------------------


def test_result_does_not_depend_on_input_order() -> None:
    results = [
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True),
        hit("url_blocklist", ScamType.PHISHING_LINK, hard=True),
        hit("guaranteed_return", ScamType.FAKE_INVESTMENT),
        hit("parcel_notice", ScamType.FAKE_PARCEL),
    ]
    rng = random.Random(20260914)

    answers = set()
    for _ in range(20):
        shuffled = list(results)
        rng.shuffle(shuffled)
        answers.add(resolve_type(shuffled, TABLE).scam_type)

    assert answers == {ScamType.FAKE_AUTHORITY}


class StaticCheck:
    """回傳固定結果的假檢查。"""

    stage = Stage.LOCAL

    def __init__(self, name: str, results: list[CheckResult]) -> None:
        self.name = name
        self.results = results

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return list(self.results)


def test_registration_order_does_not_change_the_type() -> None:
    checks = [
        StaticCheck("url_blocklist", [hit("url_blocklist", ScamType.PHISHING_LINK, hard=True)]),
        StaticCheck("safe_account", [hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True)]),
    ]
    forward = CheckRegistry()
    for check in checks:
        forward.register(check)
    backward = CheckRegistry()
    for check in reversed(checks):
        backward.register(check)

    request = Request.from_text("請匯入監管帳戶 https://evil.com")

    assert (
        detect(request, forward, TABLE).scam_type
        == detect(request, backward, TABLE).scam_type
        == ScamType.FAKE_AUTHORITY
    )


# --- 衝突與規則優先 -----------------------------------------------------


def test_unequal_sets_with_the_same_top_are_not_a_conflict() -> None:
    results = [
        hit("url_blocklist", ScamType.PHISHING_LINK, hard=True),
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True),
        hit("llm_category", ScamType.FAKE_AUTHORITY),
    ]

    assert resolve_type(results, TABLE_WITH_LLM, LLM_SIGNALS).conflict is False


def test_different_tops_are_a_conflict() -> None:
    results = [
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True),
        hit("llm_category", ScamType.FAKE_LOAN),
    ]

    assert resolve_type(results, TABLE_WITH_LLM, LLM_SIGNALS).conflict is True


def test_refinement_is_not_a_conflict() -> None:
    assert is_conflict(ScamType.FAKE_INVESTMENT, ScamType.ROMANCE_INVESTMENT) is False
    assert REFINEMENTS == frozenset({(ScamType.FAKE_INVESTMENT, ScamType.ROMANCE_INVESTMENT)})


def test_rule_source_wins_a_conflict() -> None:
    results = [
        hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True),
        hit("llm_category", ScamType.FAKE_LOAN),
    ]

    assert resolve_type(results, TABLE_WITH_LLM, LLM_SIGNALS).scam_type is ScamType.FAKE_AUTHORITY


def test_llm_fills_in_when_rules_have_no_candidate() -> None:
    results = [hit("evasion_invisible"), hit("llm_category", ScamType.GUESS_WHO)]

    assert resolve_type(results, TABLE_WITH_LLM, LLM_SIGNALS).scam_type is ScamType.GUESS_WHO


def test_llm_cannot_refine_on_its_own() -> None:
    """關係經營未命中時，LLM MUST NOT 把 FAKE_INVESTMENT 升級為 ROMANCE_INVESTMENT。"""
    results = [
        hit("guaranteed_return", ScamType.FAKE_INVESTMENT),
        hit("llm_category", ScamType.ROMANCE_INVESTMENT),
        miss("relationship_building"),
    ]

    assert resolve_type(results, TABLE_WITH_LLM, LLM_SIGNALS).scam_type is ScamType.FAKE_INVESTMENT


def test_no_candidate_at_all_yields_none() -> None:
    resolution = resolve_type([hit("evasion_invisible")], TABLE)

    assert resolution.scam_type is None
    assert resolution.evidence == ()


def test_llm_absent_completes_normally() -> None:
    results = [hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True)]

    assert resolve_type(results, TABLE).scam_type is ScamType.FAKE_AUTHORITY


# --- 類型無高低、衝突不影響信心、契約未變 -------------------------------


def test_implementation_has_no_severity_comparison() -> None:
    """類型之間沒有高低。只看**程式碼**：docstring 必須談論這件事才能解釋它。"""
    tree = ast.parse((REPO_ROOT / "scam_guard" / "type_resolve.py").read_text(encoding="utf-8"))
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef)
    docstrings = {
        ast.get_docstring(node, clean=False) for node in ast.walk(tree) if isinstance(node, holders)
    }
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    identifiers |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }

    assert not [text for text in identifiers | literals if "severity" in text or "嚴重" in text]


def test_conflict_does_not_lower_confidence() -> None:
    registry = CheckRegistry()
    registry.register(
        StaticCheck("safe_account", [hit("safe_account", ScamType.FAKE_AUTHORITY, hard=True)])
    )

    verdict = detect(Request.from_text("請匯入監管帳戶"), registry, TABLE)

    assert verdict.confidence == TABLE.threshold("base_hard")
    assert verdict.scam_probability is not None


def test_verdict_fields_are_unchanged() -> None:
    assert [field for field in Verdict.__dataclass_fields__] == [
        "scam_probability",
        "confidence",
        "scam_type",
        "evidence",
        "actions",
        "checks",
    ]


def test_resolve_type_does_not_modify_its_inputs() -> None:
    results = [
        hit("guaranteed_return", ScamType.FAKE_INVESTMENT, evidence=((0, 0),)),
        hit("relationship_building", evidence=((1, 0),)),
    ]
    doc = build_document([Message(text="保證獲利"), Message(text="我一個人在國外")])
    before_results = copy.deepcopy(results)
    before_doc = copy.deepcopy(doc)

    resolve_type(results, TABLE)

    assert results == before_results
    assert doc == before_doc


def test_module_does_not_import_pipeline_or_rules() -> None:
    source = (REPO_ROOT / "scam_guard" / "type_resolve.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert not any("scam_guard.pipeline" in line for line in imports)
    assert not any("scam_guard.rules" in line for line in imports)
