"""計分：對數勝算、同群組取 max、下限、引述的兩段式處置、機率映射。"""

import ast
import math
import random
from pathlib import Path

import pytest

from scam_guard.pipeline import NOT_HIT, SKIPPED
from scam_guard.scoring import _sigmoid, abstention_rate, compute_score, is_decision
from scam_guard.types import CheckResult, ScamType, Verdict
from scam_guard.weights import load_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE = load_weights()

AUTHORITY_SCRIPT = ("safe_account", "atm_operation", "secrecy_demand")


def hit(name: str, *, hard: bool = False, detail: str = "命中") -> CheckResult:
    return CheckResult(name=name, hit=True, detail=detail, hard=hard)


def miss(name: str) -> CheckResult:
    return CheckResult(name=name, hit=False, detail=NOT_HIT)


def skipped(name: str) -> CheckResult:
    return CheckResult(name=name, hit=False, detail=SKIPPED)


# --- 尺度與只採計命中 ----------------------------------------------------


def test_no_hits_scores_zero() -> None:
    score = compute_score([miss("solicit_otp"), miss("url_brand")], TABLE)

    assert score.value == 0.0
    assert score.probability == 0.5


def test_unhit_and_skipped_records_do_not_score() -> None:
    with_noise = compute_score(
        [hit("solicit_otp", hard=True), miss("url_brand"), skipped("domain_age")], TABLE
    )
    clean = compute_score([hit("solicit_otp", hard=True)], TABLE)

    assert with_noise.value == clean.value == 2.5


def test_unlisted_signal_name_raises() -> None:
    with pytest.raises(KeyError, match="沒登錄的訊號"):
        compute_score([hit("沒登錄的訊號")], TABLE)


# --- 同群組取 max --------------------------------------------------------


def test_same_group_takes_the_max_not_the_sum() -> None:
    results = [hit(name, hard=True) for name in AUTHORITY_SCRIPT]

    score = compute_score(results, TABLE)

    assert score.value == 2.5
    assert [c.group for c in score.group_contributions] == ["authority_script"]


def test_two_groups_add_up() -> None:
    score = compute_score([hit("safe_account", hard=True), hit("guaranteed_return")], TABLE)

    assert score.value == pytest.approx(3.1)


def test_hard_and_soft_in_one_group_takes_the_hard_weight() -> None:
    score = compute_score([hit("safe_account", hard=True), hit("atm_operation")], TABLE)

    assert score.value == 2.5


def test_two_malicious_domains_count_once() -> None:
    """取 max 的範圍是群組，不細分為（群組 × 證據主體）。"""
    results = [
        hit("url_blocklist", hard=True, detail="evil-a.com"),
        hit("url_blocklist", hard=True, detail="evil-b.com"),
    ]

    score = compute_score(results, TABLE)

    assert score.value == 2.5


def test_shadowed_results_are_kept() -> None:
    results = [hit(name, hard=True) for name in AUTHORITY_SCRIPT]

    (contribution,) = compute_score(results, TABLE).group_contributions

    assert len(contribution.shadowed) == 2
    assert contribution.taken in results


# --- 下限 ---------------------------------------------------------------


def test_negative_weight_cannot_push_the_score_below_zero() -> None:
    score = compute_score([hit("guaranteed_return"), hit("quotation")], TABLE)

    assert score.value == 0.0
    assert score.probability == 0.5
    assert score.quotation_applied is True


def test_many_negative_signals_still_floor_at_zero() -> None:
    derived = TABLE.with_overrides(weights={("quotation", False): -10.0})

    score = compute_score([hit("quotation")], derived)

    assert score.value == 0.0


def test_floor_does_not_touch_the_ordinary_case() -> None:
    score = compute_score([hit("safe_account", hard=True), hit("guaranteed_return")], TABLE)

    assert score.value == pytest.approx(3.1)


# --- 引述的兩段式處置 ----------------------------------------------------


def test_tier_a_with_quotation_does_not_subtract() -> None:
    score = compute_score([hit("safe_account", hard=True), hit("quotation")], TABLE)

    assert score.value == 2.5
    assert score.quotation_applied is False


def test_tier_a_with_quotation_marks_contradiction() -> None:
    score = compute_score([hit("safe_account", hard=True), hit("quotation")], TABLE)

    assert score.contradicted is True


def test_blocklist_only_hard_evidence_is_not_a_contradiction() -> None:
    """黑名單命中是關於世界的事實，與「這則訊息是引述」在邏輯上不衝突。"""
    score = compute_score([hit("url_blocklist", hard=True), hit("quotation")], TABLE)

    assert score.contradicted is False


def test_blocklist_together_with_a_rule_is_a_contradiction() -> None:
    """例外只涵蓋硬證據**全部**來自黑名單的情形。"""
    results = [
        hit("url_blocklist", hard=True),
        hit("safe_account", hard=True),
        hit("quotation"),
    ]

    assert compute_score(results, TABLE).contradicted is True


def test_quotation_not_hit_means_no_contradiction() -> None:
    score = compute_score([hit("safe_account", hard=True), miss("quotation")], TABLE)

    assert score.contradicted is False
    assert score.quotation_applied is False


def test_quotation_check_not_registered_at_all() -> None:
    score = compute_score([hit("safe_account", hard=True)], TABLE)

    assert score.contradicted is False
    assert score.value == 2.5


def test_awareness_post_regression() -> None:
    """「最近很多假檢警詐騙，會叫你把錢匯到監管帳戶，千萬不要相信」

    這一則同時命中 `safe_account`（Tier-A）與 `quotation`（宣導框架）。
    **被取代的結果**：負權重方案得到 `2.5 - 1.5 = +1.0`，仍判為詐騙 ——
    一則防詐宣導文被判為詐騙。現在的結果是分數照舊 `2.5`、標記矛盾，
    由信心層把輸出降為「無法判定」加上查證建議。
    """
    results = [hit("safe_account", hard=True, detail="要求匯入監管帳戶"), hit("quotation")]

    score = compute_score(results, TABLE)

    assert score.contradicted is True
    assert score.value == 2.5


def test_adversarial_awareness_prefix_regression() -> None:
    """宣導框架前綴 + 黑名單精確命中的釣魚連結 —— 照常判為詐騙。"""
    results = [hit("url_blocklist", hard=True, detail="命中 165 涉詐網站清單"), hit("quotation")]

    score = compute_score(results, TABLE)

    assert score.contradicted is False
    assert score.value >= TABLE.threshold("decision_score")


# --- 機率映射 -----------------------------------------------------------


def test_probability_is_strictly_monotonic() -> None:
    scores = [-5.0, -1.0, 0.0, 0.6, 2.5, 10.0]
    probabilities = [_sigmoid(value) for value in scores]

    assert probabilities == sorted(probabilities)
    assert len(set(probabilities)) == len(probabilities)


def test_extreme_scores_do_not_overflow() -> None:
    assert _sigmoid(1e6) == pytest.approx(1.0)
    assert _sigmoid(-1e6) == pytest.approx(0.0)


def test_zero_score_maps_to_one_half() -> None:
    assert _sigmoid(0.0) == 0.5


def test_temperature_can_be_swept_without_touching_the_original() -> None:
    derived = TABLE.with_overrides(thresholds={"sigmoid_temperature": 2.0})
    results = [hit("safe_account", hard=True)]

    assert compute_score(results, derived).probability == pytest.approx(1 / (1 + math.exp(-1.25)))
    assert compute_score(results, TABLE).probability == pytest.approx(1 / (1 + math.exp(-2.5)))


# --- 單調升級 -----------------------------------------------------------


def test_adding_a_hit_in_a_new_group_raises_the_score() -> None:
    base = compute_score([hit("safe_account", hard=True)], TABLE)
    more = compute_score([hit("safe_account", hard=True), hit("guaranteed_return")], TABLE)

    assert more.value == pytest.approx(base.value + 0.6)


def test_adding_a_weaker_hit_in_the_same_group_changes_nothing() -> None:
    base = compute_score([hit("safe_account", hard=True)], TABLE)
    more = compute_score([hit("safe_account", hard=True), hit("atm_operation")], TABLE)

    assert more.value == base.value


NON_NEGATIVE_SIGNALS = tuple(
    name for name in load_weights().signals if name != load_weights().roles["quotation"]
)


def test_more_evidence_never_lowers_the_score() -> None:
    """性質測試：加入任一權重非負的新命中，分數恆不下降。"""
    rng = random.Random(20260914)
    for _ in range(200):
        chosen = rng.sample(NON_NEGATIVE_SIGNALS, rng.randint(0, 5))
        results = [
            hit(name, hard=TABLE.signals[name].hard_capable and rng.random() < 0.5)
            for name in chosen
        ]
        extra = rng.choice(NON_NEGATIVE_SIGNALS)
        added = hit(extra, hard=TABLE.signals[extra].hard_capable and rng.random() < 0.5)

        before = compute_score(results, TABLE).value
        after = compute_score([*results, added], TABLE).value

        assert after >= before


# --- LLM 缺席 -----------------------------------------------------------


def test_scoring_module_names_no_llm_check() -> None:
    """計分只認「命中的結果」與「表裡的條目」，不認得任何 LLM 檢查的名稱。

    只看**程式碼中的字串常數**：docstring 可以談論 LLM（它必須談，那是設計的
    一部分），但不得有任何一個字串字面值是某個 LLM 檢查的名稱。
    """
    tree = ast.parse((REPO_ROOT / "scam_guard" / "scoring.py").read_text(encoding="utf-8"))
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef)
    docstrings = {
        ast.get_docstring(node, clean=False) for node in ast.walk(tree) if isinstance(node, holders)
    }
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    ]

    assert not [text for text in literals if "llm" in text.lower()]


def test_unmounted_and_mounted_but_unhit_score_the_same() -> None:
    unmounted = compute_score([hit("safe_account", hard=True)], TABLE)
    mounted = compute_score([hit("safe_account", hard=True), miss("domain_age")], TABLE)

    assert unmounted.value == mounted.value
    assert unmounted.probability == mounted.probability


# --- 判定事件與棄權率 ----------------------------------------------------


def a_verdict(probability: float | None) -> Verdict:
    return Verdict(
        scam_probability=probability,
        confidence=0.9,
        scam_type=ScamType.FAKE_AUTHORITY,
        evidence=[],
        actions=[],
        checks=[],
    )


def test_single_tier_a_crosses_the_decision_threshold() -> None:
    score = compute_score([hit("safe_account", hard=True)], TABLE)

    assert is_decision(score, a_verdict(score.probability), TABLE) is True


def test_single_tier_b_does_not_cross_the_decision_threshold() -> None:
    score = compute_score([hit("guaranteed_return")], TABLE)

    assert is_decision(score, a_verdict(score.probability), TABLE) is False


def test_abstention_is_not_a_decision() -> None:
    score = compute_score([hit("safe_account", hard=True)], TABLE)

    assert is_decision(score, a_verdict(None), TABLE) is False


def test_abstention_rate_of_an_all_abstaining_set() -> None:
    assert abstention_rate([a_verdict(None) for _ in range(5)]) == 1.0


def test_abstention_rate_counts_only_none() -> None:
    verdicts = [a_verdict(None), a_verdict(0.9), a_verdict(None), a_verdict(0.8)]

    assert abstention_rate(verdicts) == 0.5


def test_abstention_rate_of_an_empty_set_raises() -> None:
    with pytest.raises(ValueError, match="空樣本集"):
        abstention_rate([])


@pytest.mark.skip(
    reason=(
        "誤判率門檻（95% Wilson 上界 ≤ 2%）在 add-testset 之前無法驗證："
        "需 149 則 hard negative；30 則零誤判的 95% Wilson 上界為 11.35%。"
        "此測試刻意**存在**而非不存在 —— skipped 每次跑 pytest 都會被印出來，"
        "而一個不存在的測試不會有任何地方提醒它不存在。"
    )
)
def test_false_positive_rate_on_hard_negatives() -> None:
    raise AssertionError("待 add-testset 提供 hard negative 集後實作")


# --- 界線 ---------------------------------------------------------------


def test_module_does_not_import_pipeline_or_rules() -> None:
    source = (REPO_ROOT / "scam_guard" / "scoring.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert not any("scam_guard.pipeline" in line for line in imports)
    assert not any("scam_guard.rules" in line for line in imports)
