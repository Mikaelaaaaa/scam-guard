"""消融：逐訊號關閉、分群拆解、建議表，以及「不修改權重表」這條界線。"""

from hashlib import sha256
from pathlib import Path

from scam_guard.check import CheckRegistry
from scam_guard.weights import load_weights
from tools.eval.ablation import (
    BLOCKED,
    CONFIDENCE_SWEEPS,
    DOMAIN_AGE_THRESHOLDS,
    PLACEHOLDER_LIMITATION,
    PRIOR_SWEEP,
    URL_BRAND_SHORT_LABEL_LENGTHS,
    Outcome,
    _outcome,
    contradiction_rate,
    contribution,
    prior_sweep,
    split_groups,
    truncation_count,
    truncation_invariant_holds,
)
from tools.eval.random_escalation import DEFAULT_SEED
from tools.eval.recommendations import (
    DISCLAIMER,
    INSUFFICIENT,
    KEEP,
    REGROUP,
    REMOVE,
    recommend,
)
from tools.eval.run import run_over
from tools.eval.stats import Rate
from tests.test_random_escalation import KeywordCheck, registry_with, samples

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE = load_weights()
FLOOR = TABLE.threshold("confidence_floor")


def outcome_of(records: object) -> Outcome:
    return _outcome(records)  # type: ignore[arg-type]


# --- 逐訊號消融 -----------------------------------------------------------


def test_ablating_a_signal_that_never_hits_changes_nothing() -> None:
    """零命中的訊號被關閉後，Δ召回率與Δ誤判率皆為零。"""
    registry = registry_with(KeywordCheck())
    baseline = run_over(samples(), registry, TABLE)
    result = contribution(
        "parcel_notice",
        "signal",
        ["parcel_notice"],
        samples(),
        registry,
        TABLE,
        baseline,
        outcome_of(baseline),
        seed=DEFAULT_SEED,
    )
    assert result.registered is False
    assert result.delta_recall == 0.0
    assert result.delta_false_positive_rate == 0.0
    assert "未註冊" in result.note


def test_an_unregistered_signal_is_recorded_not_skipped() -> None:
    """`domain_age` 預設不註冊 —— 那件事本身就是報告要寫的一行，不是一次中止。"""
    registry = registry_with(KeywordCheck())
    baseline = run_over(samples(), registry, TABLE)
    result = contribution(
        "domain_age",
        "signal",
        ["domain_age"],
        samples(),
        registry,
        TABLE,
        baseline,
        outcome_of(baseline),
        seed=DEFAULT_SEED,
    )
    assert result.registered is False
    assert result.hit_rate is None
    assert result.distinguishable_from_random is None


# --- 分群拆解 -------------------------------------------------------------


def test_splitting_groups_makes_every_signal_its_own_group_without_touching_the_original() -> None:
    split = split_groups(TABLE)
    assert len(split.groups) == len(split.signals)
    assert all(members == (name,) for name, members in split.groups.items())
    assert len(TABLE.groups) < len(TABLE.signals)
    assert split.path == TABLE.path


def test_non_unit_groups_are_the_split_targets() -> None:
    """非單元群組即分群拆解的對象。

    不斷言「恰好六個」——`llm-layer` 落地時多了 `llm_semantic`，而那是預期的
    擴充不是回歸。改為斷言規則層與 URL 層的六個群組都在（那是拆解一定要涵蓋
    的），並要求每個非單元群組的成員都確實登記在表中。
    """
    non_unit = {name for name, members in TABLE.groups.items() if len(members) > 1}
    assert {
        "authority_script",
        "credential_solicit",
        "advance_fee",
        "too_good_offer",
        "url_reputation",
        "evasion",
    } <= non_unit
    for name in non_unit:
        for member in TABLE.groups[name]:
            assert member in TABLE.signals, f"群組 {name!r} 的成員 {member!r} 不在表中"


# --- 門檻掃描的範圍守住相對關係 -------------------------------------------


def test_confidence_sweep_ranges_respect_the_six_invariants() -> None:
    """例：`base_no_hit` 的上界不超過地板，否則「完全無訊號必定拒答」就被打破。"""
    for key in ("base_no_hit", "base_single_group", "cap_contradiction", "cap_unseen_pattern"):
        assert max(CONFIDENCE_SWEEPS[key]) < FLOOR, key
    for key in ("base_hard", "base_multi_group", "cap_truncated"):
        assert min(CONFIDENCE_SWEEPS[key]) > FLOOR, key
    assert set(CONFIDENCE_SWEEPS) == {
        "base_hard",
        "base_multi_group",
        "base_single_group",
        "base_no_hit",
        "cap_contradiction",
        "cap_unseen_pattern",
        "cap_truncated",
        "confidence_floor",
    }


def test_prior_sweep_is_pure_arithmetic_and_zero_reproduces_the_baseline() -> None:
    registry = registry_with(KeywordCheck())
    baseline = run_over(samples(), registry, TABLE)
    rows = {value: outcome for _, value, outcome in prior_sweep(baseline, TABLE, PRIOR_SWEEP)}
    assert 0.0 in rows
    assert rows[0.0].recall == outcome_of(baseline).recall
    assert rows[1.5].recall.numerator >= rows[0.0].recall.numerator


# --- 材料不足時誠實記錄 ---------------------------------------------------


def test_no_sample_triggers_truncation_so_no_sensitivity_curve_is_produced() -> None:
    assert truncation_count(samples()) == 0
    assert truncation_invariant_holds(105) is True
    assert truncation_invariant_holds(3) is True


# --- 引述兩段式的代價 -----------------------------------------------------


def test_contradiction_rate_is_computed_on_the_scam_side() -> None:
    registry = registry_with(KeywordCheck())
    records = run_over(samples(), registry, TABLE)
    rate = contradiction_rate(records)
    assert rate.denominator == 100
    assert rate.numerator == 0


# --- 建議表 ---------------------------------------------------------------


def test_recommendation_values_cover_the_four_cases() -> None:
    zero = recommend(
        "x",
        "signal",
        registered=True,
        hit_rate=Rate(numerator=0, denominator=100),
        delta_recall=0.0,
        shadowed_count=0,
        distinguishable_from_random=None,
    )
    assert zero.verdict == REMOVE

    shadowed = recommend(
        "x",
        "signal",
        registered=True,
        hit_rate=Rate(numerator=10, denominator=100),
        delta_recall=-0.01,
        shadowed_count=10,
        distinguishable_from_random=True,
    )
    assert shadowed.verdict == REGROUP

    random_like = recommend(
        "x",
        "signal",
        registered=True,
        hit_rate=Rate(numerator=10, denominator=100),
        delta_recall=-0.01,
        shadowed_count=0,
        distinguishable_from_random=False,
    )
    assert random_like.verdict == INSUFFICIENT
    assert "無法與隨機注入區分" in random_like.reason

    useful = recommend(
        "x",
        "signal",
        registered=True,
        hit_rate=Rate(numerator=10, denominator=100),
        delta_recall=-0.05,
        shadowed_count=0,
        distinguishable_from_random=True,
    )
    assert useful.verdict == KEEP

    missing = recommend(
        "x",
        "signal",
        registered=False,
        hit_rate=None,
        delta_recall=0.0,
        shadowed_count=0,
        distinguishable_from_random=None,
    )
    assert missing.verdict == INSUFFICIENT


def test_every_recommendation_carries_its_supporting_numbers() -> None:
    advice = recommend(
        "x",
        "signal",
        registered=True,
        hit_rate=Rate(numerator=10, denominator=100),
        delta_recall=-0.05,
        shadowed_count=0,
        distinguishable_from_random=True,
    )
    assert "%" in advice.reason


def test_the_disclaimer_says_it_is_not_a_change_authorisation() -> None:
    assert "不構成" in DISCLAIMER
    assert "weights.toml" in DISCLAIMER


# --- 不修改權重表 ---------------------------------------------------------


def test_running_the_whole_flow_leaves_weights_toml_byte_identical() -> None:
    before = sha256(TABLE.path.read_bytes()).hexdigest()
    registry = registry_with(KeywordCheck())
    baseline = run_over(samples(), registry, TABLE)
    contribution(
        "solicit_otp",
        "signal",
        ["solicit_otp"],
        samples(),
        registry,
        TABLE,
        baseline,
        outcome_of(baseline),
        seed=DEFAULT_SEED,
    )
    split_groups(TABLE)
    prior_sweep(baseline, TABLE, PRIOR_SWEEP)
    TABLE.with_overrides(thresholds={"confidence_floor": 0.9})
    assert sha256(TABLE.path.read_bytes()).hexdigest() == before


def test_the_ablation_module_never_writes_to_the_weight_table() -> None:
    source = (REPO_ROOT / "tools" / "eval" / "ablation.py").read_text(encoding="utf-8")
    assert "write_text(" not in source.split("def _handoff_markdown")[0]
    assert "weights.toml" not in source.replace("scam_guard/tables/weights.toml", "")


# --- blocked 的交辦被誠實標記 ---------------------------------------------


def test_blocked_handoffs_are_named_with_their_blocking_reason() -> None:
    assert "add-tranco-allowlist" in BLOCKED
    assert "pii_nlp" in BLOCKED
    for reason in BLOCKED.values():
        assert "不可執行" in reason or "不執行" in reason or "不重開" in reason
    assert "blocked_on" in BLOCKED["add-tranco-allowlist"]
    assert "blocked_on" in BLOCKED["pii_nlp"]


def test_placeholder_limitation_is_stated_as_a_fixed_paragraph() -> None:
    assert "placeholder" in PLACEHOLDER_LIMITATION
    assert "命中率" in PLACEHOLDER_LIMITATION
    assert "判別力" in PLACEHOLDER_LIMITATION


def test_swept_parameter_ranges_are_registered_constants() -> None:
    assert URL_BRAND_SHORT_LABEL_LENGTHS == (4, 5, 6)
    assert DOMAIN_AGE_THRESHOLDS == (7, 14, 30, 60, 90)
    assert 30 in DOMAIN_AGE_THRESHOLDS and 60 in DOMAIN_AGE_THRESHOLDS


def test_a_fresh_registry_is_used_for_each_ablation() -> None:
    """消融不得改到基準 registry —— 改到了的話後面每一輪都少一個檢查。"""
    registry = registry_with(KeywordCheck())
    before = [check.name for check in registry.enabled()]
    baseline = run_over(samples(), registry, TABLE)
    contribution(
        "solicit_otp",
        "signal",
        ["solicit_otp"],
        samples(),
        registry,
        TABLE,
        baseline,
        outcome_of(baseline),
        seed=DEFAULT_SEED,
    )
    assert [check.name for check in registry.enabled()] == before
    assert isinstance(registry, CheckRegistry)
