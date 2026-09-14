"""Wilson 區間、比率型別與樣本量對照表。

本檔不吃資料 —— 統計是純函式，它的正確性不該依賴任何一份語料。
"""

import math

import pytest

from tools.eval.stats import (
    ONE_ERROR_SAMPLE_SIZE,
    RULE_OF_THREE_NOTE,
    TARGET_UPPER,
    TWO_ERROR_SAMPLE_SIZE,
    ZERO_ERROR_SAMPLE_SIZE,
    Rate,
    minimum_n_for,
    sample_size_table,
    wilson_interval,
    wilson_upper,
)


def test_rate_on_an_empty_subset_raises() -> None:
    """空子集上的「誤判率 0%」是把大聲的失敗換成安靜的錯答案。"""
    with pytest.raises(ValueError, match="空樣本集"):
        Rate(numerator=0, denominator=0)


def test_rate_carries_its_interval_and_orders_correctly() -> None:
    rate = Rate(numerator=4, denominator=428)
    assert rate.value == pytest.approx(4 / 428)
    assert rate.lower <= rate.value <= rate.upper
    assert "4/428" in str(rate)


def test_rate_fields_cannot_be_supplied_by_the_caller() -> None:
    """`value`/`lower`/`upper` 是推導欄位 —— 提供得了就能提供一個不一致的區間。"""
    with pytest.raises(TypeError):
        Rate(numerator=1, denominator=10, value=0.9)  # type: ignore[call-arg]


def test_zero_hits_gives_a_nonzero_upper_bound() -> None:
    """Wald 在 `x = 0` 時給 `[0, 0]`，而那正是本專題最常遇到的情形。"""
    lower, upper = wilson_interval(0, 400)
    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(0.0095, abs=1e-4)


def test_zero_hit_upper_bound_matches_the_closed_form() -> None:
    """`x = 0` 時 Wilson 上界化簡為 `z² / (n + z²)`。"""
    z_squared = 1.959963984540054**2
    for n in (30, 189, 600):
        assert wilson_upper(0, n) == pytest.approx(z_squared / (n + z_squared))


def test_wilson_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="樣本量"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="不落在"):
        wilson_interval(5, 3)


def test_149_is_rule_of_three_not_wilson() -> None:
    """既有 spec 三處寫的「149 則可宣稱 ≤ 2%」是 `3/n` 的結果，不是 Wilson。

    這條測試的存在就是為了讓任何人改動這一段時看到那兩個數字不一樣。
    """
    assert wilson_upper(0, 149) > TARGET_UPPER
    assert wilson_upper(0, 149) == pytest.approx(0.0251, abs=1e-4)
    assert 3 / 149 < 0.021
    assert wilson_upper(0, ZERO_ERROR_SAMPLE_SIZE) <= TARGET_UPPER
    assert wilson_upper(0, ZERO_ERROR_SAMPLE_SIZE - 1) > TARGET_UPPER


def test_sample_size_table_lists_both_formulas_and_they_differ_at_149() -> None:
    table = dict((n, (wilson, rule_of_three)) for n, wilson, rule_of_three in sample_size_table())
    assert 149 in table and 189 in table
    wilson, rule_of_three = table[149]
    assert wilson != rule_of_three
    assert wilson > rule_of_three
    assert "rule of three" in RULE_OF_THREE_NOTE
    assert "189" in RULE_OF_THREE_NOTE


def test_one_and_two_errors_need_far_more_samples() -> None:
    """189 則的子集只要出現**一次**誤判就不夠了 —— 這決定門檻有多脆弱。"""
    assert minimum_n_for(0) == ZERO_ERROR_SAMPLE_SIZE
    assert minimum_n_for(1) == ONE_ERROR_SAMPLE_SIZE
    assert minimum_n_for(2) == TWO_ERROR_SAMPLE_SIZE
    assert wilson_upper(1, ZERO_ERROR_SAMPLE_SIZE) > TARGET_UPPER


def test_intervals_overlap_is_symmetric_and_conservative() -> None:
    """區間重疊是 `add-ablation` 的判準，**不是假設檢定**。"""
    left = Rate(numerator=10, denominator=100)
    right = Rate(numerator=12, denominator=100)
    far = Rate(numerator=90, denominator=100)
    assert left.overlaps(right) and right.overlaps(left)
    assert not left.overlaps(far)


def test_wilson_matches_a_hand_computed_value() -> None:
    """一個獨立算出來的參考值，防止公式被改成別的區間而測試仍然通過。"""
    x, n, z = 4, 428, 1.959963984540054
    proportion = x / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    assert wilson_interval(x, n) == pytest.approx((centre - half, centre + half))
