"""輸出契約 `CheckResult` / `Verdict` 的行為。"""

from dataclasses import FrozenInstanceError

import pytest

from scam_guard.types import CheckResult, Verdict


def test_check_result_with_all_fields() -> None:
    result = CheckResult(
        name="brand_similarity",
        hit=True,
        weight=1.8,
        detail="0.87/0.85",
        evidence=[(0, 3), (0, 4)],
        scam_types=["釣魚網站"],
        hard=True,
    )

    assert result.name == "brand_similarity"
    assert result.hit is True
    assert result.weight == 1.8
    assert result.detail == "0.87/0.85"
    assert result.evidence == [(0, 3), (0, 4)]
    assert result.scam_types == ["釣魚網站"]
    assert result.hard is True


def test_check_result_evidence_may_span_messages() -> None:
    result = CheckResult(
        name="trajectory",
        hit=True,
        weight=1.2,
        detail="關係建立後隨即出現金錢話題",
        evidence=[(0, 2), (4, 0)],
    )

    assert result.evidence == [(0, 2), (4, 0)]


def test_hit_without_evidence_is_legal() -> None:
    """整體特徵的訊號（則數異常、時間集中度）不指向特定句子。"""
    result = CheckResult(name="burst", hit=True, weight=0.4, detail="10 分鐘內 30 則")

    assert result.hit is True
    assert result.evidence == []


def test_check_result_defaults_are_empty_and_not_shared() -> None:
    first = CheckResult(name="blocklist", hit=False, weight=0.0, detail="未命中")
    second = CheckResult(name="domain_age", hit=False, weight=0.0, detail="未命中")

    assert first.evidence == []
    assert first.scam_types == []
    assert first.hard is False
    assert first.evidence is not second.evidence
    assert first.scam_types is not second.scam_types


def test_verdict_scam_probability_can_be_none() -> None:
    verdict = Verdict(
        scam_probability=None,
        confidence=0.05,
        scam_type=None,
        evidence=[],
        actions=[],
        checks=[],
    )

    assert verdict.scam_probability is None
    assert verdict.confidence == 0.05


def test_verdict_keeps_unhit_checks() -> None:
    checks = [
        CheckResult(
            name="blocklist", hit=True, weight=2.5, detail="命中 165 涉詐網站清單", hard=True
        ),
        CheckResult(name="solicit_otp", hit=True, weight=1.5, detail="第 2 句索取驗證碼"),
        CheckResult(name="domain_age", hit=False, weight=0.0, detail="未命中"),
        CheckResult(name="evasion", hit=False, weight=0.0, detail="未命中"),
        CheckResult(name="quotation", hit=False, weight=0.0, detail="未命中"),
    ]
    verdict = Verdict(
        scam_probability=0.87,
        confidence=0.9,
        scam_type="假檢警",
        evidence=["要求匯入監管帳戶（我國法制不存在此類帳戶）"],
        actions=["不要照做，撥打 165 查證"],
        checks=checks,
    )

    assert len(verdict.checks) == 5
    assert [c.name for c in verdict.checks if not c.hit] == ["domain_age", "evasion", "quotation"]


def test_check_result_is_immutable() -> None:
    result = CheckResult(name="blocklist", hit=False, weight=0.0, detail="未命中")

    with pytest.raises(FrozenInstanceError):
        result.hit = True  # type: ignore[misc]


def test_verdict_is_immutable() -> None:
    verdict = Verdict(
        scam_probability=None,
        confidence=0.0,
        scam_type=None,
        evidence=[],
        actions=[],
        checks=[],
    )

    with pytest.raises(FrozenInstanceError):
        verdict.confidence = 1.0  # type: ignore[misc]
