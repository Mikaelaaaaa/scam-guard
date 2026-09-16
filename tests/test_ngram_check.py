"""字元 n-gram Check 的輸出契約。"""

import math
import json
from pathlib import Path

import pytest

import scam_guard.ngram as ngram
from scam_guard.check import CheckRegistry, Stage
from scam_guard.ngram import NgramClassifierCheck, NgramModel, register_ngram_check
from scam_guard.normalize import build_document
from scam_guard.pipeline import detect
from scam_guard.render import CALIBRATION_CLAIMS, SPECULATIVE_TERMS, VERDICT_CLAIMS
from scam_guard.types import Request
from scam_guard.weights import load_weights


TABLE = load_weights()


def _model(*, intercept: float) -> NgramModel:
    return NgramModel(
        path=ngram.DEFAULT_MODEL_PATH,
        manifest={"ngram_range": [1, 2], "n_scam": 572, "n_ham": 824},
        intercept=intercept,
        terms={},
    )


def test_check_shape_and_below_threshold_result() -> None:
    request = Request.from_text("今天下午開會")
    doc = build_document(request.messages)
    check = NgramClassifierCheck(_model(intercept=-10.0), TABLE)
    assert check.stage is Stage.LOCAL
    assert check(request, doc) == []


def test_hit_is_soft_untyped_and_detail_does_not_leak() -> None:
    request = Request.from_text("今天下午開會")
    result = NgramClassifierCheck(_model(intercept=10.0), TABLE)(
        request, build_document(request.messages)
    )[0]
    assert result.hit
    assert result.hard is False
    assert result.scam_types == []
    assert result.evidence == []
    forbidden = ("ngram_classifier", "10.0", "4.351698", "今天下午開會")
    assert not any(value in result.detail for value in forbidden)
    assert not any(value in result.detail for value in SPECULATIVE_TERMS)
    assert not any(value in result.detail for value in VERDICT_CLAIMS)
    assert not any(value in result.detail for value in CALIBRATION_CLAIMS)


def test_no_model_means_no_registration() -> None:
    registry = CheckRegistry()
    register_ngram_check(registry, TABLE)
    assert registry.enabled() == []


def test_evidence_share_does_not_change_score_or_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ngram.load_model()
    text = "包裹因關稅未繳而暫扣，請立即付款"
    before = ngram.score(model, text)
    request = Request.from_text(text)
    check = NgramClassifierCheck(model, TABLE)
    first = bool(check(request, build_document(request.messages)))
    monkeypatch.setattr(ngram, "EVIDENCE_MASS_SHARE", 0.8)
    after = ngram.score(model, text)
    second = bool(check(request, build_document(request.messages)))
    assert math.isclose(before.value, after.value, abs_tol=0.0)
    assert first == second


def test_soft_only_weight_rejects_hard_lookup() -> None:
    with pytest.raises(ValueError, match="hard_capable=false"):
        TABLE.weight_for("ngram_classifier", hard=True)


def test_signal_and_threshold_share_the_same_measurement() -> None:
    weight = TABLE.signals["ngram_classifier"].weight_soft
    threshold = TABLE.thresholds["ngram_threshold"]
    assert weight.measured_on == threshold.measured_on
    assert weight.p_hit_given_scam == threshold.p_hit_given_scam
    assert weight.p_hit_given_ham == threshold.p_hit_given_ham
    assert math.isclose(
        weight.value,
        math.log(weight.p_hit_given_scam / weight.p_hit_given_ham),  # type: ignore[operator]
        abs_tol=0.01,
    )


def test_generic_scoring_modules_do_not_name_the_classifier() -> None:
    root = Path(__file__).resolve().parent.parent
    for relative in (
        "scam_guard/weights.py",
        "scam_guard/scoring.py",
        "scam_guard/confidence.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "ngram_classifier" not in source
        assert "ngram_threshold" not in source


def test_classifier_only_still_abstains() -> None:
    registry = CheckRegistry()
    register_ngram_check(registry, TABLE, model=_model(intercept=10.0))
    verdict = detect(Request.from_text("今天下午開會"), registry, TABLE)
    assert verdict.scam_probability is None
    assert verdict.scam_type is None


def test_phishing_case_records_the_unchanged_result() -> None:
    import app

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "ngram_phishing_case.json").read_text(
            encoding="utf-8"
        )
    )
    registry = app.build_registry(None, None, None, None)
    verdict = detect(Request.from_text(fixture["text"]), registry, app.TABLE)
    expected = fixture["after"]
    measured_score = ngram.score(ngram.load_model(), fixture["text"])
    assert measured_score.value == pytest.approx(expected["ngram_score"])
    assert app.TABLE.threshold("ngram_threshold") == expected["ngram_threshold"]
    assert verdict.scam_probability is expected["scam_probability"]
    assert verdict.confidence == expected["confidence"]
    assert verdict.scam_type is expected["scam_type"]
    assert len(verdict.evidence) == expected["evidence_lines"]
    assert len(verdict.checks) == expected["checks"]
    assert sum(result.hit for result in verdict.checks) == expected["hits"]
