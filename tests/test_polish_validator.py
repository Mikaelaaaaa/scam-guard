"""措辭潤飾層的驗證 —— `demo_ui.PolishValidator` 與它的允許集合。

這些測試原本在 `tests/test_gradio_demo.py`，隨 `PolishValidator` 由 `app.py`
搬進 `demo_ui.py` 而獨立成一個檔案：潤飾層不再是 gradio 介面的東西，兩個載體
（`app.py` 與 `docs/pages_app.py`）用的是同一份判定。
**本檔因此不 `importorskip("gradio")`**，斷言的預期值與搬家前逐字相同。
"""

from demo_ui import PolishValidator, allowed_numbers, verdict_segments
from scam_guard.redact import RedactedText
from scam_guard.types import ScamType, Verdict

EMPTY_REDACTED = RedactedText(sentences=[], coords=[], counts={})

EMPTY_VERDICT = Verdict(
    scam_probability=None,
    confidence=0.0,
    scam_type=None,
    evidence=[],
    actions=[],
    checks=[],
    redacted=EMPTY_REDACTED,
)


def verdict_with(**overrides) -> Verdict:
    fields = {
        "scam_probability": None,
        "confidence": 0.0,
        "scam_type": None,
        "evidence": [],
        "actions": [],
        "checks": [],
        "redacted": EMPTY_REDACTED,
    }
    fields.update(overrides)
    return Verdict(**fields)


def test_polish_rejects_invented_number() -> None:
    validator = PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("這則訊息有 87% 的機率是詐騙") is False


def test_polish_rejects_url() -> None:
    validator = PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("你可以去 http") is False


def test_polish_rejects_www_prefix() -> None:
    validator = PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("去看看 www.") is False


def test_polish_rejects_scam_type_absent_from_verdict() -> None:
    validator = PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("這看起來是假檢警/假冒公務機關的手法") is False


def test_polish_accepts_scam_type_present_in_verdict() -> None:
    verdict = verdict_with(scam_type=ScamType.FAKE_AUTHORITY)
    validator = PolishValidator(["這是假檢警/假冒公務機關。撥打 165 查證。"], verdict)
    assert validator.feed("聽起來是假檢警/假冒公務機關，我先撥打 165 查證") is True


def test_polish_accepts_rewording() -> None:
    segments = ["要求不得告知家人、行員或警察。"]
    validator = PolishValidator(segments, EMPTY_VERDICT)
    assert validator.feed("你叫我不要跟家人講，這件事我得跟家人講一下。") is True


def test_polish_validation_is_prefix_decidable() -> None:
    """逐字元餵入，驗證器在違規字元出現的那一步即回報，不需要完整字串。"""
    segments = ["撥打 165 反詐騙專線查證。"]
    validator = PolishValidator(segments, EMPTY_VERDICT)
    text = "我覺得有 87 成機率"
    failed_at = None
    for index, character in enumerate(text):
        if not validator.feed(character):
            failed_at = index
            break
    assert failed_at == text.index("8")
    assert failed_at < len(text) - 1


def test_polish_allows_prefix_of_allowed_number() -> None:
    validator = PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("1") is True
    assert validator.feed("6") is True
    assert validator.feed("5") is True
    assert validator.feed("7") is False


def test_allowed_numbers_extracts_every_digit_run() -> None:
    assert allowed_numbers(["撥打 165 查證", "共 27 項"]) == frozenset({"165", "27"})


def test_persona_may_use_a_number_from_the_second_evidence_line() -> None:
    verdict = verdict_with(
        evidence=[
            "第一項依據：「使用者原文」",
            "網域註冊於 6 天前；門檻為 30 天：「另一段使用者原文」",
        ]
    )
    segments = verdict_segments(verdict)
    validator = PolishValidator(segments, verdict)
    assert validator.feed("這個網域才註冊 6 天，我不會照做。") is True
    assert all("使用者原文" not in segment for segment in segments)


def test_persona_still_rejects_a_type_absent_from_the_full_verdict() -> None:
    verdict = verdict_with(evidence=["網域註冊於 6 天前：「原文」"])
    validator = PolishValidator(verdict_segments(verdict), verdict)
    assert validator.feed(f"這是{ScamType.FAKE_AUTHORITY.value}") is False


def test_persona_still_rejects_a_url_with_the_full_verdict_allowlist() -> None:
    verdict = verdict_with(evidence=["網域註冊於 6 天前：「原文」"])
    validator = PolishValidator(verdict_segments(verdict), verdict)
    assert validator.feed("請看 http") is False
