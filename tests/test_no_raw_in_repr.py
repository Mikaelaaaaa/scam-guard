"""原文不得經由 `repr()`、`str()` 或 log 格式化洩漏。

這些測試看起來像在測字串格式，實際上測的是一條**洩漏路徑**：
一行 `logger.info("收到 %s", message)` 就足以把使用者的 LINE 對話寫進
保存數個月的檔案，而寫那行的人不會意識到自己在做這件事。
"""

import logging

import pytest

from scam_guard.check import CheckRegistry
from scam_guard.normalize import Document, NormalizedText, build_document, normalize_text
from scam_guard.pipeline import detect
from scam_guard.redact import RedactedText, redact_document
from scam_guard.types import CheckResult, Message, Request, ScamType, Verdict
from scam_guard.weights import load_weights

SECRET = "獨一無二的原文片段甲乙丙丁"
"""刻意選一個不會出現在任何欄位名、型別名或標點裡的字串。"""


def _message() -> Message:
    return Message(text=f"{SECRET}，請立刻回覆。", sender="them")


def _request() -> Request:
    return Request(messages=[_message()])


def _document() -> Document:
    return build_document([_message()])


def _normalized() -> NormalizedText:
    return normalize_text(f"{SECRET}，請立刻回覆。")


def _verdict() -> Verdict:
    doc = _document()
    return Verdict(
        scam_probability=0.9,
        confidence=0.8,
        scam_type=ScamType.FAKE_AUTHORITY,
        evidence=[SECRET],
        actions=["撥打 165 查證"],
        checks=[CheckResult(name="rule", hit=True, detail="命中")],
        redacted=redact_document(doc),
    )


CARRIERS = {
    "Message": _message,
    "Request": _request,
    "NormalizedText": _normalized,
    "Document": _document,
    "Verdict": _verdict,
}


@pytest.mark.parametrize("name", sorted(CARRIERS))
def test_repr_does_not_contain_raw_text(name: str) -> None:
    assert SECRET not in repr(CARRIERS[name]())


@pytest.mark.parametrize("name", sorted(CARRIERS))
def test_str_does_not_contain_raw_text(name: str) -> None:
    assert SECRET not in f"{CARRIERS[name]()}"


@pytest.mark.parametrize("name", sorted(CARRIERS))
def test_logging_percent_s_does_not_contain_raw_text(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """直接對應真實的洩漏路徑，不只測 `repr()`。"""
    with caplog.at_level(logging.INFO):
        logging.getLogger("test").info("載體 %s", CARRIERS[name]())

    assert SECRET not in caplog.text


def test_message_repr_keeps_length_and_sender() -> None:
    """封鎖內容不等於讓 `repr()` 失去用處 —— debug 仍需要長度與發送者。"""
    message = _message()

    assert repr(message) == f"Message(len={len(message.text)}, sender='them', sent_at=None)"


def test_document_repr_keeps_counts_and_truncation() -> None:
    rendered = repr(_document())

    assert "sentences=1" in rendered
    assert "truncated=False" in rendered
    assert "dropped_messages=0" in rendered


def test_verdict_repr_keeps_evidence_count() -> None:
    rendered = repr(_verdict())

    assert "evidence=1" in rendered
    assert "checks=1" in rendered


def test_invalid_coord_error_contains_no_sentence_text() -> None:
    doc = _document()
    redacted = redact_document(doc)

    with pytest.raises(KeyError) as document_error:
        doc.index_of((9, 9))
    with pytest.raises(KeyError) as projection_error:
        redacted.index_of((9, 9))

    assert "(9, 9)" in str(document_error.value)
    assert SECRET not in str(document_error.value)
    assert "(9, 9)" in str(projection_error.value)
    assert SECRET not in str(projection_error.value)


def test_full_detect_verdict_does_not_leak_an_id_number() -> None:
    req = Request(messages=[Message(text="我的身分證是A123456789，請幫我處理。")])

    verdict = detect(req, CheckRegistry(), load_weights())

    assert "A123456789" not in repr(verdict)
    assert "A123456789" not in f"{verdict}"


def test_projection_is_the_one_type_that_may_print_its_text() -> None:
    """`RedactedText` 是唯一的例外 —— 沒有它，debug 時沒有任何可用的輸出。"""
    redacted = RedactedText(sentences=["<TW_MOBILE>"], coords=[(0, 0)], counts={})

    assert "<TW_MOBILE>" in repr(redacted)
