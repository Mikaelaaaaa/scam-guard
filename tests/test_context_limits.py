"""則數與字元數雙上限、丟棄策略與截斷痕跡。"""

import random

import pytest

from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document, normalize_text
from scam_guard.types import Message

ZWSP = "​"


def messages_of(count: int, text: str) -> list[Message]:
    return [Message(text=f"{i} {text}") for i in range(count)]


def kept_indices(doc: Document) -> list[int]:
    seen: list[int] = []
    for message_index, _ in doc.coords:
        if message_index not in seen:
            seen.append(message_index)
    return seen


def kept_chars(messages: list[Message], doc: Document) -> int:
    return sum(len(normalize_text(m.text).text) for m in messages[doc.dropped_messages :])


def test_message_limit_keeps_the_newest_hundred() -> None:
    messages = messages_of(150, "短訊息")

    doc = build_document(messages)

    assert doc.dropped_messages == 50
    assert len(kept_indices(doc)) == 100
    assert kept_indices(doc)[0] == 50


def test_char_limit_applies_to_long_messages() -> None:
    messages = messages_of(20, "長" * 4000)

    doc = build_document(messages)

    assert kept_chars(messages, doc) <= DEFAULT_LIMITS.max_chars
    assert doc.truncated is True


def test_nothing_is_dropped_when_both_limits_are_respected() -> None:
    messages = messages_of(10, "短訊息")

    doc = build_document(messages)

    assert doc.truncated is False
    assert doc.dropped_messages == 0
    assert len(kept_indices(doc)) == 10


def test_both_limits_exceeded_are_both_satisfied() -> None:
    messages = messages_of(150, "長" * 1000)

    doc = build_document(messages)

    assert len(kept_indices(doc)) <= DEFAULT_LIMITS.max_messages
    assert kept_chars(messages, doc) <= DEFAULT_LIMITS.max_chars


def test_custom_limits_take_effect() -> None:
    messages = messages_of(10, "短訊息")

    by_count = build_document(messages, Limits(max_messages=5))
    by_chars = build_document(messages, Limits(max_chars=30))

    assert len(kept_indices(by_count)) == 5
    assert kept_chars(messages, by_chars) <= 30


def test_illegal_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_messages"):
        Limits(max_messages=0)

    with pytest.raises(ValueError, match="max_chars"):
        Limits(max_chars=-1)


def test_oldest_messages_are_dropped_and_order_is_kept() -> None:
    messages = messages_of(10, "訊息")

    doc = build_document(messages, Limits(max_messages=3))

    assert kept_indices(doc) == [7, 8, 9]


def test_kept_messages_are_not_partially_truncated() -> None:
    messages = messages_of(10, "第一句。第二句。第三句")

    full = build_document(messages)
    limited = build_document(messages, Limits(max_messages=3))

    for message_index in kept_indices(limited):
        assert [limited.sentences[i] for i in limited.message_range(message_index)] == [
            full.sentences[i] for i in full.message_range(message_index)
        ]


def test_latest_message_is_kept_whole_even_when_it_alone_exceeds_the_char_limit() -> None:
    messages = [Message(text="前文"), Message(text="長" * 500)]

    doc = build_document(messages, Limits(max_chars=100))

    assert doc.dropped_messages == 1
    assert kept_indices(doc) == [1]
    assert doc.text_at((1, 0)) == "長" * 500


def test_message_limit_of_one_keeps_only_the_latest() -> None:
    messages = messages_of(5, "訊息")

    doc = build_document(messages, Limits(max_messages=1))

    assert kept_indices(doc) == [4]


def test_invisible_characters_do_not_consume_the_char_budget() -> None:
    padded = [Message(text="前文一"), Message(text="前文二"), Message(text=ZWSP * 500 + "最新")]

    doc = build_document(padded, Limits(max_chars=12))

    assert doc.truncated is False
    assert kept_indices(doc) == [0, 1, 2]


def test_truncation_marks_are_recorded() -> None:
    messages = messages_of(40, "訊息")

    doc = build_document(messages, Limits(max_messages=3))

    assert doc.truncated is True
    assert doc.dropped_messages == 37


def test_message_index_does_not_shift_after_dropping() -> None:
    messages = messages_of(6, "訊息")

    doc = build_document(messages, Limits(max_messages=3))

    assert doc.coords[0][0] == 3


def test_limits_hold_for_random_message_sequences() -> None:
    rng = random.Random(20260914)

    for _ in range(200):
        count = rng.randint(1, 30)
        messages = [Message(text="字" * rng.randint(0, 60)) for _ in range(count)]
        limits = Limits(max_messages=rng.randint(1, 10), max_chars=rng.randint(1, 300))

        doc = build_document(messages, limits)

        kept = len(messages) - doc.dropped_messages
        assert kept >= 1
        assert kept <= limits.max_messages
        assert doc.truncated == (doc.dropped_messages > 0)
        # 最後一則永遠保留，因此只有「僅剩最後一則」時才允許超出字元上限。
        assert kept_chars(messages, doc) <= limits.max_chars or kept == 1
