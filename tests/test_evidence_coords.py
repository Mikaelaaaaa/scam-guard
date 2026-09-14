"""證據座標與 `Document` 的往返：檢查回報座標，呈現層據此取回原文。"""

import pytest

from scam_guard.normalize import Limits, build_document
from scam_guard.types import CheckResult, Message


def a_conversation() -> list[Message]:
    return [
        Message(text="在嗎？我是你朋友"),
        Message(text="這檔標的這週進場。明天匯款到這個帳戶"),
    ]


def test_valid_coordinate_resolves_to_the_sentence_the_check_saw() -> None:
    doc = build_document(a_conversation())
    result = CheckResult(
        name="solicit_transfer",
        hit=True,
        weight=1.5,
        detail="要求匯款",
        evidence=[(1, 1)],
    )

    coord = result.evidence[0]

    assert doc.text_at(coord) == "明天匯款到這個帳戶"
    assert doc.raw_at(coord) == "明天匯款到這個帳戶"


def test_invalid_coordinate_raises_instead_of_returning_nothing() -> None:
    doc = build_document(a_conversation())
    result = CheckResult(name="broken", hit=True, weight=1.0, detail="算錯位置", evidence=[(9, 9)])

    with pytest.raises(KeyError, match="座標不存在"):
        doc.index_of(result.evidence[0])


def test_coordinate_round_trip_matches_the_normalized_content() -> None:
    messages = [Message(text="您的驗證碼是 １２３４５６。請勿告訴他人")]
    doc = build_document(messages)
    seen = doc.sentences[0]
    coord = doc.coords[0]

    assert doc.text_at(coord) == seen
    assert doc.raw_at(coord) == "您的驗證碼是 １２３４５６。"


def test_message_index_survives_truncation() -> None:
    messages = [Message(text=f"第 {i} 則") for i in range(6)]

    doc = build_document(messages, Limits(max_messages=3))
    result = CheckResult(
        name="solicit_otp",
        hit=True,
        weight=1.5,
        detail="索取驗證碼",
        evidence=[doc.coords[0]],
    )

    assert doc.dropped_messages == 3
    assert result.evidence == [(3, 0)]
    assert doc.text_at(result.evidence[0]) == "第 3 則"
