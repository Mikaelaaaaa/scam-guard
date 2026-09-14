"""輸入契約 `Message` / `Request` 的行為。"""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from scam_guard.types import Message, Request


def test_message_with_text_only() -> None:
    msg = Message(text="您的包裹待領取")

    assert msg.text == "您的包裹待領取"
    assert msg.sender is None
    assert msg.sent_at is None


def test_message_with_all_fields() -> None:
    at = datetime(2026, 9, 14, 10, 0, 0)
    msg = Message(text="您的包裹待領取", sender="them", sent_at=at)

    assert msg.text == "您的包裹待領取"
    assert msg.sender == "them"
    assert msg.sent_at == at


def test_latest_and_context_with_three_messages() -> None:
    first = Message(text="第一則")
    second = Message(text="第二則")
    third = Message(text="第三則")
    req = Request(messages=[first, second, third])

    assert req.latest is third
    assert req.context == [first, second]


def test_context_is_empty_for_single_message() -> None:
    only = Message(text="唯一一則")
    req = Request(messages=[only])

    assert req.latest is only
    assert req.context == []


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValueError, match="至少需要一則訊息"):
        Request(messages=[])


def test_from_text_builds_single_message_request() -> None:
    req = Request.from_text("您的包裹待領取")

    assert len(req.messages) == 1
    assert req.latest.text == "您的包裹待領取"


def test_message_is_immutable() -> None:
    msg = Message(text="您的包裹待領取")

    with pytest.raises(FrozenInstanceError):
        msg.text = "改寫"  # type: ignore[misc]
