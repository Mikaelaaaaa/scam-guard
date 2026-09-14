"""`Document` 的欄位、座標語意與建構不變式。"""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from scam_guard.normalize import Document, build_document
from scam_guard.types import Message, Request

ZWSP = "​"


def test_all_messages_enter_the_document() -> None:
    doc = build_document(
        [
            Message(text="第一句。第二句"),
            Message(text="第三句。第四句"),
            Message(text="第五句。第六句"),
        ]
    )

    assert list(doc.sentences) == [
        "第一句。",
        "第二句",
        "第三句。",
        "第四句",
        "第五句。",
        "第六句",
    ]


def test_single_message_behaves_like_many() -> None:
    doc = build_document([Message(text="第一句。第二句")])

    assert list(doc.coords) == [(0, 0), (0, 1)]


def test_sentence_index_restarts_within_each_message() -> None:
    doc = build_document([Message(text="一。二"), Message(text="三。四")])

    assert list(doc.coords) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_raw_sentences_match_in_length_and_keep_zero_width_characters() -> None:
    doc = build_document([Message(text=f"驗{ZWSP}證碼。請勿外流")])

    assert len(doc.sentences) == len(doc.raw_sentences)
    assert doc.sentences[0] == "驗證碼。"
    assert doc.raw_sentences[0] == f"驗{ZWSP}證碼。"


def test_raw_sentences_keep_fullwidth_digits() -> None:
    doc = build_document([Message(text="金額１２３萬")])

    assert doc.sentences[0] == "金額123萬"
    assert doc.raw_sentences[0] == "金額１２３萬"


def test_coordinate_lookups() -> None:
    doc = build_document([Message(text="一。二"), Message(text="三。四")])

    assert doc.index_of((1, 0)) == 2
    assert doc.text_at((1, 1)) == "四"
    assert doc.raw_at((0, 0)) == "一。"


def test_invalid_coordinate_raises() -> None:
    doc = build_document([Message(text="一。二")])

    with pytest.raises(KeyError, match="座標不存在"):
        doc.index_of((5, 0))


def test_message_range() -> None:
    doc = build_document([Message(text="一。二"), Message(text="   "), Message(text="三。四。五")])

    assert doc.message_range(0) == range(0, 2)
    assert doc.message_range(1) == range(0)
    assert doc.message_range(2) == range(2, 5)
    assert [doc.sentences[i] for i in doc.message_range(2)] == ["三。", "四。", "五"]


def test_message_without_sentences_does_not_shift_later_indices() -> None:
    doc = build_document([Message(text="一"), Message(text=""), Message(text="三")])

    assert list(doc.coords) == [(0, 0), (2, 0)]


def test_truncation_fields_default_to_untruncated() -> None:
    doc = build_document([Message(text="一。二")])

    assert doc.truncated is False
    assert doc.dropped_messages == 0


def test_document_is_immutable() -> None:
    doc = build_document([Message(text="一。二")])

    with pytest.raises(FrozenInstanceError):
        doc.sentences = ("改寫",)  # type: ignore[misc]

    with pytest.raises(AttributeError):
        doc.sentences.append("追加")  # type: ignore[attr-defined]


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="等長"):
        Document(sentences=["一", "二"], raw_sentences=["一"], coords=[(0, 0), (0, 1)])


def test_coord_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="座標數"):
        Document(sentences=["一", "二"], raw_sentences=["一", "二"], coords=[(0, 0)])


def test_non_increasing_coords_raise() -> None:
    with pytest.raises(ValueError, match="訊息序號必須遞增"):
        Document(sentences=["一", "二"], raw_sentences=["一", "二"], coords=[(1, 0), (0, 0)])

    with pytest.raises(ValueError, match="必須連續"):
        Document(sentences=["一", "二"], raw_sentences=["一", "二"], coords=[(0, 0), (0, 2)])

    with pytest.raises(ValueError, match="自 0 起算"):
        Document(sentences=["一", "二"], raw_sentences=["一", "二"], coords=[(0, 0), (1, 1)])


def test_empty_document_is_legal() -> None:
    doc = build_document([Message(text="   "), Message(text=ZWSP)])

    assert list(doc.sentences) == []
    assert list(doc.raw_sentences) == []
    assert list(doc.coords) == []


def test_coordinate_points_back_at_the_original_message() -> None:
    sent_at = datetime(2026, 9, 14, 10, 0, 0)
    req = Request(
        messages=[
            Message(text="在嗎？"),
            Message(text="這檔標的這週進場。明天匯款", sender="them", sent_at=sent_at),
        ]
    )
    doc = build_document(req.messages)
    coord = doc.coords[doc.index_of((1, 1))]

    assert req.messages[coord[0]].sent_at == sent_at
    assert doc.text_at(coord) == "明天匯款"


def test_negative_coordinate_components_are_rejected() -> None:
    """訊息序號 -1 會讓 `req.messages[-1]` 安靜指向最後一則訊息。"""
    with pytest.raises(ValueError, match="不可為負"):
        Document(
            sentences=("一", "二"),
            raw_sentences=("一", "二"),
            coords=((-1, 0), (-1, 1)),
        )


def test_negative_dropped_messages_is_rejected() -> None:
    with pytest.raises(ValueError, match="丟棄則數不可為負"):
        Document(sentences=(), raw_sentences=(), coords=(), dropped_messages=-3)


def test_message_range_rejects_negative_index() -> None:
    doc = Document(sentences=("一",), raw_sentences=("一",), coords=((0, 0),))

    with pytest.raises(ValueError, match="訊息序號不可為負"):
        doc.message_range(-1)
