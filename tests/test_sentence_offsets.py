"""句內偏移對映：不變式、跨越不可逆正規化的映回，以及越界的失敗方式。

這些測試釘住的是一件事 —— **句內的字元位置不得由字串搜尋還原**。
NFKC 既有一對多（`㈱` → `(株)`）也有多對一（`\\r\\n` → `\\n`），
零寬字元與全形寫法讓 `raw.find(片段)` 必然找不到或找錯；
而找錯沒有任何機制會報告，結果是遮蔽漏字、底線框錯位置。
"""

import pytest

from scam_guard.normalize import Document, Limits, build_document
from scam_guard.types import Message

ZWSP = "​"


def trivial_offsets(*sentences: str) -> tuple[tuple[int, ...], ...]:
    """每句一份一對一的句內偏移，供只關心其他不變式的測試使用。"""
    return tuple(tuple(range(len(sentence) + 1)) for sentence in sentences)


def test_four_sequences_have_the_same_length() -> None:
    doc = build_document([Message(text="第一句。第二句"), Message(text="第三句")])

    assert len(doc.sentence_offsets) == len(doc.sentences)
    assert len(doc.sentence_offsets) == len(doc.raw_sentences)
    assert len(doc.sentence_offsets) == len(doc.coords)


def test_offset_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="句內偏移數必須等於句子數"):
        Document(
            sentences=["一", "二"],
            raw_sentences=["一", "二"],
            coords=[(0, 0), (0, 1)],
            sentence_offsets=trivial_offsets("一"),
        )


def test_per_sentence_offset_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="第 1 句的句內偏移長度必須為句長加一"):
        Document(
            sentences=["一", "二三"],
            raw_sentences=["一", "二三"],
            coords=[(0, 0), (0, 1)],
            sentence_offsets=((0, 1), (0, 2)),
        )


def test_non_monotonic_offsets_raise() -> None:
    with pytest.raises(ValueError, match="第 0 句的句內偏移必須單調不減"):
        Document(
            sentences=["一二三"],
            raw_sentences=["一二三"],
            coords=[(0, 0)],
            sentence_offsets=((0, 2, 1, 3),),
        )


def test_sentinel_not_equal_to_raw_fragment_length_raises() -> None:
    with pytest.raises(ValueError, match="第 0 句的句內偏移哨兵必須為原文片段長度"):
        Document(
            sentences=["一二"],
            raw_sentences=[f"一{ZWSP}二"],
            coords=[(0, 0)],
            sentence_offsets=((0, 1, 2),),
        )


def test_first_offset_must_be_zero() -> None:
    with pytest.raises(ValueError, match="第 0 句的句內偏移首項必須為 0"):
        Document(
            sentences=["一二"],
            raw_sentences=["一二"],
            coords=[(0, 0)],
            sentence_offsets=((1, 1, 2),),
        )


@pytest.mark.parametrize(
    "raw",
    [
        "你好。 驗證碼是多少？",
        f"你好。 {ZWSP}驗證碼",
        "  驗 證 碼   是多少？尾段",
        "請點 https://x.cc/a?id=1;v=2 領取。謝謝",
        "第一行\n第二行\n\n第三行",
    ],
)
def test_whole_sentence_span_equals_the_raw_fragment(raw: str) -> None:
    """兩端都吸收：不能讓同一句有兩份原文。"""
    doc = build_document([Message(text=raw)])

    assert doc.sentences
    for coord in doc.coords:
        assert doc.raw_span_at(coord, 0, len(doc.text_at(coord))) == doc.raw_at(coord)


def test_zero_width_character_inside_an_id_number() -> None:
    """design 的反例：天真的 `raw_start + a` 會少掉最後一碼，而且不拋例外。"""
    raw = f"我的身分證是A1234{ZWSP}56789"
    doc = build_document([Message(text=raw)])
    coord = (0, 0)

    assert doc.text_at(coord) == "我的身分證是A123456789"
    assert doc.raw_at(coord) == raw

    assert doc.raw_span_at(coord, 6, 16) == f"A1234{ZWSP}56789"
    assert doc.raw_span_at(coord, 6, 16).endswith("9")
    assert ZWSP in doc.raw_span_at(coord, 6, 16)

    # 天真算法：句子在原文中的起始索引加上句內位置。這一句正是整則訊息，
    # 起始索引為 0，於是它退化成 `raw[6:16]` —— 少一個字元，安靜地錯。
    naive = raw[0 + 6 : 0 + 16]
    assert naive == f"A1234{ZWSP}5678"
    assert len(naive) == len(doc.raw_span_at(coord, 6, 16)) - 1
    assert not naive.endswith("9")


def test_fullwidth_digits_map_back_to_the_fullwidth_writing() -> None:
    doc = build_document([Message(text="我的身分證是Ａ１２３４５６７８９")])
    coord = (0, 0)

    assert doc.text_at(coord) == "我的身分證是A123456789"
    assert doc.raw_span_at(coord, 6, 16) == "Ａ１２３４５６７８９"


def test_crlf_collapses_without_misaligning_either_side() -> None:
    """`\\r\\n` 是唯一的多對一映射，被吃掉的字元不歸屬於任何一句。"""
    raw = "第一句。\r\n第二句"
    doc = build_document([Message(text=raw)])

    assert list(doc.sentences) == ["第一句。", "第二句"]
    assert list(doc.raw_sentences) == ["第一句。", "第二句"]
    assert doc.raw_span_at((0, 0), 0, 4) == "第一句。"
    assert doc.raw_span_at((0, 1), 0, 3) == "第二句"
    assert "\r" not in doc.raw_at((0, 0))
    assert "\r" not in doc.raw_at((0, 1))
    assert doc.raw_at((0, 0)) + "\r\n" + doc.raw_at((0, 1)) == raw


def test_bounds_and_span_agree() -> None:
    doc = build_document([Message(text=f"驗{ZWSP}證碼是 123456。請勿外流")])

    for coord in doc.coords:
        length = len(doc.text_at(coord))
        for start in range(length + 1):
            for end in range(start, length + 1):
                a, b = doc.raw_bounds_at(coord, start, end)
                assert 0 <= a <= b <= len(doc.raw_at(coord))
                assert doc.raw_at(coord)[a:b] == doc.raw_span_at(coord, start, end)


def test_out_of_range_intervals_raise_and_never_clamp() -> None:
    doc = build_document([Message(text="驗證碼是多少")])
    coord = (0, 0)
    length = len(doc.text_at(coord))

    with pytest.raises(ValueError, match="句內起始不可為負") as negative:
        doc.raw_span_at(coord, -1, 3)
    assert "start=-1" in str(negative.value)

    with pytest.raises(ValueError, match="句內結束不可超過句長") as beyond:
        doc.raw_span_at(coord, 0, length + 1)
    assert f"end={length + 1}" in str(beyond.value)
    assert f"句長={length}" in str(beyond.value)
    assert str(coord) in str(beyond.value)

    with pytest.raises(ValueError, match="句內起始不可大於結束") as inverted:
        doc.raw_span_at(coord, 4, 2)
    assert "start=4" in str(inverted.value)
    assert "end=2" in str(inverted.value)

    with pytest.raises(KeyError, match="座標不存在"):
        doc.raw_span_at((9, 9), 0, 1)

    # 越界 MUST NOT 被截斷至合法範圍：`end = 句長 + 1` 不得回傳任何字串。
    with pytest.raises(ValueError):
        doc.raw_bounds_at(coord, 0, length + 1)


def test_empty_interval_is_legal() -> None:
    doc = build_document([Message(text="驗證碼是多少")])
    coord = (0, 0)

    assert doc.raw_span_at(coord, 3, 3) == ""
    a, b = doc.raw_bounds_at(coord, 3, 3)
    assert a == b


def test_offsets_are_rebased_per_sentence_across_messages() -> None:
    doc = build_document(
        [
            Message(text="第一句。第二句"),
            Message(text=f"帳號是0912{ZWSP}345678。請盡快匯款"),
            Message(text="第五句。第六句"),
        ]
    )
    coord = (1, 1)

    assert doc.text_at(coord) == "請盡快匯款"
    # 基準是這一句自己的原文片段，不是整則訊息、也不是整份請求。
    assert doc.sentence_offsets[doc.index_of(coord)] == (0, 1, 2, 3, 4, 5)
    assert doc.raw_span_at(coord, 0, 5) == "請盡快匯款"
    assert doc.raw_span_at((1, 0), 3, 13) == f"0912{ZWSP}345678"


def test_truncation_does_not_shift_any_offset() -> None:
    messages = [Message(text=f"第{n}則。{ZWSP}內容{n}") for n in range(6)]
    full = build_document(messages)
    truncated = build_document(messages, Limits(max_messages=3))

    assert truncated.truncated is True
    assert truncated.dropped_messages == 3
    assert len(truncated.sentence_offsets) == len(truncated.sentences)

    for coord in truncated.coords:
        assert (
            truncated.sentence_offsets[truncated.index_of(coord)]
            == full.sentence_offsets[full.index_of(coord)]
        )
    assert all(coord[0] >= 3 for coord in truncated.coords)


def test_repeated_substring_is_located_by_offsets_not_by_search() -> None:
    raw = f"帳號123，帳號1{ZWSP}23"
    doc = build_document([Message(text=raw)])
    coord = (0, 0)

    assert doc.text_at(coord) == "帳號123,帳號123"
    assert doc.raw_bounds_at(coord, 2, 5) == (2, 5)
    assert doc.raw_bounds_at(coord, 8, 11) == (8, 12)
    assert doc.raw_span_at(coord, 2, 5) == "123"
    assert doc.raw_span_at(coord, 8, 11) == f"1{ZWSP}23"
    # 搜尋只會指到第一個相符處，第二段因插入零寬字元根本搜不到。
    assert raw.find("123") == 2
    assert raw.find("123", 6) == -1


def test_empty_document_is_legal() -> None:
    doc = build_document([Message(text="   "), Message(text=ZWSP)])

    assert list(doc.sentence_offsets) == []


def test_offsets_are_nested_tuples_and_immutable() -> None:
    doc = build_document([Message(text="第一句。第二句")])

    assert isinstance(doc.sentence_offsets, tuple)
    assert all(isinstance(offsets, tuple) for offsets in doc.sentence_offsets)

    with pytest.raises(AttributeError):
        doc.sentence_offsets[0].append(99)  # type: ignore[attr-defined]


def test_repr_reports_offset_size_without_printing_the_integers() -> None:
    """預設 repr 會印出數萬個整數；自訂 repr 只留量級。"""
    doc = build_document([Message(text="第一句。第二句")])
    rendered = repr(doc)

    assert "offset_entries=9" in rendered
    assert "第一句" not in rendered
