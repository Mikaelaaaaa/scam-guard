"""切句 `split_sentences()` 的行為與句子編號的穩定性。"""

from scam_guard.normalize import normalize_text, split_sentences

ZWSP = "​"


def split(raw: str) -> list[tuple[str, str, tuple[int, ...]]]:
    return split_sentences(normalize_text(raw))


def texts(raw: str) -> list[str]:
    return [text for text, _, _ in split(raw)]


def test_period_splits() -> None:
    assert texts("我是警察。請配合調查") == ["我是警察。", "請配合調查"]


def test_exclamation_question_and_semicolon_split() -> None:
    assert texts("快點！") == ["快點!"]
    assert texts("在嗎？你好") == ["在嗎?", "你好"]
    assert texts("第一項；第二項") == ["第一項;", "第二項"]


def test_newline_splits() -> None:
    assert texts("第一行\n第二行") == ["第一行", "第二行"]


def test_comma_and_enumeration_comma_do_not_split() -> None:
    sentences = texts("股票、基金，都可以買")

    assert sentences == ["股票、基金,都可以買"]
    assert "、" in sentences[0]
    assert "," in sentences[0]


def test_ascii_period_does_not_split() -> None:
    assert texts("請到 reurl.cc/abc 領取") == ["請到 reurl.cc/abc 領取"]
    assert texts("匯款 38.5 萬") == ["匯款 38.5 萬"]


def test_sentence_end_punctuation_is_kept_and_newline_is_not() -> None:
    sentences = texts("在嗎？\n我是你朋友")

    assert sentences == ["在嗎?", "我是你朋友"]
    assert all("\n" not in s for s in sentences)


def test_repeated_separators_blank_lines_and_trailing_separator_make_no_empty_sentence() -> None:
    assert texts("快點！！！") == ["快點!!!"]
    assert texts("第一段\n\n第二段") == ["第一段", "第二段"]
    assert texts("結束了。") == ["結束了。"]


def test_whitespace_only_input_returns_no_sentences() -> None:
    assert split("   \n\n  ") == []


def test_input_without_trailing_separator_becomes_one_sentence() -> None:
    assert texts("請盡快回覆") == ["請盡快回覆"]


def test_leading_and_trailing_spaces_are_stripped_but_inner_spaces_are_not() -> None:
    assert texts("  驗 證 碼   是多少？") == ["驗 證 碼   是多少?"]


def test_normalized_and_raw_sentences_have_the_same_length() -> None:
    sentences = split("第一句。第二句！第三句？尾段")

    assert len(sentences) == 4
    assert all(len(item) == 3 for item in sentences)
    assert all(len(offsets) == len(text) + 1 for text, _, offsets in sentences)


def test_raw_sentence_keeps_zero_width_characters() -> None:
    text, raw, _ = split(f"驗{ZWSP}證碼是多少？")[0]

    assert text == "驗證碼是多少?"
    assert raw == f"驗{ZWSP}證碼是多少？"


def test_raw_sentence_keeps_the_original_punctuation_writing() -> None:
    text, raw, _ = split("在嗎｡")[0]

    assert text == "在嗎。"
    assert raw == "在嗎｡"


def test_url_with_query_string_is_not_split() -> None:
    sentences = texts("請點 https://x.cc/a?id=1;v=2 領取。謝謝")

    assert sentences == ["請點 https://x.cc/a?id=1;v=2 領取。", "謝謝"]


def test_repeated_calls_give_the_same_result() -> None:
    raw = "在嗎？我是你朋友。請匯款 https://x.cc/a?id=1;v=2"

    assert split(raw) == split(raw)


def test_otp_message_stays_one_sentence_with_its_comma() -> None:
    """規則層的子句切分以此為前提：逗號留在句內，句子不因它而分裂。"""
    sentences = texts("您的驗證碼是 123456，請勿告訴他人")

    assert sentences == ["您的驗證碼是 123456,請勿告訴他人"]
    assert "," in sentences[0]


def test_raw_fragment_keeps_boundary_whitespace_and_zero_width() -> None:
    """句子邊界上的空白與零寬字元是規避痕跡，原文片段不得去除。"""
    sentences = split(f"你好。 {ZWSP}驗證碼")

    assert [text for text, _, _ in sentences] == ["你好。", "驗證碼"]
    assert [raw for _, raw, _ in sentences] == ["你好。", f" {ZWSP}驗證碼"]
