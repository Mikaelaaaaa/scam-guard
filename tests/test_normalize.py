"""字元層正規化 `normalize_text()` 與 `NormalizedText` 的行為。"""

import random

import pytest

from scam_guard.normalize import NormalizedText, normalize_text

ZWSP = "​"
RLO = "‮"
SOFT_HYPHEN = "­"
IDEOGRAPHIC_SPACE = "　"
NBSP = " "


def test_fullwidth_alnum_and_punct_become_ascii() -> None:
    result = normalize_text("ＡＢＣ１２３！？；，：")

    assert result.text == "ABC123!?;,:"


def test_chinese_period_and_enumeration_comma_survive() -> None:
    result = normalize_text("股票、基金。")

    assert result.text == "股票、基金。"


def test_halfwidth_ideographic_period_converges() -> None:
    """半形 `｡` 與全形 `。` 是同一個標點的兩種寫法，MUST 收斂為同一字元。"""
    assert normalize_text("在嗎｡").text == normalize_text("在嗎。").text


def test_invisible_characters_are_removed() -> None:
    result = normalize_text(f"驗{ZWSP}證{RLO}碼{SOFT_HYPHEN}")

    assert result.text == "驗證碼"


def test_input_of_only_invisible_characters_yields_empty_text() -> None:
    result = normalize_text(ZWSP * 3 + RLO)

    assert result.text == ""
    assert len(result.offsets) == 1


def test_url_and_decimal_periods_are_preserved() -> None:
    result = normalize_text("請點 https://reurl.cc/abc 匯 38.5 萬")

    assert "https://reurl.cc/abc" in result.text
    assert "38.5" in result.text
    assert "。" not in result.text


def test_enumeration_comma_and_comma_remain_distinct() -> None:
    result = normalize_text("股票、基金，請把握")

    assert "、" in result.text
    assert "," in result.text


def test_whitespace_is_unified_but_not_collapsed() -> None:
    result = normalize_text(f"驗{IDEOGRAPHIC_SPACE}證{NBSP}碼   了\n下一行")

    assert result.text == "驗 證 碼   了\n下一行"


def test_newline_writings_converge() -> None:
    assert normalize_text("一\r\n二").text == "一\n二"
    assert normalize_text("一\r二").text == "一\n二"


def test_span_maps_back_to_raw_and_keeps_removed_characters() -> None:
    raw = f"驗{ZWSP}證碼。請勿外流"
    result = normalize_text(raw)
    end = result.text.index("。") + 1

    assert result.text[:end] == "驗證碼。"
    assert result.raw_span(0, end) == f"驗{ZWSP}證碼。"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "請盡快回覆",
        f"{ZWSP}開頭就是零寬字元",
        "ＡＢＣ１２３！？",
        "一\r\n二\r三",
        f"㈱{IDEOGRAPHIC_SPACE}測試",
        "https://reurl.cc/abc?id=1;v=2",
    ],
)
def test_full_span_maps_back_to_the_whole_raw(raw: str) -> None:
    result = normalize_text(raw)

    assert result.raw_span(0, len(result.text)) == raw


def test_simplified_and_mixed_chinese_are_untouched() -> None:
    result = normalize_text("请汇款到账户，帳戶已開通")

    assert "请汇款到账户" in result.text
    assert "帳戶" in result.text
    assert "账户" in result.text


def test_uppercase_is_preserved() -> None:
    result = normalize_text("URGENT WINNER")

    assert result.text == "URGENT WINNER"


@pytest.mark.parametrize(
    "raw",
    [
        "ＡＢＣ１２３！？；",
        f"驗{ZWSP}證碼{IDEOGRAPHIC_SPACE}請勿外流。",
        "㈱台鋼① 38.5 萬",
        "一\r\n二",
        "",
    ],
)
def test_normalization_is_idempotent(raw: str) -> None:
    once = normalize_text(raw).text

    assert normalize_text(once).text == once


def test_offsets_invariants_hold_for_random_inputs() -> None:
    alphabet = list("驗證碼。，、!?;abcAB1１２ＡＢ.\n\r\t ") + [
        ZWSP,
        RLO,
        SOFT_HYPHEN,
        IDEOGRAPHIC_SPACE,
        NBSP,
        "㈱",
        "①",
    ]
    rng = random.Random(20260914)

    for _ in range(300):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        result = normalize_text(raw)

        assert len(result.offsets) == len(result.text) + 1
        assert all(
            result.offsets[i] >= result.offsets[i - 1] for i in range(1, len(result.offsets))
        )
        assert result.offsets[-1] == len(raw)
        if result.text:
            assert result.raw_span(0, len(result.text)) == raw


def test_broken_offsets_length_raises() -> None:
    with pytest.raises(ValueError, match="文字長度加一"):
        NormalizedText(raw="ab", text="ab", offsets=(0, 1))


def test_non_monotonic_offsets_raise() -> None:
    with pytest.raises(ValueError, match="單調不減"):
        NormalizedText(raw="ab", text="ab", offsets=(0, 1, 0))


def test_offsets_not_starting_at_zero_raise() -> None:
    with pytest.raises(ValueError, match="首元素"):
        NormalizedText(raw="ab", text="ab", offsets=(1, 1, 2))


def test_offsets_without_sentinel_raise() -> None:
    with pytest.raises(ValueError, match="哨兵"):
        NormalizedText(raw="ab", text="ab", offsets=(0, 1, 1))
