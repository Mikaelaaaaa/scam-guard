"""個資辨識器 —— 封閉集合的四個項目與它們的誤判抑制條件。

驗收標準刻意**不寫成 recall**：四個項目的覆蓋率本來就低，那是設計選擇。
寫成「`實驗結果.md` 的四個誤判樣本不命中」＋「已知正樣本命中」。
"""

import pytest

from scam_guard.normalize import build_document
from scam_guard.pii import PiiSpan, _luhn_ok, find_pii
from scam_guard.types import Message


def _types(sentence: str) -> list[str]:
    return [span.entity_type for span in find_pii(sentence)]


def _texts(sentence: str) -> list[str]:
    return [sentence[span.start : span.end] for span in find_pii(sentence)]


def test_valid_tw_id_is_recognised() -> None:
    spans = find_pii("我的身分證是A123456789，請查收。")

    assert [(s.entity_type, s.start, s.end) for s in spans] == [("TW_ID", 6, 16)]


def test_tw_id_checksum_rejects_every_tampered_last_digit() -> None:
    """`A12345678` 的十個末位候選中恰好只有一個通過校驗碼。

    同時驗證 checksum 沒有寫成「全部回傳 False」—— 那種錯誤在低密度語料上
    看起來與「本來就沒有身分證」一模一樣。
    """
    accepted = [digit for digit in "0123456789" if find_pii(f"A12345678{digit}")]

    assert accepted == ["9"]


def test_tw_id_second_character_must_be_one_or_two() -> None:
    assert find_pii("A323456789") == []


def test_fullwidth_tw_id_is_recognised_after_normalisation() -> None:
    """辨識在正規化後的文字上進行，所以全形書寫不需要第二套樣式。"""
    doc = build_document([Message(text="身分證Ａ１２３４５６７８９")])

    assert len(doc.sentences) == 1
    assert _texts(doc.sentences[0]) == ["A123456789"]


def test_mobile_bare_and_separated_forms() -> None:
    assert _types("0912345678") == ["TW_MOBILE"]
    assert _texts("0912-345-678") == ["0912-345-678"]


def test_mobile_requires_digit_boundary() -> None:
    """更長的連續數字中的十位子字串不是手機號碼 —— 那是單號的形狀。"""
    assert find_pii("1230912345678999") == []


def test_landline_requires_separator_or_parentheses() -> None:
    assert _types("(02)2720-8889") == ["TW_LANDLINE"]
    assert _types("04-2222-3333") == ["TW_LANDLINE"]
    assert find_pii("0227208889") == []


def test_credit_card_requires_grouping_or_known_iin() -> None:
    assert _types("4111-1111-1111-1111") == ["CREDIT_CARD"]
    assert _types("4111111111111111") == ["CREDIT_CARD"]
    # 通過 Luhn、十六位、無分組，但 `80` 不是任何發卡組織的 IIN 前綴。
    assert _luhn_ok("8011111111111113")
    assert find_pii("8011111111111113") == []


def test_credit_card_requires_luhn() -> None:
    assert find_pii("4111-1111-1111-1112") == []


@pytest.mark.parametrize(
    "sentence",
    [
        "訂單802045734652",
        "統編93538651",
        "寄件碼 E73474605986",
        "小桃代收帳號 081/ 001620814031",
    ],
)
def test_measured_false_positives_do_not_match(sentence: str) -> None:
    """`實驗結果.md` 的四個實測誤判字串，直接作為迴歸案例。"""
    assert find_pii(sentence) == []


@pytest.mark.parametrize(
    "sentence",
    ["您的驗證碼是 123456，請勿告訴他人", "認證碼 8848", "動態密碼 12345678"],
)
def test_one_time_passwords_do_not_match(sentence: str) -> None:
    """驗證碼是 `add-speech-act-rules` 最重要的 Tier-A 訊號，遮掉它等於毀掉偵測力。

    四個項目沒有任何一項會命中典型驗證碼，這是結構性條件的副產品，
    不是一條上下文例外規則。
    """
    assert find_pii(sentence) == []


def test_url_interior_is_not_recognised() -> None:
    assert find_pii("https://example.com/?gclid=Cj0KCQjwkt0912345678UB") == []


def test_pii_outside_url_is_still_recognised() -> None:
    sentence = "請看 https://example.com/?g=0912345678 身分證 A123456789"

    assert _texts(sentence) == ["A123456789"]


def test_two_kinds_in_one_sentence() -> None:
    sentence = "身分證A123456789，手機0912345678"

    assert _types(sentence) == ["TW_ID", "TW_MOBILE"]


def test_ordinary_scam_message_without_pii() -> None:
    assert find_pii("點這裡領取您的中獎獎金，名額有限！") == []


@pytest.mark.parametrize("sentence", ["", "。。。!!!", "   "])
def test_empty_and_punctuation_only_sentences(sentence: str) -> None:
    assert find_pii(sentence) == []


def test_span_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="區間必須非空"):
        PiiSpan(start=3, end=3, entity_type="TW_ID")


def test_span_indices_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="區間起點不可為負"):
        PiiSpan(start=-1, end=4, entity_type="TW_ID")
