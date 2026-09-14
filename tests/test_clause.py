"""子句切分與否定範疇。"""

import inspect

import pytest

from scam_guard.rules.clause import CLAUSE_BREAK, NEGATION, is_negated, split_clauses


def clause_texts(sentence: str) -> list[str]:
    return [sentence[a:b] for a, b in split_clauses(sentence)]


def test_comma_splits_clauses() -> None:
    assert clause_texts("您的驗證碼是 123456,請勿告訴他人") == [
        "您的驗證碼是 123456",
        "請勿告訴他人",
    ]


def test_enumeration_comma_does_not_split() -> None:
    """頓號是列舉分隔，切開會讓第二個子句失去管轄它的否定詞。"""
    sentence = "不要把帳號、密碼告訴任何人"

    assert clause_texts(sentence) == [sentence]


def test_colon_splits_clauses() -> None:
    assert clause_texts("本行公告:請至下列連結更新資料") == ["本行公告", "請至下列連結更新資料"]


def test_sentence_without_break_is_one_clause() -> None:
    sentence = "請把剛收到的驗證碼告訴我"

    assert split_clauses(sentence) == [(0, len(sentence))]


def test_blank_only_fragments_do_not_become_clauses() -> None:
    assert clause_texts("提醒您: ,請勿外洩") == ["提醒您", "請勿外洩"]


def test_negation_holds_at_nine_characters() -> None:
    """迴歸：被否決的字元距離視窗方案 ——「不要」與「告訴」之間隔了 6 個字。

    視窗要涵蓋它得開到 7 字以上，而那樣的視窗會跨過逗號誤傷隔壁子句。
    視窗這個參數在兩個方向上都錯，沒有可用的取值。
    """
    sentence = "請不要在任何情況下告訴任何人"
    clause = split_clauses(sentence)[0]
    predicate_pos = sentence.index("告訴")

    assert sentence[sentence.index("不要") + 2 : predicate_pos] == "在任何情況下"
    assert is_negated(sentence, clause, predicate_pos) is True


def test_negation_does_not_cross_clauses() -> None:
    """迴歸：前面一句無害的否定不得蓋掉後面的索取，否則加一句廢話即可繞過。"""
    sentence = "不要告訴別人,把驗證碼傳給我"
    first, second = split_clauses(sentence)

    assert is_negated(sentence, first, sentence.index("告訴")) is True
    assert is_negated(sentence, second, sentence.index("傳給")) is False


def test_negation_after_predicate_does_not_count() -> None:
    sentence = "告訴我不要擔心"
    clause = split_clauses(sentence)[0]

    assert is_negated(sentence, clause, sentence.index("告訴")) is False


def test_is_negated_takes_no_distance_parameter() -> None:
    """簽章不得長出視窗參數 —— 視窗在兩個方向上都錯，沒有可用的取值。"""
    parameters = list(inspect.signature(is_negated).parameters)

    assert parameters == ["sentence", "clause", "predicate_pos"]


def test_predicate_outside_clause_raises() -> None:
    sentence = "不要告訴別人,把驗證碼傳給我"
    second = split_clauses(sentence)[1]

    with pytest.raises(ValueError, match="述語位置不在子句內"):
        is_negated(sentence, second, 0)


def test_clause_break_is_comma_and_colon_only() -> None:
    assert set(CLAUSE_BREAK) == {",", ":"}
    assert "、" not in CLAUSE_BREAK


def test_negation_list_covers_common_forms() -> None:
    assert {"不要", "請勿", "切勿", "千萬不", "嚴禁"} <= NEGATION
