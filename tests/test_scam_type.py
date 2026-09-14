"""詐騙類型詞彙表 `ScamType` 的形狀、邊界與 165 `CaseTitle` 對照。"""

import json
import re
from pathlib import Path

import pytest

from scam_guard.types import MERGED_CASE_TITLES, ScamType, case_titles

MEMBER_NAME = re.compile(r"\A[A-Z][A-Z0-9_]*\Z")

EXCLUDED_CASE_TITLES = ("網路購物", "假廣告", "信用卡遭盜刷", "假預付型消費", "其他")

CATEGORIES_PATH = Path(__file__).resolve().parent.parent / "categories.json"


def test_vocabulary_has_eighteen_members() -> None:
    assert len(ScamType) == 18


def test_member_names_are_uppercase_identifiers_and_values_are_non_empty() -> None:
    for member in ScamType:
        assert MEMBER_NAME.fullmatch(member.name), member.name
        assert member.value.strip(), member.name


def test_names_and_values_are_unique() -> None:
    names = [member.name for member in ScamType]
    values = [member.value for member in ScamType]

    assert len(set(names)) == len(names)
    assert len(set(values)) == len(values)


def test_lookup_by_value() -> None:
    assert ScamType("釣魚簡訊/惡意連結") is ScamType.PHISHING_LINK


def test_lookup_by_name() -> None:
    assert ScamType["FAKE_AUTHORITY"] is ScamType.FAKE_AUTHORITY


def test_unknown_value_raises() -> None:
    """未知值不得被吸收 —— 沒有任何成員是它的著陸點。"""
    with pytest.raises(ValueError):
        ScamType("釣魚網站")


def test_unknown_name_raises() -> None:
    """`weights.yaml` 拼錯成員名時必須爆掉，不得安靜略過該條權重。"""
    with pytest.raises(KeyError):
        ScamType["PHISHING"]


def test_full_width_variant_is_not_accepted() -> None:
    """值是 165 原文的逐字複本，全形半形變體是另一個字串。"""
    with pytest.raises(ValueError):
        ScamType("騙取金融帳戶（卡片）")


def test_no_catch_all_member() -> None:
    values = {member.value for member in ScamType}

    assert "其他" not in values
    assert "其它" not in values
    assert "其他案類" not in values


def test_excluded_case_titles_have_no_member() -> None:
    for title in EXCLUDED_CASE_TITLES:
        with pytest.raises(ValueError):
            ScamType(title)


def test_every_member_maps_to_at_least_one_case_title() -> None:
    for member in ScamType:
        assert len(case_titles(member)) >= 1, member.name


def test_merged_member_maps_to_two_case_titles() -> None:
    assert case_titles(ScamType.INSTALLMENT_CANCEL) == (
        "解除分期付款(騙買家)",
        "解除分期付款(騙賣家)",
    )


def test_unmerged_member_maps_to_its_own_value() -> None:
    assert case_titles(ScamType.FAKE_AUTHORITY) == ("假檢警/假冒公務機關",)


@pytest.mark.skipif(
    not CATEGORIES_PATH.exists(),
    reason="categories.json 是 165 儀錶板的原始統計，operator-local 未納入版控",
)
def test_case_titles_exist_in_the_165_dashboard_export() -> None:
    """對照的每個 `CaseTitle` 都必須是 165 匯出檔的既有鍵，不得是我們自己造的字。"""
    known = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))

    for member in ScamType:
        if member not in MERGED_CASE_TITLES:
            assert member.value in known, member.name
        for title in case_titles(member):
            assert title in known, title
