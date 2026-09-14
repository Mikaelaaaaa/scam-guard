"""`url_brand` 的三種判定，以及品牌表的內容約束。"""

import json
from importlib import resources

import pytest

from scam_guard.normalize import build_document
from scam_guard.types import Message, Request, ScamType
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import Tables, UrlBrandCheck, levenshtein, load_tables, visual_normalize

PSL_TEXT = """
// ===BEGIN ICANN DOMAINS===
com
tw
com.tw
gov.tw
cn
me
net
cc
// ===END ICANN DOMAINS===
// ===BEGIN PRIVATE DOMAINS===
wixsite.com
// ===END PRIVATE DOMAINS===
"""


@pytest.fixture(name="psl")
def psl_fixture() -> PublicSuffixList:
    return PublicSuffixList.parse(PSL_TEXT)


@pytest.fixture(name="tables")
def tables_fixture() -> Tables:
    return load_tables()


@pytest.fixture(name="check")
def check_fixture(tables: Tables, psl: PublicSuffixList) -> UrlBrandCheck:
    return UrlBrandCheck(tables, psl, weight=1.0)


def run(check: UrlBrandCheck, *texts: str):
    messages = [Message(text=text) for text in texts]
    return check(Request(messages=messages), build_document(messages))


# --- 判定一：品牌 token 出現在非官方網域 ----------------------------------


def test_official_domain_used_as_a_subdomain(check: UrlBrandCheck) -> None:
    results = run(check, "https://post.gov.tw.abc.cn/query")
    assert results
    assert any("abc.cn" in result.detail for result in results)


def test_brand_label_in_someone_elses_domain(check: UrlBrandCheck) -> None:
    results = run(check, "https://esunbank.tw-verify.com/login")
    assert results
    assert any("tw-verify.com" in result.detail for result in results)


def test_official_host_does_not_hit(check: UrlBrandCheck) -> None:
    assert run(check, "https://ebank.esunbank.com.tw/") == []


def test_both_official_domains_of_one_brand_do_not_hit(check: UrlBrandCheck) -> None:
    """中國信託的 ctbcbank.com 與 ctbcbank.com.tw 同在一張憑證的 SAN 上。"""
    assert run(check, "https://www.ctbcbank.com/") == []
    assert run(check, "https://www.ctbcbank.com.tw/") == []


def test_shopee_com_tw_is_always_a_hit(check: UrlBrandCheck) -> None:
    """蝦皮官方只有 shopee.tw；shopee.com.tw 完全無 DNS 記錄。"""
    results = run(check, "https://shopee.com.tw/order")
    assert results
    assert any("shopee.com.tw" in result.detail for result in results)


# --- 判定二：編輯距離 -----------------------------------------------------


def test_one_character_away_hits_by_distance(check: UrlBrandCheck) -> None:
    results = run(check, "https://esunbannk.com/login")
    assert results
    assert any("編輯距離為 1" in result.detail for result in results)
    assert any("玉山銀行" in result.detail for result in results)


def test_visual_substitution_hits(check: UrlBrandCheck) -> None:
    """以數字 0 取代字母 o 之後，視覺替換讓距離變成 0，但網域不是官方的。"""
    results = run(check, "https://carr0usell.tw/deal")
    assert results


def test_visual_normalize_table() -> None:
    assert visual_normalize("carrn") == "carm"
    assert visual_normalize("g00gle") == "google"
    assert visual_normalize("1ine") == "line"


def test_levenshtein_is_a_real_edit_distance() -> None:
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("abc", "ac") == 1
    assert levenshtein("kitten", "sitting") == 3


# --- 判定三：文字提到品牌但連結不屬該品牌 ---------------------------------


def test_chinese_brand_name_with_an_unrelated_link(check: UrlBrandCheck) -> None:
    """`tw-mail.com` 與「中華郵政」的字面相似度是零，只有跨層比對抓得到。"""
    results = run(check, "中華郵政通知您包裹地址有誤 https://tw-mail.com/track")
    mention_results = [result for result in results if "則訊息提到" in result.detail]
    assert mention_results
    (result,) = mention_results
    assert "中華郵政" in result.detail
    assert "tw-mail.com" in result.detail
    assert len(result.evidence) == 2


def test_official_link_in_the_same_message_does_not_hit(check: UrlBrandCheck) -> None:
    assert run(check, "中華郵政通知您 https://www.post.gov.tw/ 查詢") == []


def test_mentions_do_not_cross_messages(check: UrlBrandCheck) -> None:
    texts = ["我在蝦皮購物買東西", "好", "嗯", "對", "是", "看這個 https://news.example.net/a"]
    assert run(check, *texts) == []


# --- 類型與強度 -----------------------------------------------------------


def test_parcel_sounding_domain_still_emits_phishing_link(check: UrlBrandCheck) -> None:
    """從主機名猜案類會與規則層產生無法仲裁的衝突，所以只輸出手段不輸出託辭。"""
    results = run(check, "黑貓宅急便包裹招領 https://tw-711-parcel.com/pickup")
    assert results
    for result in results:
        assert result.scam_types == [ScamType.PHISHING_LINK]
        assert ScamType.FAKE_PARCEL not in result.scam_types
        assert ScamType.ORDER_ANOMALY not in result.scam_types


def test_all_brand_results_are_soft(check: UrlBrandCheck) -> None:
    """新聞轉傳與防詐宣導會產生同樣的形態。"""
    results = run(check, "中華郵政通知 https://tw-mail.com/track")
    assert results
    assert all(result.hard is False for result in results)


# --- 品牌表的內容 ---------------------------------------------------------


def test_table_excludes_the_two_verified_non_official_domains(tables: Tables) -> None:
    assert "shopee.com.tw" not in tables.official_domains
    assert "linepay.com.tw" not in tables.official_domains
    assert "line.me" in tables.official_domains


def test_official_domains_are_lists_not_single_values(tables: Tables) -> None:
    by_name = {brand.chinese_name: brand for brand in tables.brands}
    assert set(by_name["玉山銀行"].official_domains) == {"esunbank.com", "esunbank.com.tw"}
    assert set(by_name["中國信託"].official_domains) == {"ctbcbank.com", "ctbcbank.com.tw"}
    assert by_name["蝦皮購物"].official_domains == ("shopee.tw",)


def test_every_brand_records_why_it_is_listed(tables: Tables) -> None:
    for brand in tables.brands:
        assert any(marker in brand.source for marker in ("165", "F-ISAC", "金管會")), (
            brand.chinese_name
        )


def test_chinese_brand_names_live_only_in_the_table() -> None:
    """品牌詞表與官方網域清單是同一份資料，不另建一份。"""
    raw = json.loads(
        resources.files("scam_guard.tables").joinpath("brands.json").read_text(encoding="utf-8")
    )
    names = {brand["chinese_name"] for brand in raw["brands"]}
    import scam_guard.url_check as module
    import inspect

    source = inspect.getsource(module)
    assert not (names & set(source.split())), "品牌中文名不得寫進程式碼"


def test_the_165_internal_api_is_not_a_data_source() -> None:
    import inspect

    import scam_guard.url_check as module
    import tools.fetch_blocklist as fetch

    for target in (module, fetch):
        assert "165.npa.gov.tw" not in inspect.getsource(target)
