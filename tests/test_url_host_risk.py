"""`url_shortener`、`url_tld_risk`、`url_host_shape` 三個檢查，以及三份對照表。"""

from datetime import date, timedelta

import pytest

from scam_guard.normalize import build_document
from scam_guard.types import Message, Request, ScamType
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import (
    Tables,
    UrlHostShapeCheck,
    UrlShortenerCheck,
    UrlTldRiskCheck,
    load_tables,
    mixed_scripts,
)

PSL_TEXT = """
// ===BEGIN ICANN DOMAINS===
com
tw
com.tw
gov.tw
cc
me
ly
is
co
net
link
icu
cfd
cyou
example
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


def run(check, *texts: str):
    messages = [Message(text=text) for text in texts]
    return check(Request(messages=messages), build_document(messages))


# --- 短網址 ---------------------------------------------------------------


def test_shortener_reports_unknown_destination(tables: Tables, psl: PublicSuffixList) -> None:
    check = UrlShortenerCheck(tables, psl)
    (result,) = run(check, "包裹配送失敗 reurl.cc/2xY3z 請更新地址")
    assert result.hit is True
    assert result.hard is False
    assert result.scam_types == []
    assert "目的地未知" in result.detail
    assert "不具資訊量" in result.detail


def test_shortener_granularity_is_per_url(tables: Tables, psl: PublicSuffixList) -> None:
    """`reurl.cc/aaa` 與 `reurl.cc/bbb` 是兩個不同的目的地。"""
    check = UrlShortenerCheck(tables, psl)
    results = run(check, "reurl.cc/aaa 與 reurl.cc/bbb")
    assert len(results) == 2


def test_legitimate_shortener_is_not_a_scam_signal(tables: Tables, psl: PublicSuffixList) -> None:
    """`lin.ee` 是 LINE 自己的短網址服務。命中表達的是證據不足。"""
    check = UrlShortenerCheck(tables, psl)
    (result,) = run(check, "請加入官方帳號 https://lin.ee/xxxx")
    assert result.scam_types == []
    assert result.hard is False


def test_shortener_check_has_no_network_call() -> None:
    """程式中無任何展開請求 —— 展開會把請求送到詐騙者控制的伺服器。"""
    import inspect

    import scam_guard.url_check as module

    source = inspect.getsource(module)
    for forbidden in ("urlopen", "requests.", "httpx", "socket", "http.client"):
        assert forbidden not in source, forbidden


# --- TLD 風險 -------------------------------------------------------------


def test_high_ratio_gtld_hits(tables: Tables, psl: PublicSuffixList) -> None:
    check = UrlTldRiskCheck(tables, psl)
    (result,) = run(check, "https://win-prize.icu/a")
    assert result.hit is True
    assert result.hard is False
    assert result.scam_types == [ScamType.PHISHING_LINK]
    assert "835.2" in result.detail
    assert "Cybercrime Information Center" in result.detail
    assert "2026-05/2026-07" in result.detail


@pytest.mark.parametrize("text", ["https://a.com/x", "https://a.com.tw/x", "https://a.cc/x"])
def test_common_taiwanese_tlds_do_not_hit(text: str, tables: Tables, psl: PublicSuffixList) -> None:
    """`.com` 是絕對數量榜首但比率最低；`.tw` 與 `.cc` 是 ccTLD，刻意不收。"""
    check = UrlTldRiskCheck(tables, psl)
    assert run(check, text) == []


def test_known_service_domains_do_not_hit(tables: Tables, psl: PublicSuffixList) -> None:
    risky_shortener = Tables(
        brands=tables.brands,
        shorteners={**tables.shorteners, "known.icu": tables.shorteners["reurl.cc"]},
        tld_risk=tables.tld_risk,
        baseline_tld=tables.baseline_tld,
        baseline_value=tables.baseline_value,
    )
    check = UrlTldRiskCheck(risky_shortener, psl)
    assert run(check, "https://known.icu/a") == []


def test_tld_risk_granularity_is_per_domain(tables: Tables, psl: PublicSuffixList) -> None:
    check = UrlTldRiskCheck(tables, psl)
    (result,) = run(check, "看 https://x.icu/a 好嗎?或 https://x.icu/b 呢?還有 https://x.icu/c")
    assert len(result.evidence) == 3


def test_tld_risk_table_excludes_cctlds_and_extension_like_tlds(tables: Tables) -> None:
    for absent in ("tw", "cc", "cn", "st", "vu", "zip", "mov"):
        assert absent not in tables.tld_risk


# --- 主機結構 -------------------------------------------------------------


def test_ip_literal_hits(psl: PublicSuffixList) -> None:
    check = UrlHostShapeCheck(psl)
    (result,) = run(check, "請登入 http://192.0.2.1/login")
    assert result.hit is True
    assert "IP 位址" in result.detail


def test_userinfo_names_the_real_host(psl: PublicSuffixList) -> None:
    check = UrlHostShapeCheck(psl)
    (result,) = run(check, "https://post.gov.tw@evil.com/")
    assert "evil.com" in result.detail
    assert "使用者資訊" in result.detail


def test_idna_failure_hits(psl: PublicSuffixList) -> None:
    check = UrlHostShapeCheck(psl)
    (result,) = run(check, f"https://{'中' * 60}.com/a")
    assert "IDNA" in result.detail


def test_cyrillic_lookalike_hits(psl: PublicSuffixList) -> None:
    """`аpple.com` 的首字是西里爾 U+0430。"""
    check = UrlHostShapeCheck(psl)
    (result,) = run(check, "https://аpple.com/login")
    assert "混用" in result.detail
    assert "西里爾" in result.detail


def test_cjk_host_does_not_hit_mixed_scripts(psl: PublicSuffixList) -> None:
    """中日韓與拉丁混用在台灣是合法且常見的，刻意不命中。"""
    check = UrlHostShapeCheck(psl)
    assert run(check, "https://中国.com/a") == []


def test_mixed_scripts_is_judged_per_label() -> None:
    assert mixed_scripts("аpple.com") == ["拉丁", "西里爾"]
    assert mixed_scripts("中国.com") == []
    assert mixed_scripts("apple.com") == []


def test_host_shape_results_are_never_hard(psl: PublicSuffixList) -> None:
    """精確度高但罕見 —— 短路省下的成本極少，一次誤判卻要付出整個 LLM 層。"""
    check = UrlHostShapeCheck(psl)
    results = run(check, "http://192.0.2.1/a 與 https://post.gov.tw@evil.com/")
    assert results
    assert all(result.hard is False for result in results)
    assert all(result.scam_types == [ScamType.PHISHING_LINK] for result in results)


# --- 對照表 ---------------------------------------------------------------


def test_every_table_entry_has_a_source(tables: Tables) -> None:
    for brand in tables.brands:
        assert brand.source and brand.verified_on
    for shortener in tables.shorteners.values():
        assert shortener.source and shortener.verified_on
    for risk in tables.tld_risk.values():
        assert risk.source and risk.as_of


def test_brand_verification_dates_are_within_a_year(tables: Tables) -> None:
    """失敗不代表資料錯，代表**該重新查了**。"""
    cutoff = date.today() - timedelta(days=365)
    stale = [
        brand.chinese_name
        for brand in tables.brands
        if date.fromisoformat(brand.verified_on) < cutoff
    ]
    assert not stale, f"下列品牌的 verified_on 已超過一年，請重新查證：{stale}"


def test_tables_are_not_under_an_ignored_directory() -> None:
    """`.gitignore` 的 `data/` 無前導斜線，任何深度都生效；`*.jsonl` 同理。"""
    from importlib import resources

    root = resources.files("scam_guard.tables")
    names = sorted(entry.name for entry in root.iterdir() if entry.name.endswith(".json"))
    assert names == ["brands.json", "ngram_model.json", "shorteners.json", "tld_risk.json"]
    assert "data" not in str(root).split("/")[-1]
