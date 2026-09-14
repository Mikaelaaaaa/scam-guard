"""`url_blocklist` 的比對層級、`hard` 判定、類型映射與依據文案。"""

from pathlib import Path

import pytest

from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import build_document
from scam_guard.pipeline import SKIPPED, detect
from scam_guard.types import Message, Request, ScamType
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import UrlBlocklistCheck, load_tables, register_url_checks
from tests.test_blocklist_store import PSL_WITH_EXAMPLE, days_ago, write_snapshot

WEIGHTS = {
    "url_blocklist": 1.0,
    "url_shortener": 1.0,
    "url_tld_risk": 1.0,
    "url_host_shape": 1.0,
    "url_brand": 1.0,
}

ENTRIES = (
    {
        "host": "evil.com",
        "url": "evil.com/a",
        "source": "176455",
        "first_seen": "2026-08",
        "last_seen": "2026-08",
        "nature": "金融保險",
    },
    {
        "host": "evil.com",
        "url": "www.evil.com",
        "source": "160055",
        "first_seen": "2023-12-18",
        "last_seen": "2025-12-31",
    },
    {
        "host": "a1.other.com",
        "url": "a1.other.com",
        "source": "176455",
        "first_seen": "2026-08",
        "last_seen": "2026-08",
        "nature": "電子商務",
    },
    {
        "host": "onepage.example",
        "url": "https://onepage.example/x#/login",
        "source": "165027",
        "first_seen": "2026-08-26",
        "last_seen": "2026-08-26",
        "site_created_on": "2026-06-20T15:13:38",
    },
    {
        "host": "bad.net",
        "url": "bad.net/b",
        "source": "176455",
        "first_seen": "2026-08",
        "last_seen": "2026-08",
        "nature": "電子商務",
    },
)


@pytest.fixture(name="psl")
def psl_fixture() -> PublicSuffixList:
    return PublicSuffixList.parse(PSL_WITH_EXAMPLE)


@pytest.fixture(name="check")
def check_fixture(tmp_path: Path, psl: PublicSuffixList) -> UrlBlocklistCheck:
    directory = write_snapshot(tmp_path, ENTRIES, **{"176455": {"data_through": days_ago(10)}})
    store = BlocklistStore.load(directory, psl, max_age_days=60)
    return UrlBlocklistCheck(store, psl, load_tables(), weight=1.0)


def run(check: UrlBlocklistCheck, *texts: str):
    messages = [Message(text=text) for text in texts]
    return check(Request(messages=messages), build_document(messages))


def test_exact_host_match_is_hard_evidence(check: UrlBlocklistCheck) -> None:
    (result,) = run(check, "請點 https://evil.com/login")
    assert result.hit is True
    assert result.hard is True
    assert result.name == "url_blocklist"


def test_same_domain_different_host_is_not_hard(check: UrlBlocklistCheck) -> None:
    """PSL 的 PRIVATE 區段不完整，這個命中可能來自同一個託管平台的鄰居。"""
    (result,) = run(check, "請點 https://a2.other.com/login")
    assert result.hit is True
    assert result.hard is False
    assert "a1.other.com" in result.detail
    assert "a2.other.com" in result.detail


def test_path_difference_still_matches_exactly(check: UrlBlocklistCheck) -> None:
    """清單記錄的是 `evil.com/a`，訊息中的是 `evil.com/b`，以主機精確命中。"""
    (result,) = run(check, "https://evil.com/b")
    assert result.hard is True


def test_160055_maps_to_fake_investment(tmp_path: Path, psl: PublicSuffixList) -> None:
    entries = ({**ENTRIES[1], "host": "bet.example", "url": "bet.example"},)
    directory = write_snapshot(tmp_path, entries)
    store = BlocklistStore.load(directory, psl, max_age_days=60)
    (result,) = UrlBlocklistCheck(store, psl, load_tables(), weight=1.0)(
        Request(messages=[Message(text="https://bet.example/x")]),
        build_document([Message(text="https://bet.example/x")]),
    )
    assert result.scam_types == [ScamType.FAKE_INVESTMENT]


def test_165027_maps_to_phishing_link_not_online_shopping(check: UrlBlocklistCheck) -> None:
    """一頁式購物詐騙輸出 `PHISHING_LINK` —— 「網路購物」已被詞彙表排除。"""
    (result,) = run(check, "限時特價 https://onepage.example/x")
    assert result.scam_types == [ScamType.PHISHING_LINK]


def test_nature_goes_to_detail_not_scam_types(check: UrlBlocklistCheck) -> None:
    """『金融保險』是被冒用的產業，不是 165 案類。"""
    (result,) = run(check, "https://evil.com/login")
    assert "網站性質：金融保險" in result.detail
    assert ScamType.FAKE_INVESTMENT in result.scam_types
    assert ScamType.PHISHING_LINK in result.scam_types


def test_multiple_sources_emit_all_types(check: UrlBlocklistCheck) -> None:
    (result,) = run(check, "https://evil.com/login")
    assert sorted(t.name for t in result.scam_types) == ["FAKE_INVESTMENT", "PHISHING_LINK"]


def test_detail_is_a_factual_statement(check: UrlBlocklistCheck) -> None:
    (result,) = run(check, "https://evil.com/login")
    assert "2026-08" in result.detail
    assert "165反詐騙諮詢專線_遭停止解析涉詐網站" in result.detail
    for adjective in ("可疑", "危險"):
        assert adjective not in result.detail


def test_two_urls_in_one_domain_produce_one_result(check: UrlBlocklistCheck) -> None:
    (result,) = run(check, "看這個 https://evil.com/a 對吧?也可以看 https://evil.com/b 謝謝")
    assert result.evidence == [(0, 0), (0, 1)]


def test_two_domains_produce_two_results(check: UrlBlocklistCheck) -> None:
    results = run(check, "https://evil.com/a 與 https://bad.net/b")
    assert len(results) == 2


def test_clean_url_returns_empty(check: UrlBlocklistCheck) -> None:
    assert run(check, "https://clean.example/a") == []


def test_registry_omits_the_check_without_a_store(psl: PublicSuffixList) -> None:
    """未提供 store 時不註冊一個永遠不命中的空檢查。"""
    registry = CheckRegistry()
    register_url_checks(registry, psl, load_tables(), weights=WEIGHTS, store=None)
    assert "url_blocklist" not in [check.name for check in registry.enabled()]
    assert len(registry.enabled()) == 4


def test_registry_includes_the_check_with_a_store(
    psl: PublicSuffixList, check: UrlBlocklistCheck
) -> None:
    registry = CheckRegistry()
    register_url_checks(registry, psl, load_tables(), weights=WEIGHTS, store=check._store)
    names = [registered.name for registered in registry.enabled()]
    assert names == [
        "url_blocklist",
        "url_shortener",
        "url_tld_risk",
        "url_host_shape",
        "url_brand",
    ]
    assert all(registered.stage is Stage.LOCAL for registered in registry.enabled())


def test_hard_hit_short_circuits_expensive_checks(
    psl: PublicSuffixList, check: UrlBlocklistCheck
) -> None:
    """精確命中 `hard=True` 會讓 `EXPENSIVE` 檢查完全不執行。

    這也是 `add-domain-age` 的 RDAP **不會對最像詐騙的那批網域外流**的機制。
    """
    registry = CheckRegistry()
    register_url_checks(registry, psl, load_tables(), weights=WEIGHTS, store=check._store)
    expensive = RecordingExpensiveCheck()
    registry.register(expensive)
    verdict = detect(Request(messages=[Message(text="https://evil.com/login")]), registry)
    assert expensive.calls == 0
    skipped = [r for r in verdict.checks if r.name == "recording_expensive"]
    assert [r.detail for r in skipped] == [SKIPPED]


def test_no_hard_hit_runs_expensive_checks(psl: PublicSuffixList, check: UrlBlocklistCheck) -> None:
    registry = CheckRegistry()
    register_url_checks(registry, psl, load_tables(), weights=WEIGHTS, store=check._store)
    expensive = RecordingExpensiveCheck()
    registry.register(expensive)
    detect(Request(messages=[Message(text="https://clean.example/a")]), registry)
    assert expensive.calls == 1


class RecordingExpensiveCheck:
    """記錄自己有沒有被呼叫的 `EXPENSIVE` 檢查。"""

    name = "recording_expensive"
    stage = Stage.EXPENSIVE

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, req: Request, doc) -> list:
        self.calls += 1
        return []


def test_each_check_can_be_disabled_independently(
    psl: PublicSuffixList, check: UrlBlocklistCheck
) -> None:
    registry = CheckRegistry()
    register_url_checks(registry, psl, load_tables(), weights=WEIGHTS, store=check._store)
    registry.disable("url_tld_risk")
    names = [registered.name for registered in registry.enabled()]
    assert "url_tld_risk" not in names
    assert len(names) == 4
