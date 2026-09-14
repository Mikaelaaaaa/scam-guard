"""白名單套用於 `url_blocklist` 的三種情形，以及它 MUST NOT 影響的四個檢查。

白名單做錯不是漏抓一則訊息，是**替攻擊者開一張通行證**，而那張通行證對所有
使用同一個平台的攻擊者一體適用。這份測試因此大半在驗「白名單沒有作用」。
"""

from inspect import signature
from pathlib import Path

import pytest

from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.normalize import build_document
from scam_guard.types import CheckResult, Message, Request, ScamType
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import (
    UrlBlocklistCheck,
    UrlBrandCheck,
    UrlHostShapeCheck,
    UrlShortenerCheck,
    UrlTldRiskCheck,
    load_tables,
    register_url_checks,
)
from tests.test_allowlist_store import write_snapshot as write_allowlist
from tests.test_blocklist_store import PSL_WITH_EXAMPLE, days_ago
from tests.test_blocklist_store import write_snapshot as write_blocklist

# `google.com`：平台被它的使用者連累（`play.google.com` 實測在 160055 上）。
# `mql5.com`：165 直接指控整個網站，而它也在 Tranco 上（排名 6,881）。
ALLOWLIST_ENTRIES = (
    {"domain": "google.com", "rank": 1},
    {"domain": "mql5.com", "rank": 881},
    {"domain": "bit.ly", "rank": 118},
    {"domain": "example.com", "rank": 900},
)

BLOCKLIST_ENTRIES = (
    {
        "host": "play.google.com",
        "url": "play.google.com/store/apps",
        "source": "160055",
        "first_seen": "2023-11-27",
        "last_seen": "2023-11-27",
    },
    {
        "host": "mql5.com",
        "url": "www.mql5.com",
        "source": "160055",
        "first_seen": "2023-01-02",
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
)


@pytest.fixture(name="psl")
def psl_fixture() -> PublicSuffixList:
    # `google.com` 與 `mql5.com` 要能算出可註冊網域，PSL 需要 `com`；
    # `PSL_WITH_EXAMPLE` 已含 `com`、`example`、`net`。
    return PublicSuffixList.parse(PSL_WITH_EXAMPLE)


@pytest.fixture(name="allowlist")
def allowlist_fixture(tmp_path: Path) -> RankAllowlist:
    directory = write_allowlist(tmp_path / "allowlist", ALLOWLIST_ENTRIES)
    return RankAllowlist.load(directory, max_age_days=30)


@pytest.fixture(name="store")
def store_fixture(tmp_path: Path, psl: PublicSuffixList) -> BlocklistStore:
    directory = write_blocklist(
        tmp_path / "blocklist", BLOCKLIST_ENTRIES, **{"176455": {"data_through": days_ago(10)}}
    )
    return BlocklistStore.load(directory, psl, max_age_days=60)


def run(check, *texts: str) -> list[CheckResult]:
    messages = [Message(text=text) for text in texts]
    return check(Request(messages=messages), build_document(messages))


def blocklist_check(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist | None
) -> UrlBlocklistCheck:
    return UrlBlocklistCheck(store, psl, load_tables(), allowlist=allowlist)


# --- 第一級：可註冊網域層的鄰居比對 ---------------------------------------


def test_platform_neighbour_produces_nothing(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    """`docs.google.com` 沒被通報，被通報的是 `play.google.com`。

    這正是實測中 599 則非詐騙訊息裡 6 則命中的那個形態。
    """
    assert run(blocklist_check(store, psl, allowlist), "https://docs.google.com/document/x") == []


def test_neighbour_matching_is_unchanged_off_the_allowlist(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    (result,) = run(blocklist_check(store, psl, allowlist), "https://a2.other.com/login")
    assert result.hit is True
    assert result.hard is False
    assert "a1.other.com" in result.detail


# --- 第二級：白名單網域的子網域精確命中 -----------------------------------


def test_subdomain_exact_hit_is_downgraded_not_deleted(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    """`play.google.com` 的硬證據偽陽性 —— 訊號仍在，只是不再短路 LLM。"""
    (result,) = run(blocklist_check(store, psl, allowlist), "https://play.google.com/store/apps")
    assert result.hit is True
    assert result.hard is False
    assert "play.google.com" in result.detail
    assert result.scam_types == [ScamType.FAKE_INVESTMENT]


def test_subdomain_parasite_is_not_let_through(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist, tmp_path: Path
) -> None:
    """`sites.google.com/view/xxx` 的主機不等於 `google.com`，白名單查不到它。

    結果仍然產出、依據文案與未套用白名單時逐字相同 —— apex 規則擋的正是
    「在高排名網站上開一個使用者頁面放釣魚內容」這個手法。
    """
    entries = ({**BLOCKLIST_ENTRIES[0], "host": "sites.google.com", "url": "sites.google.com/v"},)
    directory = write_blocklist(tmp_path / "parasite", entries)
    parasite_store = BlocklistStore.load(directory, psl, max_age_days=100_000)
    text = "https://sites.google.com/view/xxx-phishing"
    (with_allowlist,) = run(blocklist_check(parasite_store, psl, allowlist), text)
    (without,) = run(blocklist_check(parasite_store, psl, None), text)
    assert with_allowlist.hit is True
    assert with_allowlist.detail.startswith(without.detail)
    assert with_allowlist.scam_types == without.scam_types


def test_downgrade_detail_states_the_rank_and_list_id(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    """文案是可查證的事實，不是「知名網站」這類無法反駁的形容詞。"""
    (result,) = run(blocklist_check(store, psl, allowlist), "https://play.google.com/store/apps")
    assert "GQNVK" in result.detail
    assert "第 1 名" in result.detail
    for adjective in ("知名", "可信", "安全", "可疑"):
        assert adjective not in result.detail


def test_exact_hit_off_the_allowlist_stays_hard(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist, tmp_path: Path
) -> None:
    entries = ({**BLOCKLIST_ENTRIES[0], "host": "evil.com", "url": "evil.com/a"},)
    directory = write_blocklist(tmp_path / "off", entries)
    off_store = BlocklistStore.load(directory, psl, max_age_days=100_000)
    (result,) = run(blocklist_check(off_store, psl, allowlist), "https://evil.com/login")
    assert result.hard is True


# --- 第三種情形：黑名單直接指控整個網站 -----------------------------------


def test_blocklist_wins_when_the_apex_itself_is_listed(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    """流量不能推翻法律程序：一個有流量的詐騙網站仍然是詐騙網站。"""
    (result,) = run(blocklist_check(store, psl, allowlist), "https://mql5.com/")
    assert result.hit is True
    assert result.hard is True
    assert result.scam_types == [ScamType.FAKE_INVESTMENT]
    assert "GQNVK" not in result.detail


def test_www_prefix_counts_as_apex(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    """`normalize_host` 已去掉前導 `www.`，兩者在 apex 規則下是同一個東西。"""
    (result,) = run(blocklist_check(store, psl, allowlist), "https://www.mql5.com/")
    assert result.hard is True


# --- 未提供白名單時行為完全不變 -------------------------------------------


def test_neighbour_detail_keeps_its_exact_wording(
    store: BlocklistStore, psl: PublicSuffixList
) -> None:
    """未提供白名單時的依據文案與本 change 之前逐字相同。

    `_detail` 由「依 `hard` 推導後綴」改成「由呼叫端帶入後綴」，這條測試
    釘住那次重構沒有改到任何一個字。
    """
    (result,) = run(UrlBlocklistCheck(store, psl, load_tables()), "https://a2.other.com/login")
    assert result.detail == (
        "a1.other.com 於 2026-08 列入 165反詐騙諮詢專線_遭停止解析涉詐網站，網站性質：電子商務。"
        "訊息中的主機為 a2.other.com，與被通報的主機不同，同屬可註冊網域 other.com"
    )


def test_domains_off_the_allowlist_are_bit_for_bit_identical(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    for text in ("https://a2.other.com/login", "https://clean.example/a"):
        with_allowlist = run(blocklist_check(store, psl, allowlist), text)
        without = run(blocklist_check(store, psl, None), text)
        assert [(r.hit, r.hard, r.detail, r.scam_types, r.evidence) for r in with_allowlist] == [
            (r.hit, r.hard, r.detail, r.scam_types, r.evidence) for r in without
        ], text


def test_ablation_uses_two_registries_not_disable(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    """白名單不是 `Check`，所以它不出現在 registry 裡，也關不掉。"""
    tables = load_tables()
    with_allowlist = CheckRegistry()
    without = CheckRegistry()
    register_url_checks(with_allowlist, psl, tables, store=store, allowlist=allowlist)
    register_url_checks(without, psl, tables, store=store)
    names = [check.name for check in with_allowlist.enabled()]
    assert names == [check.name for check in without.enabled()]
    assert not any("allow" in name for name in names)
    text = "https://play.google.com/store/apps"
    (downgraded,) = run(
        [check for check in with_allowlist.enabled() if check.name == "url_blocklist"][0], text
    )
    (hard,) = run([check for check in without.enabled() if check.name == "url_blocklist"][0], text)
    assert downgraded.hard is False
    assert hard.hard is True


# --- 其餘四個檢查一律不受影響 ---------------------------------------------


def test_the_other_four_checks_cannot_receive_an_allowlist() -> None:
    """白名單接不進去，所以它們不可能受影響 —— 這比比對輸出更強的保證。"""
    for check_class in (UrlShortenerCheck, UrlTldRiskCheck, UrlHostShapeCheck, UrlBrandCheck):
        parameters = signature(check_class.__init__).parameters
        assert "allowlist" not in parameters, check_class.__name__


def test_shortener_check_still_reports_a_high_ranked_shortener(psl: PublicSuffixList) -> None:
    """`bit.ly` 實測排第 118 名；抑制它會讓信心值把「沒有訊號」誤讀為「乾淨」。"""
    results = run(UrlShortenerCheck(load_tables(), psl), "請點 https://bit.ly/xxx")
    assert [result.hit for result in results] == [True]
    assert "目的地未知" in results[0].detail


def test_brand_check_is_untouched(psl: PublicSuffixList) -> None:
    """跨層比對正是抓「寄生在高排名網站上的釣魚」的那條規則。"""
    tables = load_tables()
    brand = tables.brands[0]
    text = f"{brand.chinese_name}通知，請至 https://example.com/login 確認"
    results = run(UrlBrandCheck(tables, psl), text)
    assert any(brand.chinese_name in result.detail for result in results)


def test_host_shape_check_is_untouched(psl: PublicSuffixList) -> None:
    results = run(UrlHostShapeCheck(psl), "https://evil.com@example.com/")
    assert any("使用者資訊" in result.detail for result in results)


def test_tld_risk_check_is_untouched(psl: PublicSuffixList) -> None:
    """前 1,000 名內落在九個高風險 gTLD 的網域實測為 0，白名單對它是空操作。"""
    tables = load_tables()
    assert run(UrlTldRiskCheck(tables, psl), "https://example.com/a") == []


def test_allowlist_produces_no_check_result(
    store: BlocklistStore, psl: PublicSuffixList, allowlist: RankAllowlist
) -> None:
    """白名單不進 `Verdict.checks` —— 一個「這個網域很紅」的結果會被下游誤用。"""
    results = run(blocklist_check(store, psl, allowlist), "https://docs.google.com/document/x")
    assert results == []
    hit = run(blocklist_check(store, psl, allowlist), "https://play.google.com/store/apps")
    assert [result.name for result in hit] == ["url_blocklist"]
