"""多來源快照的載入：逐 source 的新鮮度門檻，與不可對外顯示來源的攔阻。

這兩件事都是「加入一個性質完全不同的來源」照出來的既有設計缺陷。
一份月粒度的政府名單與一份宣稱「這些網址現在還線上」的小時級 feed
不可能共用同一個新鮮度門檻；而一個授權條款禁止對第三方顯示的來源，
不能靠「文件裡寫了」來擋。
"""

import json
from pathlib import Path

import pytest

from scam_guard.blocklist import MANIFEST_FILENAME, BlocklistStore
from scam_guard.normalize import build_document
from scam_guard.types import Message, Request, ScamType
from scam_guard.url_check import UrlBlocklistCheck, load_tables
from tests.test_blocklist_store import (
    DEFAULT_ENTRIES,
    GOVERNMENT_MAX_AGE,
    days_ago,
    psl_of,
    write_snapshot,
)

FEED_ENTRIES = DEFAULT_ENTRIES + (
    {
        "host": "paypal-secure.example",
        "url": "https://paypal-secure.example/login",
        "source": "phishtank",
        "first_seen": "2026-09-13T04:12:31+00:00",
        "last_seen": "2026-09-13T04:12:31+00:00",
        "target": "PayPal",
    },
)

PHISHTANK_META: dict[str, object] = {
    "title": "PhishTank online-valid",
    "agency": "Cisco Talos Intelligence Group",
    "data_through": f"{days_ago(1)}T04:12:31+00:00",
    "data_through_granularity": "second",
    "data_through_source": "parsed_from_content",
    "redistributable": True,
    "domain_level_matching": False,
    "license": "PhishTank Archived ToU",
    "license_note": "現行條款指向 Cisco EULA，未查證",
}

OPENPHISH_META: dict[str, object] = {
    **PHISHTANK_META,
    "title": "OpenPhish community feed",
    "agency": "OpenPhish",
    "redistributable": False,
    "license": "未宣告授權（GitHub API 回 license: null），適用其 Terms of Use",
    "license_note": (
        "you agree not to ... distribute, display, disclose ... "
        "any portion of the information ... to any third party"
    ),
}

OPENPHISH_ENTRIES = DEFAULT_ENTRIES + (
    {
        "host": "wallet-restore.example",
        "url": "https://wallet-restore.example/seed",
        "source": "openphish",
        "first_seen": "2026-09-14T12:00:03+00:00",
        "last_seen": "2026-09-14T12:00:03+00:00",
    },
)

WITH_PHISHTANK = {"phishtank": PHISHTANK_META}
WITH_OPENPHISH = {"openphish": OPENPHISH_META}


def load(directory: Path, **kwargs: object) -> BlocklistStore:
    return BlocklistStore.load(directory, psl_of(), **kwargs)


# --- 逐 source 的新鮮度門檻 -----------------------------------------------


def test_thresholds_are_per_source(tmp_path: Path) -> None:
    """176455 的 60 天與 phishtank 的 2 天同時成立。"""
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    assert store.by_host("paypal-secure.example")


def test_missing_threshold_names_the_source(tmp_path: Path) -> None:
    """沿用「呼叫端為每一個數字負責」，只是從一個數字擴充成一組數字。"""
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    with pytest.raises(ValueError) as excinfo:
        load(directory, max_age_days=GOVERNMENT_MAX_AGE)
    message = str(excinfo.value)
    assert "phishtank" in message
    assert "max_age_days" in message


def test_each_source_is_judged_independently(tmp_path: Path) -> None:
    """一份三天前的小時級 feed 過期，而同一份快照裡的政府名單完全正常。"""
    directory = write_snapshot(
        tmp_path,
        FEED_ENTRIES,
        **{"phishtank": {**PHISHTANK_META, "data_through": f"{days_ago(3)}T04:12:31+00:00"}},
    )
    with pytest.raises(ValueError) as excinfo:
        load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    message = str(excinfo.value)
    assert "phishtank" in message
    assert "176455" not in message


def test_retired_source_needs_no_threshold(tmp_path: Path) -> None:
    """160055 已停止更新，要求它的門檻等於要求一個永遠不會被滿足的數字。"""
    directory = write_snapshot(tmp_path)
    assert load(directory, max_age_days=GOVERNMENT_MAX_AGE).entry_count == len(DEFAULT_ENTRIES)


# --- 不可對外顯示的來源 ---------------------------------------------------


def test_non_redistributable_source_is_refused_by_default(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, OPENPHISH_ENTRIES, **WITH_OPENPHISH)
    with pytest.raises(ValueError) as excinfo:
        load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "openphish": 2})
    message = str(excinfo.value)
    assert "openphish" in message
    assert "distribute, display, disclose" in message


def test_offline_evaluation_passes_the_flag_explicitly(tmp_path: Path) -> None:
    """那一行程式碼本身就是一份紀錄：誰在什麼地方決定了這件事。"""
    directory = write_snapshot(tmp_path, OPENPHISH_ENTRIES, **WITH_OPENPHISH)
    store = load(
        directory,
        max_age_days={**GOVERNMENT_MAX_AGE, "openphish": 2},
        require_redistributable=False,
    )
    assert store.entry_count == len(OPENPHISH_ENTRIES)


def test_redistributable_is_checked_before_freshness(tmp_path: Path) -> None:
    """授權問題不該被一個新鮮度錯誤蓋掉 —— 兩者的處置完全不同。"""
    directory = write_snapshot(
        tmp_path,
        OPENPHISH_ENTRIES,
        **{"openphish": {**OPENPHISH_META, "data_through": f"{days_ago(90)}T00:00:00+00:00"}},
    )
    with pytest.raises(ValueError, match="openphish"):
        load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "openphish": 2})


# --- 舊快照 ---------------------------------------------------------------


def test_old_snapshot_missing_the_new_fields_raises(tmp_path: Path) -> None:
    """缺欄位就拋例外並說明重跑取得程式即可，不猜一個預設值。"""
    directory = write_snapshot(tmp_path)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["sources"]["176455"]["redistributable"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load(directory, max_age_days=GOVERNMENT_MAX_AGE)
    assert "redistributable" in str(excinfo.value)
    assert "176455" in str(excinfo.value)


# --- 比對層級與強度（`url_blocklist`）-------------------------------------


def check_of(store: BlocklistStore) -> UrlBlocklistCheck:
    return UrlBlocklistCheck(store, psl_of(), load_tables())


def run(check: UrlBlocklistCheck, text: str) -> list:
    messages = [Message(text=text)]
    return check(Request(messages=messages), build_document(messages))


def test_feed_records_do_not_join_the_domain_level_index(tmp_path: Path) -> None:
    """平台鄰居不牽連：讓 feed 參與網域層比對，等於讓該平台每一個合法使用者命中。"""
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    assert store.by_registrable_domain("paypal-secure.example") == ()
    assert run(check_of(store), "https://another.paypal-secure.example/x") == []


def test_government_records_still_match_at_domain_level(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    (result,) = run(check_of(store), "https://a2.other.com/login")
    assert result.hard is False
    assert "a1.other.com" in result.detail


def test_feed_exact_host_hit_is_not_hard_evidence(tmp_path: Path) -> None:
    """`hard=True` 會短路掉整個 LLM 層，其門檻維持為已走完法律程序的第一方事實。"""
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    (result,) = run(check_of(store), "https://paypal-secure.example/login")
    assert result.hit is True
    assert result.hard is False
    assert result.scam_types == [ScamType.PHISHING_LINK]


def test_government_exact_host_hit_stays_hard(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    (result,) = run(check_of(store), "https://evil.com/login")
    assert result.hard is True


def test_feed_detail_states_the_verification_moment_not_the_present(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    (result,) = run(check_of(store), "https://paypal-secure.example/login")
    assert "2026-09-13T04:12:31+00:00" in result.detail
    assert "在該時點仍在線上" in result.detail
    assert "目前仍在運作" not in result.detail
    assert "列入" not in result.detail


def test_target_goes_to_the_detail_not_the_scam_types(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, FEED_ENTRIES, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    (result,) = run(check_of(store), "https://paypal-secure.example/login")
    assert "冒用品牌：PayPal" in result.detail
    assert result.scam_types == [ScamType.PHISHING_LINK]


def test_same_host_in_two_sources_writes_both_statements(tmp_path: Path) -> None:
    """兩句都是事實，不需要仲裁，也不擇一捨棄。"""
    entries = FEED_ENTRIES + (
        {
            "host": "evil.com",
            "url": "https://evil.com/login",
            "source": "phishtank",
            "first_seen": "2026-09-13T04:12:31+00:00",
            "last_seen": "2026-09-13T04:12:31+00:00",
            "target": "PayPal",
        },
    )
    directory = write_snapshot(tmp_path, entries, **WITH_PHISHTANK)
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "phishtank": 2})
    (result,) = run(check_of(store), "https://evil.com/login")
    assert "列入 165反詐騙諮詢專線_遭停止解析涉詐網站" in result.detail
    assert "PhishTank online-valid 驗證為釣魚網址" in result.detail
    assert result.hard is True


def test_unknown_source_raises_and_lists_the_known_keys(tmp_path: Path) -> None:
    entries = DEFAULT_ENTRIES + (
        {
            "host": "mystery.example",
            "url": "https://mystery.example/",
            "source": "someone-elses-feed",
            "first_seen": "2026-09-13",
            "last_seen": "2026-09-13",
        },
    )
    directory = write_snapshot(tmp_path, entries, **{"someone-elses-feed": {}})
    store = load(directory, max_age_days={**GOVERNMENT_MAX_AGE, "someone-elses-feed": 2})
    with pytest.raises(ValueError) as excinfo:
        run(check_of(store), "https://mystery.example/")
    assert "someone-elses-feed" in str(excinfo.value)
    assert "phishtank" in str(excinfo.value)
