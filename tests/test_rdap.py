"""`net/rdap.py` 的狀態碼分類、事件解析、bootstrap 與快取行為。

**全部以構造的回應測試，不對外發出任何請求。** 假 transport 取代真正的
HTTPS 往返，`RdapLookup` 裡的判斷邏輯（狀態碼分類、逐一列舉的例外捕捉、
事件解析、快取讀寫）全部照常執行 —— 被換掉的只有 socket。
在無網路的環境執行本檔案應全數通過。
"""

import json
import socket
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from net.rdap import (
    BOOTSTRAP_FETCH_COMMAND,
    BOOTSTRAP_FILENAME,
    HttpResponse,
    RdapBootstrap,
    RdapLookup,
)
from net.rdap_cache import RdapCache
from scam_guard.domain_age import AgeOutcome, DomainAge

BOOTSTRAP_PAYLOAD = {
    "version": "1.0",
    "publication": "2026-09-09T23:00:03Z",
    "services": [
        [["com", "net"], ["https://rdap.example-registry.test/com/v1/"]],
        [["tw"], ["https://ccrdap.example-twnic.test/tw/"]],
        [["kg"], ["http://rdap.plaintext-only.test/"]],
    ],
}

REGISTERED_RESPONSE = {
    "objectClassName": "domain",
    "ldhName": "evil.com",
    "events": [
        {"eventAction": "registration", "eventDate": "2026-09-08T03:57:36Z"},
        {"eventAction": "expiration", "eventDate": "2029-05-30T16:00:00Z"},
        {"eventAction": "last changed", "eventDate": "2026-08-27T10:19:09Z"},
        {"eventAction": "last update of RDAP database", "eventDate": "2026-09-14T07:10:27Z"},
    ],
    "entities": [{"handle": "REDACTED", "roles": ["registrant"]}],
}

UPDATED_ONLY_RESPONSE = {
    "objectClassName": "domain",
    "events": [
        {"eventAction": "last changed", "eventDate": "2026-08-27T10:19:09Z"},
        {"eventAction": "last update of RDAP database", "eventDate": "2026-09-14T07:10:27Z"},
    ],
}


class FakeTransport:
    """回應由測試安排的假 transport，記錄每一次被請求的 URL。

    `raises` 讓測試直接指定要拋的例外實例 —— 逾時、DNS 失敗、未列舉的型別
    都由此模擬，於是 `RdapLookup` 真正的 except 子句在測試中被執行到，
    而不是被一個模擬過的版本取代。
    """

    def __init__(self, response: HttpResponse | None = None, raises: BaseException | None = None):
        self.requests: list[str] = []
        self._response = response
        self._raises = raises

    def __call__(self, url: str) -> HttpResponse:
        self.requests.append(url)
        if self._raises is not None:
            raise self._raises
        if self._response is None:
            raise AssertionError("測試未安排回應，且未安排例外")
        return self._response


def ok(payload: object) -> HttpResponse:
    return HttpResponse(status=200, body=json.dumps(payload).encode("utf-8"))


def lookup_of(transport: FakeTransport, tmp_path: Path, **cache_kwargs: object) -> RdapLookup:
    cache = RdapCache(tmp_path / "cache.sqlite3", **cache_kwargs)
    return RdapLookup(RdapBootstrap.parse(BOOTSTRAP_PAYLOAD), cache, transport)


# --- 註冊日期的取得 ---------------------------------------------------------


def test_registration_event_is_extracted(tmp_path: Path) -> None:
    transport = FakeTransport(ok(REGISTERED_RESPONSE))
    age = lookup_of(transport, tmp_path)("evil.com")
    assert age == DomainAge("evil.com", AgeOutcome.KNOWN, date(2026, 9, 8))
    assert transport.requests == ["https://rdap.example-registry.test/com/v1/domain/evil.com"]


def test_only_last_changed_is_no_data(tmp_path: Path) -> None:
    age = lookup_of(FakeTransport(ok(UPDATED_ONLY_RESPONSE)), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.NO_DATA
    assert age.registered_on is None


def test_missing_events_array_is_no_data(tmp_path: Path) -> None:
    age = lookup_of(FakeTransport(ok({"objectClassName": "domain"})), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.NO_DATA


def test_unparsable_event_date_is_no_data(tmp_path: Path) -> None:
    payload = {"events": [{"eventAction": "registration", "eventDate": "不是日期"}]}
    age = lookup_of(FakeTransport(ok(payload)), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.NO_DATA


def test_event_date_is_converted_to_utc(tmp_path: Path) -> None:
    payload = {
        "events": [{"eventAction": "registration", "eventDate": "2026-09-09T07:30:00+08:00"}]
    }
    age = lookup_of(FakeTransport(ok(payload)), tmp_path)("evil.com")
    assert age.registered_on == date(2026, 9, 8)


# --- 狀態碼與例外的分類 -----------------------------------------------------


def test_not_found_is_no_data(tmp_path: Path) -> None:
    age = lookup_of(FakeTransport(HttpResponse(404, b"")), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.NO_DATA


def test_upgrade_required_is_no_data(tmp_path: Path) -> None:
    """426 是關於**這個註冊局**的事實（它不接受 HTTP/1.1），不是關於這次請求的。
    TWNIC（`.tw`）就是這一種 —— 歸 UNAVAILABLE 會讓系統每隔幾分鐘重打一次
    一個注定失敗的請求。"""
    age = lookup_of(FakeTransport(HttpResponse(426, b"")), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.NO_DATA


@pytest.mark.parametrize("status", [429, 500, 503, 403, 302])
def test_non_200_non_404_is_unavailable(tmp_path: Path, status: int) -> None:
    age = lookup_of(FakeTransport(HttpResponse(status, b"")), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.UNAVAILABLE


def test_malformed_json_is_unavailable(tmp_path: Path) -> None:
    age = lookup_of(FakeTransport(HttpResponse(200, b"{not json")), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.UNAVAILABLE


def test_json_that_is_not_an_object_is_unavailable(tmp_path: Path) -> None:
    age = lookup_of(FakeTransport(HttpResponse(200, b"[]")), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.UNAVAILABLE


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("讀取逾時"),
        socket.gaierror("DNS 解析失敗"),
        ConnectionResetError("連線被重設"),
        ConnectionRefusedError("連線被拒"),
    ],
)
def test_enumerated_network_errors_become_unavailable(tmp_path: Path, error: Exception) -> None:
    age = lookup_of(FakeTransport(raises=error), tmp_path)("evil.com")
    assert age.outcome is AgeOutcome.UNAVAILABLE


def test_unenumerated_exception_propagates(tmp_path: Path) -> None:
    """未列舉的例外是「我們沒想到的失敗」，必須大聲壞掉 ——
    把它轉成 `UNAVAILABLE` 等於讓一個程式錯誤永遠偽裝成一個關於網域的事實。"""
    with pytest.raises(ValueError, match="沒想到的失敗"):
        lookup_of(FakeTransport(raises=ValueError("沒想到的失敗")), tmp_path)("evil.com")


# --- bootstrap --------------------------------------------------------------


def test_tld_not_in_bootstrap_is_no_data_without_any_request(tmp_path: Path) -> None:
    transport = FakeTransport(ok(REGISTERED_RESPONSE))
    age = lookup_of(transport, tmp_path)("example.jp")
    assert age.outcome is AgeOutcome.NO_DATA
    assert transport.requests == []


def test_plaintext_only_endpoint_is_not_used(tmp_path: Path) -> None:
    """只登記 http:// 端點的 TLD 視同未登記 —— 明文查詢會把網域暴露給
    路徑上的每一個觀察者，那正是本模組費力避免的事。"""
    transport = FakeTransport(ok(REGISTERED_RESPONSE))
    age = lookup_of(transport, tmp_path)("example.kg")
    assert age.outcome is AgeOutcome.NO_DATA
    assert transport.requests == []


def test_missing_bootstrap_snapshot_names_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        RdapBootstrap.load(tmp_path)
    assert BOOTSTRAP_FETCH_COMMAND in str(excinfo.value)


def test_bootstrap_loads_from_snapshot(tmp_path: Path) -> None:
    (tmp_path / BOOTSTRAP_FILENAME).write_text(json.dumps(BOOTSTRAP_PAYLOAD), encoding="utf-8")
    bootstrap = RdapBootstrap.load(tmp_path)
    assert bootstrap.endpoint_for("TW") == "https://ccrdap.example-twnic.test/tw/"
    assert bootstrap.endpoint_for("jp") is None


def test_bootstrap_without_services_is_rejected() -> None:
    with pytest.raises(ValueError, match="services"):
        RdapBootstrap.parse({"version": "1.0"})


# --- 快取 -------------------------------------------------------------------


def test_known_within_ttl_is_not_requested_twice(tmp_path: Path) -> None:
    transport = FakeTransport(ok(REGISTERED_RESPONSE))
    lookup = lookup_of(transport, tmp_path)
    first = lookup("evil.com")
    second = lookup("evil.com")
    assert first == second
    assert len(transport.requests) == 1


def test_cached_unavailable_stays_unavailable(tmp_path: Path) -> None:
    cache = RdapCache(tmp_path / "cache.sqlite3")
    now = datetime.now(timezone.utc)
    cache.put(DomainAge("evil.com", AgeOutcome.UNAVAILABLE), now=now)
    transport = FakeTransport(ok(REGISTERED_RESPONSE))
    lookup = RdapLookup(RdapBootstrap.parse(BOOTSTRAP_PAYLOAD), cache, transport)
    assert lookup("evil.com").outcome is AgeOutcome.UNAVAILABLE
    assert transport.requests == []


def test_unavailable_is_requeried_after_its_short_ttl(tmp_path: Path) -> None:
    cache = RdapCache(tmp_path / "cache.sqlite3", unavailable_ttl=timedelta(minutes=15))
    stale = datetime.now(timezone.utc) - timedelta(hours=3)
    cache.put(DomainAge("evil.com", AgeOutcome.UNAVAILABLE), now=stale)
    transport = FakeTransport(ok(REGISTERED_RESPONSE))
    lookup = RdapLookup(RdapBootstrap.parse(BOOTSTRAP_PAYLOAD), cache, transport)
    assert lookup("evil.com").outcome is AgeOutcome.KNOWN
    assert len(transport.requests) == 1


def test_known_is_requeried_after_seven_days(tmp_path: Path) -> None:
    cache = RdapCache(tmp_path / "cache.sqlite3")
    stale = datetime.now(timezone.utc) - timedelta(days=8)
    cache.put(DomainAge("evil.com", AgeOutcome.KNOWN, date(2020, 1, 1)), now=stale)
    transport = FakeTransport(ok(REGISTERED_RESPONSE))
    lookup = RdapLookup(RdapBootstrap.parse(BOOTSTRAP_PAYLOAD), cache, transport)
    assert lookup("evil.com").registered_on == date(2026, 9, 8)
    assert len(transport.requests) == 1


def test_unavailable_ttl_must_differ_from_the_seven_day_ttl(tmp_path: Path) -> None:
    cache = RdapCache(tmp_path / "cache.sqlite3")
    assert cache.ttl_for(AgeOutcome.UNAVAILABLE) < cache.ttl_for(AgeOutcome.KNOWN)
    assert cache.ttl_for(AgeOutcome.NO_DATA) == cache.ttl_for(AgeOutcome.KNOWN)
    assert cache.ttl_for(AgeOutcome.UNAVAILABLE) <= timedelta(hours=1)


def test_cache_table_has_exactly_four_columns(tmp_path: Path) -> None:
    """原始回應、註冊人、聯絡方式、註冊商都不存 —— 那是第三方個資，
    而我們需要的只有一個日期。"""
    path = tmp_path / "cache.sqlite3"
    RdapCache(path).put(
        DomainAge("evil.com", AgeOutcome.KNOWN, date(2026, 9, 8)),
        now=datetime.now(timezone.utc),
    )
    with closing(sqlite3.connect(path)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(domain_age)")]
    assert columns == ["domain", "outcome", "registered_on", "fetched_at"]


def test_prune_removes_only_expired_rows(tmp_path: Path) -> None:
    cache = RdapCache(tmp_path / "cache.sqlite3")
    now = datetime.now(timezone.utc)
    cache.put(DomainAge("fresh.com", AgeOutcome.KNOWN, date(2026, 9, 8)), now=now)
    cache.put(
        DomainAge("stale.com", AgeOutcome.KNOWN, date(2026, 9, 8)), now=now - timedelta(days=8)
    )
    cache.put(DomainAge("failed.com", AgeOutcome.UNAVAILABLE), now=now - timedelta(hours=3))
    assert cache.prune(now=now) == 2
    assert cache.get("fresh.com", now=now) is not None
    assert cache.count() == 1


def test_cache_rejects_a_naive_timestamp(tmp_path: Path) -> None:
    cache = RdapCache(tmp_path / "cache.sqlite3")
    with pytest.raises(ValueError, match="時區"):
        cache.put(DomainAge("evil.com", AgeOutcome.NO_DATA), now=datetime(2026, 9, 14))
