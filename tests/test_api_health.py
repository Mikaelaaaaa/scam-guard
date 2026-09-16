"""`GET /health`。

最重要的一條是「新鮮度每次呼叫時重算」。它不是效能取捨，是一個安靜失敗的
修補：`BlocklistStore.load()` 的檢查只在載入當下執行一次，於是一個啟動時通過
檢查的黑名單會在行程不重啟的情況下悄悄跨過門檻，而沉默在這個系統裡與
「沒有命中」完全同一個樣子。
"""

import socket
import urllib.request
from datetime import date

import pytest
from fastapi.testclient import TestClient

import api.app
import api.health
from api.app import BLOCKLIST_MAX_AGE_DAYS, REGISTRY, TABLE, UNREGISTERED_CHECKS, app
from api.health import STATUS_DEGRADED, STATUS_OK, blocklist_health, build_health

TODAY = date(2026, 9, 15)

FRESH = "2026-09-10"
STALE = "2026-01-01"


class _FakeStore:
    """只回 `manifest` 的替身。健康檢查只讀這一個公開屬性。"""

    def __init__(self, sources: dict[str, dict[str, object]]) -> None:
        self.manifest: dict[str, object] = {"sources": sources}


class _FrozenToday:
    """釘住的「今天」。用小類別而不是 lambda —— 專案禁止巢狀 `def` 與閉包，
    而「要捕捉一個值」正是改用顯式物件的信號。"""

    def __init__(self, day: date) -> None:
        self._day = day

    def __call__(self) -> date:
        return self._day


class _Tripwire:
    """任何一次呼叫都讓測試失敗。用來證明健康檢查不碰網路、不跑偵測。"""

    def __init__(self, what: str) -> None:
        self._what = what

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError(f"GET /health 不該觸發 {self._what}")


def _source(data_through: str, *, retired: bool = False) -> dict[str, object]:
    return {"data_through": data_through, "retired": retired}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def frozen_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """把「今天」釘住，讓新鮮度的測試不必真的等天數過去。"""
    monkeypatch.setattr(api.health, "utc_today", _FrozenToday(TODAY))


def test_health_is_always_a_two_hundred(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_health_reports_the_enabled_check_count(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["checks_registered"] == len(REGISTRY.enabled())
    assert body["checks_registered"] > 0


def test_health_lists_every_unregistered_check_with_its_reason(client: TestClient) -> None:
    listed = {
        item["name"]: item["reason"] for item in client.get("/health").json()["checks_unregistered"]
    }
    assert listed == dict(UNREGISTERED_CHECKS)
    assert listed["domain_age"] == "未注入外部查詢解析器"


def test_health_reports_the_weight_table(client: TestClient) -> None:
    weights = client.get("/health").json()["weights"]
    assert weights["signals"] == len(TABLE.signals)
    assert weights["path"].endswith("weights.toml")


def test_health_has_no_model_loaded_field(client: TestClient) -> None:
    """LLM 狀態由 `checks_registered`／`checks_unregistered` 既有機制回報，
    不另加只服務單一檢查的 `llm_loaded` schema 欄位。

    比對的是**欄位名稱**而非回應字串的子字串：`weights.path` 是伺服器上的
    絕對路徑，任何含有 `llm`／`model` 的目錄名（開發時的 worktree 就是一例）
    都會讓子字串比對誤判，而那與「有沒有回報模型狀態」完全無關。
    """
    payload = client.get("/health").json()
    forbidden = {"model", "llm", "gemma", "loaded_model", "model_loaded"}
    assert forbidden.isdisjoint(payload)
    for value in payload.values():
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)


def test_health_does_not_touch_the_network_or_run_detection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket, "create_connection", _Tripwire("建立 TCP 連線"))
    monkeypatch.setattr(urllib.request, "urlopen", _Tripwire("HTTP 請求"))
    monkeypatch.setattr(api.app, "detect", _Tripwire("一次完整的偵測流程"))
    assert client.get("/health").status_code == 200


def test_an_unloaded_blocklist_is_reported_as_such() -> None:
    health = blocklist_health(None, BLOCKLIST_MAX_AGE_DAYS)
    assert health.loaded is False
    assert health.sources == []


def test_an_unloaded_blocklist_does_not_degrade_the_status() -> None:
    health = build_health(REGISTRY, UNREGISTERED_CHECKS, None, BLOCKLIST_MAX_AGE_DAYS, TABLE)
    assert health.blocklist.loaded is False
    assert health.status == STATUS_OK


def test_a_failed_blocklist_load_degrades_the_status() -> None:
    unregistered = (*UNREGISTERED_CHECKS, ("url_blocklist", "snapshot broken"))
    health = build_health(REGISTRY, unregistered, None, BLOCKLIST_MAX_AGE_DAYS, TABLE)
    assert health.blocklist.loaded is False
    assert health.status == STATUS_DEGRADED


def test_a_fresh_blocklist_is_ok(frozen_today: None) -> None:
    store = _FakeStore({"176455": _source(FRESH)})
    health = build_health(REGISTRY, UNREGISTERED_CHECKS, store, {"176455": 7}, TABLE)
    assert health.status == STATUS_OK
    assert health.blocklist.loaded is True
    assert health.blocklist.sources[0].age_days == 5
    assert health.blocklist.sources[0].stale is False


def test_freshness_is_recomputed_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """行程啟動時新鮮、其後在不重啟的情況下跨過門檻 —— 此後的呼叫必須說話。

    這正是 `BlocklistStore.load()` 那一次檢查看不到的區間。
    """
    store = _FakeStore({"176455": _source("2026-09-10")})

    monkeypatch.setattr(api.health, "utc_today", _FrozenToday(date(2026, 9, 15)))
    at_startup = build_health(REGISTRY, UNREGISTERED_CHECKS, store, {"176455": 7}, TABLE)
    assert at_startup.status == STATUS_OK

    monkeypatch.setattr(api.health, "utc_today", _FrozenToday(date(2026, 9, 20)))
    later = build_health(REGISTRY, UNREGISTERED_CHECKS, store, {"176455": 7}, TABLE)
    assert later.status == STATUS_DEGRADED
    assert later.blocklist.sources[0].stale is True
    assert later.blocklist.sources[0].age_days == 10


def test_a_retired_source_never_degrades_the_status(frozen_today: None) -> None:
    """黑名單過期意味著召回下降而不是答案變錯 —— 一個 2024 年被通報的假投資
    網站今天仍然是一個假投資網站。停用的資料集不該擋住整個系統。"""
    store = _FakeStore({"160055": _source(STALE, retired=True)})
    health = build_health(REGISTRY, UNREGISTERED_CHECKS, store, {}, TABLE)
    assert health.status == STATUS_OK
    assert health.blocklist.sources[0].retired is True
    assert health.blocklist.sources[0].max_age_days == 0
    assert health.blocklist.sources[0].stale is False


def test_one_stale_source_among_fresh_ones_degrades_the_status(frozen_today: None) -> None:
    store = _FakeStore(
        {
            "160055": _source(STALE, retired=True),
            "165027": _source(FRESH),
            "176455": _source(STALE),
        }
    )
    health = build_health(
        REGISTRY,
        UNREGISTERED_CHECKS,
        store,
        {"165027": 7, "176455": 7},
        TABLE,
    )
    assert health.status == STATUS_DEGRADED
    assert {source.source: source.stale for source in health.blocklist.sources} == {
        "160055": False,
        "165027": False,
        "176455": True,
    }


def test_each_source_uses_its_own_threshold(frozen_today: None) -> None:
    """新鮮度門檻是逐 source 的部署參數，不可把單一數字套給所有來源。"""
    store = _FakeStore({"165027": _source(FRESH), "176455": _source(FRESH)})
    health = build_health(
        REGISTRY,
        UNREGISTERED_CHECKS,
        store,
        {"165027": 3, "176455": 30},
        TABLE,
    )

    assert health.status == STATUS_DEGRADED
    assert {
        source.source: (source.max_age_days, source.stale) for source in health.blocklist.sources
    } == {"165027": (3, True), "176455": (30, False)}


def test_a_non_retired_source_without_a_threshold_raises_and_names_it(
    frozen_today: None,
) -> None:
    store = _FakeStore({"165027": _source(FRESH), "176455": _source(FRESH)})

    with pytest.raises(ValueError) as caught:
        blocklist_health(store, {"176455": 30})

    message = str(caught.value)
    assert "165027" in message
    assert "max_age_days" in message


def test_a_missing_manifest_field_raises_and_names_it() -> None:
    """輸入不符要求就拋例外並指出欄位名，不猜一個預設值。"""
    store = _FakeStore({"176455": {"retired": False}})
    with pytest.raises(ValueError) as caught:
        blocklist_health(store, {"176455": 7})
    assert "data_through" in str(caught.value)


def test_a_non_object_source_raises_and_names_the_dataset() -> None:
    store = _FakeStore({"176455": "2026-09-10"})
    with pytest.raises(ValueError) as caught:
        blocklist_health(store, {"176455": 7})
    assert "176455" in str(caught.value)
