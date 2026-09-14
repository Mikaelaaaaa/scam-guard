"""`POST /check` 的路由、組裝層與 log 政策。

本檔測的是接線：形狀由 `tests/test_api_schema.py` 守，這裡守的是
「那份形狀真的接到了 `detect()`」與「組裝層的決定是顯式的」。
"""

import logging

import pytest
from fastapi.testclient import TestClient

from api.app import (
    BLOCKLIST_MAX_AGE_DAYS,
    REGISTRY,
    TABLE,
    UNREGISTERED_CHECKS,
    app,
    build_registry,
)
from scam_guard.check import CheckRegistry, Stage
from scam_guard.types import CheckResult, Request
from scam_guard.weights import load_weights

DOMAIN_AGE = "domain_age"

SCAM_TEXT = "您好，我是警察，您的帳戶涉及洗錢，請立刻到 ATM 依指示操作解除分期付款。"
BENIGN_TEXT = "明天見。"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class _NeverRegisteredCheck:
    """一個權重表裡不存在的檢查，用來證明啟動時的驗證真的會擋人。"""

    name = "never_in_the_weight_table"
    stage = Stage.LOCAL

    def __call__(self, req: Request, doc: object) -> list[CheckResult]:
        return []


def test_valid_request_gets_a_verdict(client: TestClient) -> None:
    response = client.post("/check", json={"messages": [{"text": SCAM_TEXT, "from": "them"}]})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "scam_probability",
        "abstained",
        "confidence",
        "scam_type",
        "evidence",
        "actions",
    }
    assert body["abstained"] is False
    assert body["scam_probability"] is not None
    assert body["evidence"]


def test_abstention_is_a_two_hundred(client: TestClient) -> None:
    """拒答是系統正確地判斷自己沒有依據，是一次成功的推論，不是一次失敗。"""
    response = client.post("/check", json={"messages": [{"text": BENIGN_TEXT}]})
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["scam_probability"] is None


def test_context_and_latest_are_both_passed_through(client: TestClient) -> None:
    """最後一則是待判定的訊息，其餘是前文 —— 這個切分由 `Request` 負責。"""
    response = client.post(
        "/check",
        json={
            "messages": [
                {"text": "你好", "from": "them", "at": "2026-09-14T09:00:00Z"},
                {"text": SCAM_TEXT, "from": "them", "at": "2026-09-14T10:00:00Z"},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["abstained"] is False


def test_registry_and_table_are_shared_across_requests(client: TestClient) -> None:
    """兩次呼叫用的是同一份實例，不重新載入。"""
    before = (id(REGISTRY), id(TABLE))
    client.post("/check", json={"messages": [{"text": BENIGN_TEXT}]})
    client.post("/check", json={"messages": [{"text": BENIGN_TEXT}]})
    from api import app as module

    assert (id(module.REGISTRY), id(module.TABLE)) == before


def test_startup_validation_rejects_a_check_missing_from_the_weight_table() -> None:
    """一個註冊了但沒登錄於表的檢查，不能撐到第一個真實請求才炸。

    以測試用的組裝驗證，不依賴真的啟動一個 ASGI server ——
    行程啟動時執行的是同一個呼叫。
    """
    registry = build_registry()
    registry.register(_NeverRegisteredCheck())
    with pytest.raises(ValueError) as caught:
        load_weights().validate_against(registry)
    assert _NeverRegisteredCheck.name in str(caught.value)


def test_the_shipped_assembly_passes_its_own_validation() -> None:
    TABLE.validate_against(REGISTRY)


def test_domain_age_is_not_in_the_enabled_checks() -> None:
    """公開且不要求認證的端點，不注入會對外部服務發出查詢的解析器。"""
    assert DOMAIN_AGE not in {check.name for check in REGISTRY.enabled()}


def test_domain_age_absence_has_a_written_reason() -> None:
    reasons = dict(UNREGISTERED_CHECKS)
    assert reasons[DOMAIN_AGE] == "未注入外部查詢解析器"


def test_every_unregistered_check_carries_a_reason() -> None:
    for name, reason in UNREGISTERED_CHECKS:
        assert name and reason


def test_unregistered_checks_do_not_overlap_the_registry() -> None:
    """同一個名字不能同時「已註冊」與「刻意不註冊」—— 那兩張表就對不起來了。"""
    enabled = {check.name for check in REGISTRY.enabled()}
    assert enabled.isdisjoint({name for name, _ in UNREGISTERED_CHECKS})


def test_build_registry_returns_a_fresh_instance() -> None:
    first = build_registry()
    second = build_registry()
    assert first is not second
    assert isinstance(first, CheckRegistry)
    assert [check.name for check in first.enabled()] == [check.name for check in second.enabled()]


def test_blocklist_max_age_days_is_a_single_source() -> None:
    """載入時與健康檢查重算時用同一個數字，不在兩處各寫一個。"""
    assert isinstance(BLOCKLIST_MAX_AGE_DAYS, int)
    assert BLOCKLIST_MAX_AGE_DAYS > 0


def test_log_record_carries_no_message_text(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """MUST NOT 記錄請求 body 與回應的依據、建議。

    記的是延遲、`abstained`、`confidence` 與類型的成員名 —— 供日後從實際流量
    統計棄權率，而那組欄位一個字都不含訊息內容。
    """
    with caplog.at_level(logging.INFO, logger="api.app"):
        response = client.post("/check", json={"messages": [{"text": SCAM_TEXT}]})
    assert response.status_code == 200
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "abstained=" in logged
    assert SCAM_TEXT not in logged
    for line in response.json()["evidence"]:
        assert line not in logged
    for line in response.json()["actions"]:
        assert line not in logged


def test_failed_validation_does_not_log_the_body(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="api.app"):
        response = client.post("/check", json={"messages": [{"text": SCAM_TEXT, "from": ""}]})
    assert response.status_code == 422
    assert SCAM_TEXT not in "\n".join(record.getMessage() for record in caplog.records)


def test_openapi_documents_both_routes(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/check" in schema["paths"]
    assert "/health" in schema["paths"]


def test_openapi_carries_the_field_descriptions(client: TestClient) -> None:
    """說明 MUST 出現在對外的介面文件，因為 `/docs` 是寫 client 的人唯一會讀的
    東西 —— 寫在 design 裡的警告永遠到不了他們面前。"""
    schema = client.get("/openapi.json").json()
    properties = schema["components"]["schemas"]["CheckResponse"]["properties"]
    assert "原文片段" in properties["evidence"]["description"]
    assert "不是詐騙機率" in properties["confidence"]["description"]
