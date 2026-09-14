"""驗證失敗與未預期例外如何變成回應。

兩件事要守。第一件是**不回音違規值**：FastAPI 預設的 422 body 會把違規值原樣
放進 `input` 欄位，對 `text` 欄位那等於把整則訊息寫進錯誤回應。第二件是
**未預期例外不被包成 `ErrorResponse`** —— 那條路徑上的回應形狀不一致是
遵守「禁止 `except Exception`」這條硬性規範的直接代價，不是疏漏。
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.app import app
from api.errors import INVALID_REQUEST_CODE, field_path, validation_error_handler
from api.schema import CheckRequest

SECRET = "獨一無二的原文片段甲乙丙丁"

BOOM = "這是一個叫不出名字的失敗"


def _explode(payload: CheckRequest) -> None:
    """一個刻意在請求處理路徑上炸掉的 handler。"""
    raise RuntimeError(BOOM)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_field_validation_failure_is_a_422_error_response(client: TestClient) -> None:
    response = client.post("/check", json={"messages": []})
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "field", "message"}
    assert body["error"]["code"] == INVALID_REQUEST_CODE
    assert body["error"]["field"] == "messages"


def test_the_framework_default_structure_is_gone(client: TestClient) -> None:
    """`detail` 是框架預設的鍵，`input` 是它回音違規值的地方。兩者都不該出現。"""
    response = client.post("/check", json={"messages": [{"text": 12345}]})
    assert response.status_code == 422
    body = response.json()
    assert "detail" not in body
    assert "input" not in response.text


def test_wrong_typed_text_reports_the_path_but_not_the_value(client: TestClient) -> None:
    response = client.post("/check", json={"messages": [{"text": SECRET.encode().hex()}]})
    assert response.status_code == 200

    response = client.post("/check", json={"messages": [{"text": {"raw": SECRET}}]})
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "messages.0.text"
    assert SECRET not in response.text


def test_empty_sender_reports_the_fact_but_not_the_value(client: TestClient) -> None:
    response = client.post("/check", json={"messages": [{"text": "您好", "from": ""}]})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["field"] == "messages.0.from"
    assert "空字串" in body["error"]["message"]


def test_naive_timestamp_reports_the_string(client: TestClient) -> None:
    """時間戳不是訊息內容 —— 這是那張窮舉表上唯一被允許回音的一項。"""
    response = client.post(
        "/check", json={"messages": [{"text": "您好", "at": "2026-09-14T10:00:00"}]}
    )
    assert response.status_code == 422
    assert "2026-09-14T10:00:00" in response.json()["error"]["message"]


def test_unknown_field_reports_the_name_but_not_the_value(client: TestClient) -> None:
    response = client.post("/check", json={"messages": [{"text": "您好"}], "trace_id": SECRET})
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "trace_id"
    assert SECRET not in response.text


def test_invalid_json_is_a_422_with_a_null_field(client: TestClient) -> None:
    """語法錯誤與欄位錯誤走同一個 handler —— FastAPI 把 `json.JSONDecodeError`
    包成 `type: "json_invalid"` 的 `RequestValidationError`。"""
    response = client.post(
        "/check", content=b'{"messages": [', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["field"] is None
    assert body["error"]["code"] == INVALID_REQUEST_CODE


def test_invalid_json_does_not_echo_the_fragment_it_choked_on(client: TestClient) -> None:
    """`json_invalid` 的 `ctx` 會帶解析器看到的片段，而那是原文的一部分。"""
    response = client.post(
        "/check",
        content=f'{{"messages": [{{"text": "{SECRET}"'.encode(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert SECRET not in response.text


def test_field_path_drops_the_body_prefix() -> None:
    """線上的路徑是 `messages.0.at`，呼叫端送的 body 也是那個形狀。"""
    assert field_path(("body", "messages", 0, "at")) == "messages.0.at"
    assert field_path(("body",)) is None
    assert field_path(()) is None


def test_the_handler_refuses_exceptions_it_was_not_registered_for() -> None:
    """這個 handler 只註冊給 `RequestValidationError`。型別標註寫 `Exception`
    是 Starlette 對 handler 簽章的要求，不是一個攔截所有例外的處理器。"""
    with pytest.raises(TypeError):
        asyncio.run(validation_error_handler(None, RuntimeError(BOOM)))


def test_no_catch_all_exception_handler_is_registered() -> None:
    """`@app.exception_handler(Exception)` 在語法上不是 `try/except`，
    但它在效果上做的事完全一樣：攔下任何叫不出名字的例外，
    把它變成一個看起來正常的回應。"""
    assert Exception not in app.exception_handlers
    assert RequestValidationError in app.exception_handlers


def test_an_unnamed_exception_is_not_wrapped_in_an_error_response() -> None:
    """例外從 `detect()` 內部漏到這一層，代表某處違反了它自己宣告的協定 ——
    那是一個 bug，不是一個「本來就會發生」的已知狀態。

    用一個獨立的 app 驗證，不動正式的 `app`：那個 app 的例外處理器註冊表
    是被測對象之一，不該在測試裡被改。
    """
    exploding = FastAPI(debug=False)
    exploding.post("/check")(_explode)
    client = TestClient(exploding, raise_server_exceptions=False)

    response = client.post("/check", json={"messages": [{"text": SECRET}]})

    assert response.status_code == 500
    assert "error" not in response.text
    assert BOOM not in response.text
    assert SECRET not in response.text
    assert "Traceback" not in response.text
