"""請求體的位元組上限。

這些測試看起來像在測一個數字，實際上測的是**上限在哪裡生效**：
在路由與 pydantic 之前、在 body 被完整讀進記憶體之前、
而且不看 `Content-Length`。

以 `asyncio.run()` 直接驅動中介層，而不是依賴一個 async 測試外掛：
本專案的 pytest 設定沒有註冊任何 async 模式，而為了四條測試引進一個外掛
會讓「測試怎麼跑」多一個沒有人記得的設定。
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.limits import MAX_BODY_BYTES, BodySizeLimitMiddleware, TOO_LARGE_CODE

CHUNK = b"x" * 65_536


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class _RecordingReceive:
    """記下 `receive()` 被呼叫了幾次，用來證明超限之後不再繼續讀。"""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.calls = 0

    async def __call__(self) -> dict[str, object]:
        self.calls += 1
        if not self._chunks:
            return {"type": "http.request", "body": b"", "more_body": False}
        body = self._chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(self._chunks)}


class _RecordingSend:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def __call__(self, message: dict[str, object]) -> None:
        self.messages.append(message)


class _NeverCalledApp:
    """下游 app。超限時它一次都不該被呼叫。"""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, scope: object, receive: object, send: object) -> None:
        self.called = True


def _body_of(send: _RecordingSend) -> dict[str, object]:
    for message in send.messages:
        if message["type"] == "http.response.body":
            return json.loads(message["body"])
    raise AssertionError(f"沒有回應 body：{send.messages}")


def _status_of(send: _RecordingSend) -> int:
    for message in send.messages:
        if message["type"] == "http.response.start":
            return int(message["status"])
    raise AssertionError(f"沒有回應狀態碼：{send.messages}")


def test_the_limit_is_one_mebibyte() -> None:
    assert MAX_BODY_BYTES == 1_048_576


def test_oversized_body_gets_a_413_error_response(client: TestClient) -> None:
    response = client.post(
        "/check",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == TOO_LARGE_CODE
    assert body["error"]["field"] is None


def test_413_message_states_the_limit_but_not_what_was_received(client: TestClient) -> None:
    """上限值可以說，已收到的位元組數不說 —— 那個數字不敏感，
    但省略它讓這條訊息與其他錯誤保持同樣的最小揭露原則。"""
    sent = MAX_BODY_BYTES + 12_345
    response = client.post(
        "/check", content=b"x" * sent, headers={"content-type": "application/json"}
    )
    message = response.json()["error"]["message"]
    assert str(MAX_BODY_BYTES) in message
    assert str(sent) not in message


def test_a_body_within_the_limit_is_untouched(client: TestClient) -> None:
    payload = {"messages": [{"text": "明天見。"}]}
    assert len(json.dumps(payload).encode("utf-8")) < MAX_BODY_BYTES
    response = client.post("/check", json=payload)
    assert response.status_code == 200


def test_a_large_but_legal_body_still_gets_through(client: TestClient) -> None:
    """合法的最壞情境（100 則長訊息）必須通過 —— 上限擋的是惡意，不是長對話。"""
    payload = {"messages": [{"text": "詐" * 2_438} for _ in range(100)]}
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert len(encoded) < MAX_BODY_BYTES
    response = client.post("/check", content=encoded, headers={"content-type": "application/json"})
    assert response.status_code == 200


def test_reading_stops_the_moment_the_threshold_is_crossed() -> None:
    """超限之後 MUST NOT 繼續接收或緩衝剩餘的位元組。

    送 32 塊 64 KiB（共 2 MiB），上限在第 17 塊被跨過 —— 中介層應該在那裡
    就停手，而不是把剩下的 15 塊也讀完。
    """
    downstream = _NeverCalledApp()
    middleware = BodySizeLimitMiddleware(downstream)
    receive = _RecordingReceive([CHUNK] * 32)
    send = _RecordingSend()

    asyncio.run(middleware({"type": "http"}, receive, send))

    assert receive.calls == 17
    assert downstream.called is False
    assert _status_of(send) == 413


def test_the_limit_does_not_rely_on_content_length() -> None:
    """分塊傳輸編碼下 `Content-Length` 可能根本不存在，而且它可以被偽造。

    這個 scope 的 headers 完全沒有 `content-length`，位元組數仍被算出來。
    """
    downstream = _NeverCalledApp()
    middleware = BodySizeLimitMiddleware(downstream)
    receive = _RecordingReceive([CHUNK] * 32)
    send = _RecordingSend()

    asyncio.run(middleware({"type": "http", "headers": []}, receive, send))

    assert _status_of(send) == 413
    assert _body_of(send)["error"]["code"] == TOO_LARGE_CODE


def test_a_lying_content_length_does_not_let_a_large_body_through() -> None:
    downstream = _NeverCalledApp()
    middleware = BodySizeLimitMiddleware(downstream)
    receive = _RecordingReceive([CHUNK] * 32)
    send = _RecordingSend()

    asyncio.run(
        middleware({"type": "http", "headers": [(b"content-length", b"10")]}, receive, send)
    )

    assert _status_of(send) == 413


def test_non_http_scopes_pass_straight_through() -> None:
    """lifespan 與 websocket 有自己的 `receive` 協定，本層不碰它們。"""
    downstream = _NeverCalledApp()
    middleware = BodySizeLimitMiddleware(downstream)
    asyncio.run(middleware({"type": "lifespan"}, _RecordingReceive([]), _RecordingSend()))
    assert downstream.called is True


def test_the_limit_runs_before_routing_and_validation(client: TestClient) -> None:
    """超限的請求打一條不存在的路徑仍然回 413，而不是 404 或 422 ——
    證明這一層在路由與 pydantic 之前。"""
    response = client.post(
        "/no-such-route",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_an_oversized_body_that_is_also_invalid_json_is_still_a_413(
    client: TestClient,
) -> None:
    response = client.post(
        "/check",
        content=b"{not json" + b"x" * MAX_BODY_BYTES,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
