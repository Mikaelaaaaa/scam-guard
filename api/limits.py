"""請求體的位元組上限 —— 在 ASGI 的 `receive()` 通道上邊讀邊計數。

**為什麼不能交給 pydantic。** pydantic 的驗證發生在 body 已經被完整讀進記憶體
**並且已經解析成 Python 物件之後**。一個惡意的單則超大訊息（例如一個 100 MB
的 `text` 字串）會在驗證失敗**之前**就被完整緩衝、完整解析成字串物件 ——
傷害在那個時候已經發生了。

**為什麼不能交給偵測核心的雙上限。** `add-context-limits` 的 100 則／50,000
字元只作用於前文，`latest`（待判定的最後一則）永遠完整保留。一個惡意單則若是
`latest`，它不會被丟棄，會整則被送進 `normalize_text()` 逐字元處理。
那個 change 的 design 自己指名了這個缺口：「惡意使用者仍可送出單則一百萬字元的
訊息。那是 HTTP 層的請求大小限制該擋的（`detect-api`），不是這一層」——
本模組就是那個交棒的接手處。

**為什麼不看 `Content-Length`。** 它可以被偽造，而且分塊傳輸編碼下可能根本
不存在。判斷 MUST 以實際接收到的位元組數為準。

上限以外的代價寫清楚：本中介層會把請求體讀完才交給下游，因此串流請求體不再
可行。目前沒有這種端點（`detect()` 是同步呼叫），而緩衝量以 `MAX_BODY_BYTES`
封頂 —— 下游本來就會把整個 body 讀進記憶體，緩衝的位置從路由層搬到這裡，
總量沒有變多。
"""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.schema import ErrorDetail, ErrorResponse

MAX_BODY_BYTES = 1_048_576
"""1 MiB。

推導依據是 `add-context-limits` 量測過的 Cofacts 詐騙訊息長度分佈
（p90 315 字元、p99 1,008 字元、最長 2,438 字元）。一個**合法的最壞情境**是
100 則（核心的則數上限）× 2,438 字元 × 3 位元組（中文的 UTF-8 編碼長度）
≈ 731 KB 的純文字，加上每則約 80 位元組的 JSON 結構約 8 KB，合計約 739 KB。
1 MiB 留了約 40% 餘裕。

**這個數字沒有實驗依據**，與 `Limits` 的 100 則／50,000 字元同性質 ——
是從量測分佈推出的起點，不是調過的值。它是一個部署參數，有真實流量的位元組
分佈之後應該回來調整。
"""

TOO_LARGE_CODE = "request_too_large"

CONTENT_TYPE_JSON = b"application/json"


class _BufferedReceive:
    """先交出已經讀完的訊息，之後把 `receive()` 還給原本的通道。

    中介層把 body 讀完才呼叫下游，下游仍然要能呼叫 `receive()` 拿到它 ——
    這個物件就是那份重播。用類別而不是閉包：專案的硬性規範禁止巢狀 `def`，
    而「要捕捉狀態」正是改用一個小類別的信號。
    """

    def __init__(self, pending: list[Message], receive: Receive) -> None:
        self._pending = pending
        self._receive = receive

    async def __call__(self) -> Message:
        if self._pending:
            return self._pending.pop(0)
        return await self._receive()


async def send_request_too_large(send: Send, max_bytes: int) -> None:
    """回一個 413，body 為 `add-api-schema` 定義的單一錯誤形狀。

    訊息陳述上限值，**不含**呼叫端已經送出的位元組數：那個數字本身不敏感，
    但省略它讓這條訊息與其他錯誤保持同樣的最小揭露原則，沒有理由破例。
    """
    payload = ErrorResponse(
        error=ErrorDetail(
            code=TOO_LARGE_CODE,
            field=None,
            message=f"請求體超過上限 {max_bytes} 位元組。",
        )
    ).model_dump_json()
    body = payload.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", CONTENT_TYPE_JSON),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    """純 ASGI 中介層，掛在路由與驗證之前。

    寫成 ASGI 中介層而不是 `BaseHTTPMiddleware` 的子類別，理由是後者只給得到
    一個已經包裝好的 `Request`，拿不到 `receive()` 這個通道本身 ——
    而本模組要做的事就發生在那個通道上。
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        chunks: list[bytes] = []
        received = 0
        pending: list[Message] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                # `http.disconnect`：呼叫端斷線。原樣排進重播佇列交給下游，
                # 由它決定怎麼處置 —— 那不是本層的職責。
                pending.append(message)
                break
            # ASGI 規格定 `body` 為選填且預設 `b""`，「缺席」與「空」在這裡
            # 真的同義，這是 `.get(key, default)` 合法的那一種情況。
            chunk = message.get("body", b"")
            received += len(chunk)
            if received > self._max_bytes:
                # 立即中止：不再呼叫 `receive()`，剩餘的位元組不進記憶體。
                await send_request_too_large(send, self._max_bytes)
                return
            chunks.append(chunk)
            more_body = message.get("more_body", False)

        replay: list[Message] = [
            {"type": "http.request", "body": b"".join(chunks), "more_body": False}
        ]
        replay.extend(pending)
        await self._app(scope, _BufferedReceive(replay, receive), send)
