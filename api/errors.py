"""驗證失敗轉換成 `add-api-schema` 的單一錯誤形狀。

**為什麼要換掉框架預設的 422 body。** FastAPI 預設回的是

```json
{"detail": [{"type": "...", "loc": ["body","messages",0,"text"],
             "msg": "...", "input": "<違規值>"}]}
```

`input` 會把違規值**原樣**放進回應。對 `text` 欄位，那個違規值就是整則訊息。
這條路徑不在 `add-redact-apply` 盤點的五條 log 洩漏路徑上，因為那張表盤點的是
log；但它的性質完全一樣 —— 一份含原文的 JSON，送給一個很可能會記錄錯誤回應的
呼叫端。因此本模組只讀 `type`、`loc` 與 `msg`，**一行都不讀 `input` 或 `ctx`**。

**為什麼沒有攔截所有例外的處理器。** 見 `api/app.py` 的說明；那是一個違反直覺
的決定，理由寫在那裡而不是這裡，因為它是「不寫什麼」而不是「寫了什麼」。
"""

from collections.abc import Sequence

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schema import ErrorDetail, ErrorResponse

INVALID_REQUEST_CODE = "invalid_request"

JSON_INVALID_TYPE = "json_invalid"
"""FastAPI 把 body 的 `json.JSONDecodeError` 包成這個 `type` 的
`RequestValidationError` —— 所以語法錯誤與欄位錯誤走同一個 handler，
不需要為語法錯誤另開一條分支。"""

BODY_LOC = "body"
"""FastAPI 在欄位路徑前面加的一層。線上的路徑是 `messages.0.at`，
呼叫端送的 body 也是那個形狀，多一層 `body` 只會讓路徑對不上請求。"""

JSON_INVALID_MESSAGE = "請求體不是合法的 JSON。"
"""語法錯誤的說明**自己寫**而不是轉述 pydantic 的 `msg`：
`json_invalid` 的 `ctx` 會帶解析器看到的片段，而那是原文的一部分。"""


def field_path(loc: Sequence[object]) -> str | None:
    """把 pydantic 的 `loc` 轉成線上的欄位路徑。不指向任何欄位時回 `None`。"""
    parts = list(loc)
    if parts and parts[0] == BODY_LOC:
        parts = parts[1:]
    if not parts:
        return None
    return ".".join(str(part) for part in parts)


def to_error_detail(error: dict[str, object]) -> ErrorDetail:
    """單一筆 pydantic 錯誤 → `ErrorDetail`。只讀 `type`、`loc` 與 `msg`。"""
    error_type = error["type"]
    if error_type == JSON_INVALID_TYPE:
        # `loc` 在這裡是 `("body", <位元組位置>)`，那不是欄位路徑。
        return ErrorDetail(code=INVALID_REQUEST_CODE, field=None, message=JSON_INVALID_MESSAGE)
    loc = error["loc"]
    if not isinstance(loc, Sequence) or isinstance(loc, str):
        raise TypeError(
            f"pydantic 錯誤的 loc 必須為序列，收到 {type(loc).__name__}：{error_type!r}"
        )
    return ErrorDetail(
        code=INVALID_REQUEST_CODE,
        field=field_path(loc),
        message=str(error["msg"]),
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """`RequestValidationError` → 422 加單一錯誤形狀。

    **只回報第一筆錯誤。** 契約定的 `error` 是一個物件而不是陣列，
    要逐一回報全部錯誤就得改形狀，而那是 `add-api-schema` 的決定不是本層的。
    第一筆足以讓呼叫端修一次再送一次。

    型別標註寫 `Exception` 是 Starlette 對 handler 簽章的要求（它的
    `ExceptionHandler` 協定如此），**不是**一個攔截所有例外的處理器 ——
    這個 handler 只註冊給 `RequestValidationError` 一個型別。
    """
    if not isinstance(exc, RequestValidationError):
        raise TypeError(f"此 handler 只處理 RequestValidationError，收到 {type(exc).__name__}")
    errors = exc.errors()
    if not errors:
        raise ValueError("RequestValidationError 不含任何錯誤，無法組出欄位路徑")
    body = ErrorResponse(error=to_error_detail(dict(errors[0])))
    return JSONResponse(status_code=422, content=body.model_dump())
