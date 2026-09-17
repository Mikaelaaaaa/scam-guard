"""LINE Messaging API webhook adapter —— 第四個介面層。

收 LINE 訊息 → `detect()`（四層，含 Gemini 語意層）→ 把判定回覆給使用者。
跑在本機，由 cloudflared tunnel 暴露成公開 HTTPS 給 LINE 打。

啟動前必須設環境變數（都在 `.env`，gitignored）：
    LINE_CHANNEL_SECRET        驗簽用
    LINE_CHANNEL_ACCESS_TOKEN  回覆用
    GEMINI_API_KEY             語意層（缺席則降級為三層）

    uvicorn clients.line.app:app --port 8000
    cloudflared tunnel --url http://localhost:8000
LINE console 的 Webhook URL 填 https://<tunnel>/callback，按 Verify。
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import urllib.error
import urllib.request

from fastapi import FastAPI, Request, Response

from clients.line.registry import LLM_ENABLED, REGISTRY, TABLE
from clients.line.render import render_verdict
from llm_runtime.gemini import GeminiCallFailed
from scam_guard.normalize import DEFAULT_LIMITS
from scam_guard.pipeline import detect
from scam_guard.types import Message
from scam_guard.types import Request as ScamRequest

logger = logging.getLogger("scam_guard.line")

REPLY_ENDPOINT = "https://api.line.me/v2/bot/message/reply"
UNSUPPORTED = "我只看得懂文字訊息，請把你收到的可疑訊息用文字貼給我。"
LLM_FAILED = "語意判讀失敗，本次無法完成判定，請稍後再試。"
"""語意層（必備）失敗時的回覆。使用者要求：失敗就寫失敗，不給只有規則的降級結果。"""

app = FastAPI(title="scam-guard LINE adapter")


def _channel_secret() -> bytes:
    secret = os.environ.get("LINE_CHANNEL_SECRET")
    if not secret:
        raise RuntimeError("LINE_CHANNEL_SECRET 未設 —— 無法驗簽，webhook 不可運作。")
    return secret.encode("utf-8")


def _signature_ok(body: bytes, signature: str) -> bool:
    """X-Line-Signature 是 channel secret 對 request body 的 HMAC-SHA256 base64。

    用 `hmac.compare_digest` 而不是 `==` —— 防時序攻擊。
    """
    digest = hmac.new(_channel_secret(), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


def _reply(reply_token: str, text: str) -> None:
    """用 reply API 回覆。reply 不吃免費方案額度、無上限，所以一律用它、不用 push。"""
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN 未設 —— 無法回覆。")
    body = json.dumps(
        {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        REPLY_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except urllib.error.HTTPError as error:
        logger.warning("LINE reply 失敗 HTTP %s：%s", error.code, error.read()[:200])
    except urllib.error.URLError as error:
        logger.warning("LINE reply 連線失敗：%s", error.reason)


def _judge(text: str) -> str:
    """一則文字訊息 → 判定 → 回覆字串。

    語意層（Gemini，必備）失敗時回 `LLM_FAILED` —— 不優雅降級成只有規則層的結果。
    只接 `GeminiCallFailed`（具體型別），其餘例外照常傳播。
    """
    request = ScamRequest(messages=[Message(text=text)])
    try:
        verdict = detect(request, REGISTRY, TABLE, limits=DEFAULT_LIMITS)
    except GeminiCallFailed as error:
        logger.warning("語意層失敗，回覆失敗訊息：%s", error)
        return LLM_FAILED
    return render_verdict(verdict)


@app.get("/")
def health() -> dict[str, object]:
    """健康檢查，也讓部署有東西打。`llm` 欄位讓人看得出語意層有沒有掛。"""
    return {"service": "scam-guard-line", "layers": len(REGISTRY.enabled()), "llm": LLM_ENABLED}


@app.post("/callback")
async def callback(request: Request) -> Response:
    """LINE webhook 入口。先驗簽（驗不過 400），再逐事件處理。

    LINE 設定 webhook 時送一個空的驗證事件（`events` 為空），此時照常回 200 ——
    不能因為沒有訊息就報錯，否則 console 的 Verify 過不了。
    """
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    if not _signature_ok(body, signature):
        return Response(status_code=400, content="bad signature")
    payload = json.loads(body)
    for event in payload.get("events", []):
        if event.get("type") != "message":
            continue
        message = event.get("message", {})
        reply_token = event.get("replyToken")
        if not reply_token:
            continue
        if message.get("type") == "text":
            _reply(reply_token, _judge(message["text"]))
        else:
            _reply(reply_token, UNSUPPORTED)
    return Response(status_code=200, content="ok")
