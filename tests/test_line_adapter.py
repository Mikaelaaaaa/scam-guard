"""LINE adapter 的離線測試 —— 驗簽與健康檢查，不打 LINE 也不打 Gemini。

實際判定與回覆由手動驗證涵蓋（需 LINE channel + Gemini key）。這裡守住 webhook
的安全性質：驗簽正確、偽造被擋、驗證事件回 200。
"""

import base64
import hashlib
import hmac
import json
import os

import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("LINE_CHANNEL_SECRET", "test-secret-for-offline")

from fastapi.testclient import TestClient  # noqa: E402

from clients.line.app import app  # noqa: E402

client = TestClient(app)
SECRET = os.environ["LINE_CHANNEL_SECRET"].encode("utf-8")


def _sign(body: bytes) -> str:
    return base64.b64encode(hmac.new(SECRET, body, hashlib.sha256).digest()).decode("ascii")


def test_health_reports_layers() -> None:
    payload = client.get("/").json()
    assert payload["service"] == "scam-guard-line"
    assert payload["layers"] >= 1


def test_empty_verify_event_returns_200() -> None:
    """LINE 設 webhook 時送空 events —— 必須回 200，否則 console 的 Verify 過不了。"""
    body = json.dumps({"events": []}).encode("utf-8")
    response = client.post("/callback", content=body, headers={"X-Line-Signature": _sign(body)})
    assert response.status_code == 200


def test_bad_signature_rejected() -> None:
    body = json.dumps({"events": []}).encode("utf-8")
    response = client.post("/callback", content=body, headers={"X-Line-Signature": "forged"})
    assert response.status_code == 400


def test_missing_signature_rejected() -> None:
    body = json.dumps({"events": []}).encode("utf-8")
    response = client.post("/callback", content=body)
    assert response.status_code == 400
