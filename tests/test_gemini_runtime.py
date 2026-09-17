"""Gemini runtime 的離線測試 —— 不打網路。實際 API 呼叫由手動驗證涵蓋。"""

import pytest

from llm_runtime.gemini import (
    GeminiCallFailed,
    GeminiRuntime,
    GeminiUnavailable,
    _extract_text,
    _response_schema,
)
from scam_guard.llm.schema import FIELD_NAMES, LABELS
from scam_guard.types import ScamType


def test_missing_key_raises_on_construction() -> None:
    with pytest.raises(GeminiUnavailable):
        GeminiRuntime(api_key="")


def test_explicit_key_constructs() -> None:
    runtime = GeminiRuntime(api_key="test-key", model="gemini-3.6-flash")
    assert runtime._model == "gemini-3.6-flash"


def test_response_schema_forces_nested_int_coords() -> None:
    """座標宣告成 array-of-array-of-integer —— 那是修掉 Gemini 回字串 `"[0,0]"` 的關鍵。"""
    schema = _response_schema()
    coords = schema["properties"]["evidence_sentence_ids"]
    assert coords["type"] == "ARRAY"
    assert coords["items"]["type"] == "ARRAY"
    assert coords["items"]["items"]["type"] == "INTEGER"


def test_response_schema_field_order_and_enums() -> None:
    schema = _response_schema()
    assert schema["propertyOrdering"] == list(FIELD_NAMES)
    assert schema["properties"]["label"]["enum"] == list(LABELS)
    assert schema["properties"]["category_165"]["enum"] == [t.value for t in ScamType]


def test_extract_text_pulls_candidate_part() -> None:
    payload = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
    assert _extract_text(payload) == "hello"


def test_extract_text_empty_candidates_raises_not_empty_string() -> None:
    """API 沒給東西時拋 GeminiCallFailed，不回空字串 —— 空字串會被誤記成模型格式錯。"""
    with pytest.raises(GeminiCallFailed):
        _extract_text({"candidates": []})


def test_extract_text_missing_parts_raises() -> None:
    with pytest.raises(GeminiCallFailed):
        _extract_text({"candidates": [{"content": {}}]})
