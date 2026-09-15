"""`POST /check` 的請求、回應與錯誤契約。

本檔測的是**形狀與邊界**，不測任何一個真實判定的內容：路由、狀態碼與
`detect()` 的接線屬於 `add-api-server`。最重要的幾條測試看起來像在測欄位名，
實際上守的是「原文離開系統的形狀只有一種」——回應模型是顯式的一層，
`Verdict` 加欄位時它不會跟著多吐。
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from api.schema import (
    KNOWN_VERDICT_FIELDS,
    CheckRequest,
    CheckResponse,
    ErrorDetail,
    ErrorResponse,
    MessageIn,
    ScamTypeOut,
    scam_type_out,
    verdict_to_response,
)
from scam_guard.normalize import build_document
from scam_guard.redact import redact_document
from scam_guard.types import CheckResult, Message, Request, ScamType, Verdict

SECRET = "獨一無二的原文片段甲乙丙丁"
"""刻意選一個不會出現在任何欄位名、型別名或標點裡的字串，用來搜尋洩漏。"""


def _locs(error: ValidationError) -> list[tuple[object, ...]]:
    return [item["loc"] for item in error.errors()]


def _messages(error: ValidationError) -> str:
    """全部錯誤的人可讀訊息串起來 —— 這是本層會送給呼叫端的部分。

    `errors()` 的 `input` 欄位刻意不納入：那正是 `api/errors.py` 要丟掉的東西，
    而「丟掉了沒有」是那一層的測試。
    """
    return "\n".join(item["msg"] for item in error.errors())


def _verdict(
    *,
    scam_probability: float | None = 0.87,
    confidence: float = 0.9,
    scam_type: ScamType | None = ScamType.FAKE_AUTHORITY,
    evidence: list[str] | None = None,
    actions: list[str] | None = None,
    checks: list[CheckResult] | None = None,
) -> Verdict:
    doc = build_document([Message(text=f"{SECRET}，請立刻到 ATM 操作。")])
    return Verdict(
        scam_probability=scam_probability,
        confidence=confidence,
        scam_type=scam_type,
        evidence=[] if evidence is None else evidence,
        actions=[] if actions is None else actions,
        checks=[] if checks is None else checks,
        redacted=redact_document(doc),
    )


# ---------------------------------------------------------------------------
# messages：至少一則
# ---------------------------------------------------------------------------


def test_missing_messages_is_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({})
    assert ("messages",) in _locs(caught.value)


def test_null_messages_is_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": None})
    assert ("messages",) in _locs(caught.value)


def test_empty_messages_is_rejected_and_reports_the_length() -> None:
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": []})
    assert ("messages",) in _locs(caught.value)
    assert "0" in _messages(caught.value)


def test_single_message_passes() -> None:
    request = CheckRequest.model_validate({"messages": [{"text": "明天見"}]})
    assert len(request.messages) == 1


def test_multiple_messages_keep_their_order() -> None:
    request = CheckRequest.model_validate(
        {"messages": [{"text": "第一則"}, {"text": "第二則"}, {"text": "最後一則"}]}
    )
    assert [message.text for message in request.messages] == ["第一則", "第二則", "最後一則"]


# ---------------------------------------------------------------------------
# 線上欄位名：text / from / at
# ---------------------------------------------------------------------------


def test_wire_names_map_to_message_attributes() -> None:
    message = MessageIn.model_validate(
        {"text": "您好", "from": "them", "at": "2026-09-14T10:00:00Z"}
    )
    assert message.text == "您好"
    assert message.sender == "them"
    assert message.sent_at == datetime(2026, 9, 14, 10, 0, tzinfo=UTC)


def test_schema_exposes_from_and_at_not_sender_and_sent_at() -> None:
    properties = MessageIn.model_json_schema(by_alias=True)["properties"]
    assert set(properties) == {"text", "from", "at"}


def test_internal_attribute_names_are_rejected_as_unknown_fields() -> None:
    """送 `sender` 的呼叫端必須在第一次呼叫就知道自己送錯了。

    `extra="forbid"` 之外的任何選擇都會讓 `sender` 被安靜丟掉、屬性取 `None`，
    請求合法、回應正常，而軌跡檢查永遠讀不到發送者。
    """
    with pytest.raises(ValidationError) as caught:
        MessageIn.model_validate({"text": "您好", "sender": "them"})
    assert ("sender",) in _locs(caught.value)

    with pytest.raises(ValidationError) as caught:
        MessageIn.model_validate({"text": "您好", "sent_at": "2026-09-14T10:00:00Z"})
    assert ("sent_at",) in _locs(caught.value)


def test_unknown_top_level_field_is_rejected_without_echoing_its_value() -> None:
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": [{"text": "您好"}], "trace_id": SECRET})
    assert ("trace_id",) in _locs(caught.value)
    assert SECRET not in _messages(caught.value)


def test_unknown_message_level_field_is_rejected_without_echoing_its_value() -> None:
    with pytest.raises(ValidationError) as caught:
        MessageIn.model_validate({"text": "您好", "channel": SECRET})
    assert ("channel",) in _locs(caught.value)
    assert SECRET not in _messages(caught.value)


# ---------------------------------------------------------------------------
# text：必填，空字串合法
# ---------------------------------------------------------------------------


def test_empty_text_is_accepted() -> None:
    """貼圖、純圖片、只有空白的訊息不是壞掉的請求。

    偵測核心對「正規化結果為空」有明確定義的處置（不中斷流程），
    在 API 這一層拒絕它等於發明一條核心沒有的規則。
    """
    assert MessageIn.model_validate({"text": ""}).text == ""


def test_all_messages_empty_text_is_accepted() -> None:
    request = CheckRequest.model_validate({"messages": [{"text": ""}, {"text": ""}]})
    assert [message.text for message in request.messages] == ["", ""]


def test_missing_text_is_rejected_with_a_field_path() -> None:
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": [{"from": "them"}]})
    assert ("messages", 0, "text") in _locs(caught.value)


def test_null_text_is_rejected_with_a_field_path() -> None:
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": [{"text": None}]})
    assert ("messages", 0, "text") in _locs(caught.value)


def test_wrong_typed_text_does_not_echo_the_received_value() -> None:
    """錯的是型別，而能說明這件事的只需要欄位路徑與期望型別。"""
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": [{"text": 12345}]})
    assert ("messages", 0, "text") in _locs(caught.value)
    assert "12345" not in _messages(caught.value)


# ---------------------------------------------------------------------------
# from：缺席與 null 同義，空字串被拒絕
# ---------------------------------------------------------------------------


def test_missing_from_and_null_from_are_the_same_thing() -> None:
    omitted = MessageIn.model_validate({"text": "您好"})
    explicit_null = MessageIn.model_validate({"text": "您好", "from": None})
    assert omitted.sender is None
    assert explicit_null.sender is None
    assert omitted == explicit_null


def test_empty_from_is_rejected() -> None:
    """空字串不是未知 —— 接受它會讓所有這種訊息被歸成同一個發送者。"""
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": [{"text": "您好", "from": ""}]})
    assert ("messages", 0, "from") in _locs(caught.value)
    assert "空字串" in _messages(caught.value)


def test_from_validation_error_does_not_echo_the_value() -> None:
    """`Message.sender` 允許保留暱稱，所以它可能是一個真實人名。

    本層對 `from` 定義的違規只有「空字串」與「型別不是字串」兩種，
    兩者都能在不引用值的情況下描述完整 —— 不回音沒有損失任何可診斷性。
    """
    with pytest.raises(ValidationError) as caught:
        MessageIn.model_validate({"text": "您好", "from": [SECRET]})
    assert SECRET not in _messages(caught.value)


# ---------------------------------------------------------------------------
# at：缺席與 null 同義，必須帶時區
# ---------------------------------------------------------------------------


def test_missing_at_and_null_at_are_the_same_thing() -> None:
    omitted = MessageIn.model_validate({"text": "您好"})
    explicit_null = MessageIn.model_validate({"text": "您好", "at": None})
    assert omitted.sent_at is None
    assert explicit_null.sent_at is None
    assert omitted == explicit_null


def test_aware_timestamp_is_accepted() -> None:
    message = MessageIn.model_validate({"text": "您好", "at": "2026-09-14T10:00:00Z"})
    assert message.sent_at is not None
    assert message.sent_at.tzinfo is not None


def test_naive_timestamp_is_rejected_and_the_string_is_reported() -> None:
    """naive datetime 的失敗本來會發生在一個 `Check` 內部、一個真實請求的半途。

    時間戳不是訊息內容，回音它是這張窮舉表上唯一被允許的一項。
    """
    with pytest.raises(ValidationError) as caught:
        CheckRequest.model_validate({"messages": [{"text": "您好", "at": "2026-09-14T10:00:00"}]})
    assert ("messages", 0, "at") in _locs(caught.value)
    assert "2026-09-14T10:00:00" in _messages(caught.value)


def test_naive_timestamp_is_not_given_a_default_timezone() -> None:
    with pytest.raises(ValidationError):
        MessageIn.model_validate({"text": "您好", "at": datetime(2026, 9, 14, 10, 0)})


# ---------------------------------------------------------------------------
# 請求層不施加內容長度或順序限制
# ---------------------------------------------------------------------------


def test_request_layer_imposes_no_length_or_count_limits() -> None:
    """位元組上限屬 HTTP 服務層，則數與字元上限屬偵測核心。本層一個都不設。"""
    request = CheckRequest.model_validate(
        {"messages": [{"text": "長" * 60_000} for _ in range(200)]}
    )
    assert len(request.messages) == 200
    assert len(request.messages[0].text) == 60_000


def test_request_layer_does_not_require_increasing_timestamps() -> None:
    """轉傳的對話可能夾雜引用，而 LINE 的轉傳格式本來就拿不到原始時間。"""
    request = CheckRequest.model_validate(
        {
            "messages": [
                {"text": "第一則", "at": "2026-09-14T12:00:00Z"},
                {"text": "第二則", "at": "2026-09-14T09:00:00Z"},
            ]
        }
    )
    assert len(request.messages) == 2


# ---------------------------------------------------------------------------
# 回應是顯式模型
# ---------------------------------------------------------------------------


def test_response_has_exactly_six_fields() -> None:
    assert set(CheckResponse.model_fields) == {
        "scam_probability",
        "abstained",
        "confidence",
        "scam_type",
        "evidence",
        "actions",
    }


def test_verdict_field_set_is_the_one_this_module_has_seen() -> None:
    """`Verdict` 加欄位時本層的輸出不會跟著改變 —— 那是設計目的。

    代價是沒有任何機制會提醒「有一個新欄位還沒有人決定它進不進回應」。
    這條測試就是那個提醒：它不強迫你把新欄位放進回應，只強迫你看過它一眼。
    """
    actual = set(Verdict.__dataclass_fields__)
    assert actual == set(KNOWN_VERDICT_FIELDS), (
        f"Verdict 的欄位集合變了，請決定新欄位進不進回應後更新 KNOWN_VERDICT_FIELDS："
        f"新增 {sorted(actual - KNOWN_VERDICT_FIELDS)}、"
        f"消失 {sorted(KNOWN_VERDICT_FIELDS - actual)}"
    )


def test_response_carries_nothing_from_checks_or_redacted() -> None:
    """座標送了也解不對，逐項檢查記錄沒有跨 HTTP 的消費者，投影明文禁止進回應。"""
    verdict = _verdict(
        checks=[
            CheckResult(
                name="atm_operation",
                hit=True,
                detail="命中 ATM 操作指示",
                evidence=[(0, 1)],
                scam_types=[ScamType.FAKE_AUTHORITY],
                hard=True,
            )
        ],
        evidence=["要求你到 ATM 操作"],
        actions=["撥打 165 查證"],
    )
    payload = verdict_to_response(verdict).model_dump_json()

    assert "atm_operation" not in payload
    assert "命中 ATM 操作指示" not in payload
    assert "hard" not in payload
    for sentence in verdict.redacted.sentences:
        assert sentence not in payload
    for coord in verdict.redacted.coords:
        assert str(coord[0]) + ", " + str(coord[1]) not in payload
    assert "[[0,1]]" not in payload


def test_evidence_and_actions_pass_through_verbatim() -> None:
    evidence = [
        "此訊息要求你把驗證碼傳給對方：「請把簡訊收到的六位數字回傳給我」",
        "索取驗證碼（訊息內已含 6 位數字 482913，降級）",
        "命中 165 涉詐網站清單",
    ]
    actions = ["不要提供驗證碼", "撥打 165 反詐騙專線查證"]
    response = verdict_to_response(_verdict(evidence=evidence, actions=actions))
    assert response.evidence == evidence
    assert response.actions == actions


def test_empty_evidence_and_actions_are_not_filled_in() -> None:
    response = verdict_to_response(_verdict(evidence=[], actions=[]))
    assert response.evidence == []
    assert response.actions == []


# ---------------------------------------------------------------------------
# 拒答
# ---------------------------------------------------------------------------


def test_abstention_sets_the_explicit_flag() -> None:
    response = verdict_to_response(_verdict(scam_probability=None, confidence=0.3))
    assert response.scam_probability is None
    assert response.abstained is True


def test_a_decided_verdict_does_not_set_the_flag() -> None:
    response = verdict_to_response(_verdict(scam_probability=0.87))
    assert response.scam_probability == 0.87
    assert response.abstained is False


def test_decoupling_the_flag_from_the_probability_raises() -> None:
    """兩個欄位表達一件事，就必須在建構時綁死。"""
    with pytest.raises(ValidationError) as caught:
        CheckResponse(
            scam_probability=None,
            abstained=False,
            confidence=0.3,
            scam_type=None,
            evidence=[],
            actions=[],
        )
    message = _messages(caught.value)
    assert "abstained=False" in message
    assert "scam_probability=None" in message


def test_abstention_still_carries_evidence() -> None:
    """拒答是一次成功的推論，依據照常傳遞。"""
    response = verdict_to_response(
        _verdict(scam_probability=None, confidence=0.3, evidence=["依據一", "依據二"])
    )
    assert response.abstained is True
    assert response.evidence == ["依據一", "依據二"]


# ---------------------------------------------------------------------------
# 類型
# ---------------------------------------------------------------------------


def test_every_scam_type_member_maps_to_a_non_empty_pair() -> None:
    assert len(ScamType) == 18
    for member in ScamType:
        out = scam_type_out(member)
        assert out.code == member.name
        assert out.label == member.value
        assert out.code and out.label


def test_no_scam_type_is_a_single_null_not_a_pair_of_nulls() -> None:
    response = verdict_to_response(_verdict(scam_type=None))
    assert response.scam_type is None
    assert response.model_dump()["scam_type"] is None


def test_scam_type_carries_both_identifier_and_label() -> None:
    response = verdict_to_response(_verdict(scam_type=ScamType.FAKE_AUTHORITY))
    assert response.scam_type == ScamTypeOut(code="FAKE_AUTHORITY", label="假檢警/假冒公務機關")


# ---------------------------------------------------------------------------
# 欄位說明與錯誤形狀
# ---------------------------------------------------------------------------


def test_evidence_description_warns_about_raw_text_and_logging() -> None:
    """說明 MUST 出現在對外的介面文件，不能只存在於設計文件。"""
    description = CheckResponse.model_json_schema()["properties"]["evidence"]["description"]
    assert "原文片段" in description
    assert "log" in description


def test_confidence_description_refuses_two_misreadings() -> None:
    description = CheckResponse.model_json_schema()["properties"]["confidence"]["description"]
    assert "不是詐騙機率" in description
    assert "百分比" in description


def test_confidence_passes_through_without_being_bucketed() -> None:
    """文字化需要知道那六個離散值與它們的含義，那是信心層的知識，不是本層的。"""
    response = verdict_to_response(_verdict(confidence=0.55))
    assert response.confidence == 0.55


def test_error_response_has_a_single_shape() -> None:
    error = ErrorResponse(
        error=ErrorDetail(code="invalid_request", field="messages.0.at", message="at 不帶時區")
    )
    assert error.model_dump() == {
        "error": {"code": "invalid_request", "field": "messages.0.at", "message": "at 不帶時區"}
    }


def test_error_field_is_null_when_the_error_belongs_to_no_field() -> None:
    error = ErrorResponse(
        error=ErrorDetail(code="request_too_large", field=None, message="請求體超過上限")
    )
    assert error.error.field is None


# ---------------------------------------------------------------------------
# 端對端的形狀：由 `Request` 建構到回應
# ---------------------------------------------------------------------------


def test_validated_request_converts_to_core_messages() -> None:
    """線上的鍵名與核心的屬性名對得上，這是 alias 唯一要證明的事。"""
    validated = CheckRequest.model_validate(
        {"messages": [{"text": "您好", "from": "them", "at": "2026-09-14T10:00:00Z"}]}
    )
    core = Request(
        messages=[
            Message(text=item.text, sender=item.sender, sent_at=item.sent_at)
            for item in validated.messages
        ]
    )
    assert core.latest.text == "您好"
    assert core.latest.sender == "them"
    assert core.latest.sent_at == datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
