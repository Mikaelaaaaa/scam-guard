"""介面層（`app.py`）的測試。

`pytest.importorskip("gradio")` 在**模組層、檔案頂端**，不是藏在函式裡的
import。本機沒裝 `[demo]` extra 時整個檔案被跳過；CI 安裝 `.[dev,demo]`，
因為 skipped 不會讓 CI 變紅 —— 一個安靜跳過的測試檔等於沒有測試。
"""

import json
import re
import time
from datetime import UTC, datetime

import pytest

pytest.importorskip("gradio")

# `gradio` 由 `app` 持有。此處**不直接 import 它** —— `pyproject.toml` 的
# `banned-api` 只豁免 `app.py`，而本 change 不新增 ruff 豁免。
import app  # noqa: E402
from scam_guard.check import Check, CheckRegistry, Stage  # noqa: E402
from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document  # noqa: E402
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect  # noqa: E402
from scam_guard.redact import RedactedText  # noqa: E402
from scam_guard.types import CheckResult, Message, Request, ScamType, Verdict  # noqa: E402

EMPTY_REDACTED = RedactedText(sentences=[], coords=[], counts={})

EMPTY_VERDICT = Verdict(
    scam_probability=None,
    confidence=0.0,
    scam_type=None,
    evidence=[],
    actions=[],
    checks=[],
    redacted=EMPTY_REDACTED,
)


def verdict_with(**overrides) -> Verdict:
    fields = {
        "scam_probability": None,
        "confidence": 0.0,
        "scam_type": None,
        "evidence": [],
        "actions": [],
        "checks": [],
        "redacted": EMPTY_REDACTED,
    }
    fields.update(overrides)
    return Verdict(**fields)


class HardLocalCheck:
    """一定命中的硬證據 `LOCAL` 檢查，用來觸發 pipeline 的短路。"""

    name = "solicit_otp"
    stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return [CheckResult(name=self.name, hit=True, detail="假的硬證據", hard=True)]


class ExpensiveCheck:
    """永遠不該被執行到的 `EXPENSIVE` 檢查。"""

    name = "domain_age"
    stage = Stage.EXPENSIVE

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        raise AssertionError("短路時不應執行 EXPENSIVE 檢查")


class BadCoordCheck:
    """回報一個不存在於 `Document` 的座標。"""

    name = "secrecy_demand"
    stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return [CheckResult(name=self.name, hit=True, detail="越界座標", evidence=[(99, 99)])]


class RecordingCheck:
    """記下 `detect()` 內部產生的 `Document`，供座標一致性比對。"""

    name = "quotation"
    stage = Stage.LOCAL

    def __init__(self) -> None:
        self.seen: list[Document] = []

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        self.seen.append(doc)
        return []


# ---------------------------------------------------------------------------
# 11.2 – 11.6 確定性渲染層
# ---------------------------------------------------------------------------


def test_render_lines_no_probability_emits_no_number() -> None:
    lines = app.render_lines(verdict_with(scam_type=ScamType.FAKE_AUTHORITY))
    assert lines.judgement is not None
    assert "0%" not in lines.judgement
    assert not re.search(r"\d", lines.judgement)
    assert app.UNDECIDED in lines.judgement


def test_render_lines_missing_advice_segment_does_not_exist() -> None:
    lines = app.render_lines(EMPTY_VERDICT)
    assert lines.advice is None
    assert lines.advice != ""
    assert lines.segments() == []


def test_render_lines_falls_back_to_raw_check_detail_with_marker() -> None:
    checks = [
        CheckResult(name="solicit_otp", hit=True, detail="索取簡訊驗證碼"),
        CheckResult(name="secrecy_demand", hit=True, detail="要求保密"),
        CheckResult(name="atm_operation", hit=False, detail=NOT_HIT),
    ]
    lines = app.render_lines(verdict_with(checks=checks))
    assert lines.grounds is not None
    assert lines.grounds.startswith(app.RAW_GROUNDS_PREFIX)
    assert "solicit_otp：索取簡訊驗證碼" in lines.grounds
    assert "secrecy_demand：要求保密" in lines.grounds
    assert "atm_operation" not in lines.grounds


def test_render_lines_prefers_rendered_evidence_without_marker() -> None:
    checks = [CheckResult(name="solicit_otp", hit=True, detail="索取簡訊驗證碼")]
    lines = app.render_lines(verdict_with(evidence=["要求提供簡訊驗證碼"], checks=checks))
    assert lines.grounds == "要求提供簡訊驗證碼"
    assert app.RAW_GROUNDS_PREFIX not in lines.grounds


def test_practice_speech_reports_empty_registry_as_system_status() -> None:
    segments = app.practice_speech(app.render_lines(EMPTY_VERDICT), 0)
    assert "目前沒有註冊任何檢查" in segments[0]


def test_practice_speech_reports_registered_but_no_hit() -> None:
    segments = app.practice_speech(app.render_lines(EMPTY_VERDICT), 27)
    assert "27 項檢查全部未命中" in segments[0]


def test_inquiry_answer_always_gives_fallback_action_marked_as_unrelated() -> None:
    segments = app.inquiry_answer(app.render_lines(EMPTY_VERDICT))
    assert app.UNDECIDED in segments[0]
    assert app.FALLBACK_ACTION in segments[-1]
    assert app.FALLBACK_NOTE in segments[-1]


def test_both_modes_share_the_same_three_segments() -> None:
    checks = [CheckResult(name="solicit_otp", hit=True, detail="索取簡訊驗證碼")]
    verdict = verdict_with(checks=checks, actions=["不要照做"])
    lines = app.render_lines(verdict)
    assert app.practice_speech(lines, 27)[:-1] == app.inquiry_answer(lines)[:-1]


def test_render_lines_output_is_plain_text() -> None:
    checks = [CheckResult(name="solicit_otp", hit=True, detail="索取簡訊驗證碼")]
    lines = app.render_lines(verdict_with(checks=checks, actions=["撥打 165"]))
    for segment in lines.segments():
        assert "<" not in segment
        assert "](" not in segment


# ---------------------------------------------------------------------------
# 11.7 – 11.11 潤飾驗證
# ---------------------------------------------------------------------------


def test_polish_rejects_invented_number() -> None:
    validator = app.PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("這則訊息有 87% 的機率是詐騙") is False


def test_polish_rejects_url() -> None:
    validator = app.PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("你可以去 http") is False


def test_polish_rejects_www_prefix() -> None:
    validator = app.PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("去看看 www.") is False


def test_polish_rejects_scam_type_absent_from_verdict() -> None:
    validator = app.PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("這看起來是假檢警/假冒公務機關的手法") is False


def test_polish_accepts_scam_type_present_in_verdict() -> None:
    verdict = verdict_with(scam_type=ScamType.FAKE_AUTHORITY)
    validator = app.PolishValidator(["這是假檢警/假冒公務機關。撥打 165 查證。"], verdict)
    assert validator.feed("聽起來是假檢警/假冒公務機關，我先撥打 165 查證") is True


def test_polish_accepts_rewording() -> None:
    segments = ["尚無法判定這則訊息是不是詐騙。", "撥打 165 反詐騙專線查證。"]
    validator = app.PolishValidator(segments, EMPTY_VERDICT)
    assert validator.feed("我看不出這是不是詐騙耶，我還是先撥打 165 問問好了。") is True


def test_polish_validation_is_prefix_decidable() -> None:
    """逐字元餵入，驗證器在違規字元出現的那一步即回報，不需要完整字串。"""
    segments = ["撥打 165 反詐騙專線查證。"]
    validator = app.PolishValidator(segments, EMPTY_VERDICT)
    text = "我覺得有 87 成機率"
    failed_at = None
    for index, character in enumerate(text):
        if not validator.feed(character):
            failed_at = index
            break
    assert failed_at == text.index("8")
    assert failed_at < len(text) - 1


def test_polish_allows_prefix_of_allowed_number() -> None:
    validator = app.PolishValidator(["撥打 165 查證。"], EMPTY_VERDICT)
    assert validator.feed("1") is True
    assert validator.feed("6") is True
    assert validator.feed("5") is True
    assert validator.feed("7") is False


def test_allowed_numbers_extracts_every_digit_run() -> None:
    assert app.allowed_numbers(["撥打 165 查證", "共 27 項"]) == frozenset({"165", "27"})


# ---------------------------------------------------------------------------
# 11.12 – 11.13 檢查四種狀態
# ---------------------------------------------------------------------------


def test_pipeline_detail_constants_are_locked() -> None:
    """狀態分類靠這兩個常數，值變了就會靜默地把狀態分錯。"""
    assert NOT_HIT == "未命中"
    assert SKIPPED == "因短路未執行"


def test_check_state_classifies_three_states() -> None:
    assert app.check_state(CheckResult(name="a", hit=True, detail="命中了")) == (app.STATE_HIT)
    assert app.check_state(CheckResult(name="b", hit=False, detail=NOT_HIT)) == (app.STATE_NOT_HIT)
    assert app.check_state(CheckResult(name="c", hit=False, detail=SKIPPED)) == (app.STATE_SKIPPED)


def test_check_state_refuses_to_guess() -> None:
    with pytest.raises(ValueError, match="無法分類的檢查記錄"):
        app.check_state(CheckResult(name="d", hit=False, detail="別的東西"))


def test_skipped_check_renders_as_its_own_state() -> None:
    """走 `_skipped()` 的實際路徑：硬證據命中使 EXPENSIVE 檢查未被執行。"""
    registry = CheckRegistry()
    registry.register(HardLocalCheck())
    registry.register(ExpensiveCheck())
    request = Request.from_text("測試訊息。")
    verdict = detect(request, registry, app.TABLE, limits=app.LIMITS)
    document = build_document(request.messages, app.LIMITS)

    states = {result.name: app.check_state(result) for result in verdict.checks}
    assert states["solicit_otp"] == app.STATE_HIT
    assert states["domain_age"] == app.STATE_SKIPPED

    panel = app.render_panel(verdict, document, len(registry.enabled()), None)
    assert f"state-{app.STATE_SKIPPED}" in panel
    assert f"state-{app.STATE_HIT}" in panel
    assert "硬證據" in panel


def test_panel_shows_the_four_states_distinctly() -> None:
    checks = [
        CheckResult(name="a", hit=True, detail="命中了"),
        CheckResult(name="b", hit=False, detail=NOT_HIT),
        CheckResult(name="c", hit=False, detail=SKIPPED),
    ]
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, app.LIMITS)
    panel = app.render_panel(verdict_with(checks=checks), document, 3, None)
    for state in (app.STATE_HIT, app.STATE_NOT_HIT, app.STATE_SKIPPED, app.STATE_UNREGISTERED):
        assert f"state-{state}" in panel
    assert "domain_age" in panel
    assert "未注入 RDAP 解析器" in panel


def test_panel_does_not_filter_out_misses() -> None:
    checks = [CheckResult(name="quiet_check", hit=False, detail=NOT_HIT)]
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, app.LIMITS)
    panel = app.render_panel(verdict_with(checks=checks), document, 1, None)
    assert "quiet_check" in panel


def test_panel_reports_registered_count() -> None:
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, app.LIMITS)
    assert "已註冊 0 個檢查" in app.render_panel(EMPTY_VERDICT, document, 0, None)
    assert "已註冊 27 個檢查" in app.render_panel(EMPTY_VERDICT, document, 27, None)


def test_unregistered_list_holds_only_implemented_checks() -> None:
    names = {name for name, _reason in app.UNREGISTERED_CHECKS}
    assert names == {"domain_age"}


# ---------------------------------------------------------------------------
# 11.14 – 11.15 座標
# ---------------------------------------------------------------------------


def test_panel_document_coords_match_detect_internal_document() -> None:
    request = Request(
        messages=[
            Message(text="您好，這裡是中華郵政。請把驗證碼告訴我。"),
            Message(text="不要告訴家人。快點！"),
        ]
    )
    recorder = RecordingCheck()
    registry = CheckRegistry()
    registry.register(recorder)
    detect(request, registry, app.TABLE, limits=app.LIMITS)
    panel_document = build_document(request.messages, app.LIMITS)
    assert recorder.seen[0].coords == panel_document.coords
    assert recorder.seen[0].raw_sentences == panel_document.raw_sentences


def test_invalid_coord_propagates_key_error() -> None:
    """越界座標 MUST 讓 `KeyError` 傳播，不得被吞成「少一條依據」。

    `add-verdict-render` 落地後拋出的位置從呈現層前移到 `detect()` 內部的
    `render_evidence()` —— 那一層同樣以 `doc.raw_at()` 解析座標。前移是好事：
    錯誤的證據在更早的地方就炸掉，而本測試鎖的是「會炸」，不是「在哪一層炸」。
    """
    registry = CheckRegistry()
    registry.register(BadCoordCheck())
    request = Request.from_text("測試訊息。")
    with pytest.raises(KeyError):
        detect(request, registry, app.TABLE, limits=app.LIMITS)


def test_evidence_links_point_to_sentence_anchors() -> None:
    checks = [
        CheckResult(name="a", hit=True, detail="命中了", evidence=[(0, 1)]),
    ]
    request = Request.from_text("第一句。第二句。")
    document = build_document(request.messages, app.LIMITS)
    panel = app.render_panel(verdict_with(checks=checks), document, 1, None)
    conversation = app.render_conversation(request.messages, (), document, "收到的訊息", None)
    assert 'href="#s-0-1"' in panel
    assert 'id="s-0-1"' in conversation


def test_hit_without_evidence_is_still_a_hit() -> None:
    checks = [CheckResult(name="a", hit=True, detail="沒有句子位置")]
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, app.LIMITS)
    panel = app.render_panel(verdict_with(checks=checks), document, 1, None)
    assert f"state-{app.STATE_HIT}" in panel
    assert "evidence-list" not in panel


# ---------------------------------------------------------------------------
# 11.16 – 11.20 兩個模式
# ---------------------------------------------------------------------------


def test_practice_messages_carry_sender_and_timestamp() -> None:
    before = datetime.now(tz=UTC)
    messages = app.practice_messages([], "你好")
    after = datetime.now(tz=UTC)
    assert len(messages) == 1
    assert messages[0].sender == app.SENDER_THEM
    assert messages[0].sent_at is not None
    assert before <= messages[0].sent_at <= after


def test_practice_request_contains_only_user_messages() -> None:
    messages: list[Message] = []
    replies: list[str] = []
    for text in ("第一則", "第二則", "第三則"):
        outputs = list(app.practice_submit(text, messages, replies))
        messages, replies = outputs[-1][0], outputs[-1][1]
    assert len(messages) == 3
    assert [message.text for message in messages] == ["第一則", "第二則", "第三則"]
    assert len(replies) == 3
    for reply in replies:
        assert reply not in [message.text for message in messages]
    assert all(message.sender == app.SENDER_THEM for message in messages)


def test_victim_reply_does_not_change_the_next_verdict() -> None:
    """受害方的台詞含詐騙關鍵詞時，下一輪的檢查結果不因它而改變。"""
    messages: list[Message] = []
    replies: list[str] = []
    outputs = list(app.practice_submit("請把驗證碼告訴我，不要告訴家人。", messages, replies))
    messages, replies = outputs[-1][0], outputs[-1][1]
    assert any("驗證碼" in reply for reply in replies)

    with_reply = detect(Request(messages=messages), app.REGISTRY, app.TABLE, limits=app.LIMITS)
    without_reply = detect(
        Request(messages=[Message(text=messages[0].text)]),
        app.REGISTRY,
        app.TABLE,
        limits=app.LIMITS,
    )
    assert [(r.name, r.hit, r.detail) for r in with_reply.checks] == [
        (r.name, r.hit, r.detail) for r in without_reply.checks
    ]


def test_inquiry_splits_on_blank_lines() -> None:
    request = app.build_inquiry_request("第一則\n\n第二則\n\n第三則")
    assert len(request.messages) == 3
    assert [message.text for message in request.messages] == ["第一則", "第二則", "第三則"]
    for message in request.messages:
        assert message.sender is None
        assert message.sent_at is None


def test_inquiry_single_message_stays_single() -> None:
    request = app.build_inquiry_request("只有一則\n但是有換行\n沒有空行")
    assert len(request.messages) == 1


def test_inquiry_does_not_parse_forwarded_format() -> None:
    """含日期與暱稱行的輸入不被切成多則 —— 那些行是一般文字。"""
    pasted = "2026/09/14 10:00 小明\n請匯款到監管帳戶\n2026/09/14 10:01 小美\n好的"
    request = app.build_inquiry_request(pasted)
    assert len(request.messages) == 1
    assert "小明" in request.messages[0].text


def test_inquiry_rejects_empty_input() -> None:
    with pytest.raises(app.gr.Error):
        app.build_inquiry_request("   \n\n  ")


def test_inquiry_ignores_the_polisher(monkeypatch: pytest.MonkeyPatch) -> None:
    """已注入潤飾層時，模式二的輸出與未注入時完全相同。"""
    text = "請把簡訊驗證碼給我，不要告訴家人。"
    monkeypatch.setattr(app, "POLISHER", None)
    without = app.inquiry_submit(text)

    def never_called(segments):
        raise AssertionError("模式二 MUST NOT 呼叫潤飾層")

    monkeypatch.setattr(app, "POLISHER", never_called)
    assert app.inquiry_submit(text) == without


def test_modes_do_not_share_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    baseline = app.inquiry_submit("獨立的一則訊息。")
    messages: list[Message] = []
    replies: list[str] = []
    for text in ("模式一的第一則", "模式一的第二則"):
        outputs = list(app.practice_submit(text, messages, replies))
        messages, replies = outputs[-1][0], outputs[-1][1]
    assert app.inquiry_submit("獨立的一則訊息。") == baseline
    assert "模式一的第一則" not in baseline[0]


# ---------------------------------------------------------------------------
# 11.21 更新順序
# ---------------------------------------------------------------------------


class SlowPolisher:
    """會 sleep 的假潤飾層。第一次產出若要等它，測試就會慢到看得出來。"""

    def __init__(self) -> None:
        self.called = False

    def __call__(self, segments):
        self.called = True
        time.sleep(0.5)
        yield "嗯"
        time.sleep(0.5)
        yield "，我先不要照做。"


def test_panel_completes_before_the_model_produces_any_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generator 的**第一次產出**已含完整判定列與偵測項目。

    這條測試會在「等模型跑完再一起更新」的實作下失敗：那種實作的第一次 `yield`
    要等 1 秒，而且 `polisher.called` 在第一次產出時已為 True。
    """
    polisher = SlowPolisher()
    monkeypatch.setattr(app, "POLISHER", polisher)

    stream = app.practice_submit("請把驗證碼告訴我。", [], [])
    started = time.monotonic()
    first = next(stream)
    elapsed = time.monotonic() - started

    assert polisher.called is False, "面板產出時潤飾層尚未被呼叫"
    assert elapsed < 0.4, f"第一次產出花了 {elapsed:.2f} 秒，代表它等了模型"

    _messages, _replies, conversation, row, panel, status, _box = first
    assert "詐騙機率" in row and "信心值" in row and "詐騙類型" in row
    assert "已註冊" in panel
    assert "solicit_otp" in panel
    assert "s-0-0" in conversation
    assert status == app.POLISH_STREAMING

    rest = list(stream)
    assert polisher.called is True
    assert len(rest) >= 2, "潤飾層的輸出 MUST 分多次更新於畫面"
    assert rest[-1][5] == app.POLISH_ACCEPTED


def test_panel_is_identical_with_and_without_polisher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    without = next(app.practice_submit("請把驗證碼告訴我。", [], []))
    monkeypatch.setattr(app, "POLISHER", SlowPolisher())
    with_polisher = next(app.practice_submit("請把驗證碼告訴我。", [], []))
    assert without[3] == with_polisher[3]
    assert without[4] == with_polisher[4]
    assert without[5] != with_polisher[5]


class LyingPolisher:
    def __call__(self, segments):
        yield "我覺得"
        yield "有 87"
        yield "% 的機率是詐騙"


def test_discarded_polish_is_visible_and_falls_back_to_deterministic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "POLISHER", LyingPolisher())
    outputs = list(app.practice_submit("請把驗證碼告訴我。", [], []))
    deterministic = outputs[0][1][-1]
    assert outputs[-1][5] == app.POLISH_DISCARDED
    assert outputs[-1][5] != app.POLISH_NOT_INJECTED
    assert outputs[-1][1][-1] == deterministic
    assert "87" not in outputs[-1][2]


def test_not_injected_and_discarded_are_different_labels() -> None:
    assert app.POLISH_NOT_INJECTED != app.POLISH_DISCARDED
    assert app.POLISH_NOT_INJECTED != app.POLISH_ACCEPTED


# ---------------------------------------------------------------------------
# 11.22 HTML 逸出
# ---------------------------------------------------------------------------


def test_user_input_is_escaped() -> None:
    request = Request.from_text("<script>alert(1)</script>")
    document = build_document(request.messages, app.LIMITS)
    conversation = app.render_conversation(request.messages, (), document, "收到的訊息", None)
    assert "<script>" not in conversation
    assert "&lt;script&gt;" in conversation


def test_escaped_url_round_trips() -> None:
    raw = "https://x.cc/a?id=1&ref=2"
    assert app.escaped(raw) == "https://x.cc/a?id=1&amp;ref=2"


# ---------------------------------------------------------------------------
# 11.23 截斷
# ---------------------------------------------------------------------------


def test_truncation_is_visible() -> None:
    tiny = Limits(max_messages=2, max_chars=1_000)
    messages = [Message(text=f"第 {index} 則訊息。") for index in range(5)]
    request = Request(messages=messages)
    document = build_document(request.messages, tiny)
    assert document.truncated is True
    panel = app.render_panel(EMPTY_VERDICT, document, 0, None)
    assert f"已丟棄最舊的 {document.dropped_messages} 則" in panel

    conversation = app.render_conversation(messages, (), document, "收到的訊息", None)
    assert conversation.count("未納入本次判定") == document.dropped_messages


def test_no_truncation_notice_when_within_limits() -> None:
    request = Request.from_text("一則短訊息。")
    document = build_document(request.messages, app.LIMITS)
    assert "已丟棄最舊的" not in app.render_panel(EMPTY_VERDICT, document, 0, None)


# ---------------------------------------------------------------------------
# 11.24 範例庫
# ---------------------------------------------------------------------------


def test_samples_load_with_source_uri() -> None:
    assert 6 <= len(app.SAMPLES) <= 10
    for sample in app.SAMPLES:
        assert sample.source_uri.startswith("https://cofacts.tw/article/")
        assert sample.text.strip()
        assert sample.label.strip()


def test_samples_file_declares_its_own_license() -> None:
    with app.SAMPLES_PATH.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["license"] == "CC BY-SA 4.0"
    assert "MIT" in loaded["attribution"]
    assert "cofacts.tw" in loaded["attribution"]


def test_missing_samples_file_raises_with_filename(tmp_path) -> None:
    missing = tmp_path / "demo_samples.json"
    with pytest.raises(FileNotFoundError, match="demo_samples.json"):
        app.load_samples(missing)


def test_fill_sample_returns_text_without_submitting() -> None:
    sample = app.SAMPLES[0]
    assert app.fill_sample(sample.label) == sample.text


def test_fill_sample_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="沒有這個標籤"):
        app.fill_sample("不存在的標籤")


# ---------------------------------------------------------------------------
# 11.25 個資標註
# ---------------------------------------------------------------------------


def recognizer_none(sentence: str) -> list[app.PiiSpan]:
    return []


def recognizer_tw_id(sentence: str) -> list[app.PiiSpan]:
    return [
        (match.start(), match.end(), "身分證") for match in re.finditer(r"[A-Z][12]\d{8}", sentence)
    ]


def test_pii_not_mounted_is_not_the_same_as_no_pii() -> None:
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, app.LIMITS)
    not_mounted = app.render_pii_block(document, None)
    mounted = app.render_pii_block(document, recognizer_none)
    assert app.PII_NOT_MOUNTED in not_mounted
    assert app.PII_NO_HIT not in not_mounted
    assert app.PII_NO_HIT in mounted
    assert app.PII_NOT_MOUNTED not in mounted
    assert not_mounted != mounted


def test_pii_block_discloses_its_scope() -> None:
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, app.LIMITS)
    block = app.render_pii_block(document, None)
    for label in ("身分證", "手機", "市話", "信用卡"):
        assert label in block
    assert "姓名與地址不在辨識範圍內" in block
    assert "預設關閉" in block


def test_pii_tags_are_sentence_level_and_do_not_rewrite_text() -> None:
    request = Request.from_text("我的身分證是 A123456789 請確認。")
    document = build_document(request.messages, app.LIMITS)
    conversation = app.render_conversation(
        request.messages, (), document, "收到的訊息", recognizer_tw_id
    )
    assert "身分證 ×1" in conversation
    assert "A123456789" in conversation
    assert "&lt;TW_ID&gt;" not in conversation
    assert "<TW_ID>" not in conversation


def test_sentence_without_pii_gets_no_tag() -> None:
    request = Request.from_text("這句沒有個資。")
    document = build_document(request.messages, app.LIMITS)
    conversation = app.render_conversation(
        request.messages, (), document, "收到的訊息", recognizer_tw_id
    )
    assert "pii-tag" not in conversation


def test_pii_highlight_slices_before_escaping() -> None:
    """迴歸測試：`html.escape()` 是一對多的（`&` → `&amp;`）。

    先逸出整句再用區間切片，從第一個 `&` 之後的所有索引全部位移，`<mark>` 會標到
    隔壁幾個字 —— 而 `&` 在帶追蹤參數的釣魚連結裡幾乎必然出現。這個錯不會拋例外，
    沒有任何測試會報告它，所以必須有這一條。
    """
    sentence = "請至 https://x.cc/a?id=1&ref=2 驗證，身分證 A123456789"
    spans = recognizer_tw_id(sentence)
    assert spans, "測試前提：句中有一筆身分證"
    rendered = app.render_pii_highlight(sentence, spans)
    marked = re.findall(r'<mark class="pii-mark"[^>]*>(.*?)</mark>', rendered)
    assert marked == ["A123456789"]
    assert "&amp;ref=2" in rendered


def test_pii_block_states_it_shows_normalized_sentences() -> None:
    request = Request.from_text("我的身分證是 Ａ１２３４５６７８９ 請確認。")
    document = build_document(request.messages, app.LIMITS)
    block = app.render_pii_block(document, recognizer_tw_id)
    assert "正規化後" in block
    assert "pii-mark" in block


# ---------------------------------------------------------------------------
# 11.26 外部查詢與隱私揭露
# ---------------------------------------------------------------------------


def test_external_block_lists_domain_age_and_its_disclosure() -> None:
    assert "domain_age" in app.EXTERNAL_BLOCK
    assert "未注入 RDAP 解析器" in app.EXTERNAL_BLOCK
    assert "registrable domain" in app.EXTERNAL_BLOCK
    assert "不含 path、query string 與子網域" in app.EXTERNAL_BLOCK
    assert "目前不對任何外部服務發出請求" in app.EXTERNAL_BLOCK


def test_privacy_note_does_not_claim_zero_trace() -> None:
    for claim in ("完全不留痕跡", "不被任何系統記錄", "完全不會被記錄"):
        assert claim not in app.PRIVACY_NOTE
    assert "不在本系統控制範圍內" in app.PRIVACY_NOTE
    assert "同一個程序" in app.PRIVACY_NOTE
    assert "access log" in app.PRIVACY_NOTE


def test_transcript_notice_only_when_logger_injected() -> None:
    assert app.transcript_notice(None) == ""
    written: list[list[str]] = []
    assert "本模式的輸入會被記錄" in app.transcript_notice(written.append)


def test_transcript_logger_receives_explicit_message_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[list[str]] = []
    monkeypatch.setattr(app, "TRANSCRIPT_LOGGER", written.append)
    monkeypatch.setattr(app, "POLISHER", None)
    list(app.practice_submit("第一則", [], []))
    assert written == [["第一則"]]
    assert "len=" not in written[0][0]


def test_no_logging_without_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "TRANSCRIPT_LOGGER", None)
    monkeypatch.setattr(app, "POLISHER", None)
    outputs = list(app.practice_submit("第一則", [], []))
    assert outputs


def test_verdict_row_has_three_labelled_columns() -> None:
    row = app.render_verdict_row(EMPTY_VERDICT)
    assert row.count('class="cell"') == 3
    for label in ("詐騙機率", "信心值", "詐騙類型"):
        assert label in row
    assert "自我信心" in row
    assert row.count("未判定") == 2
    assert "0.00" in row


def test_verdict_row_has_no_traffic_light_grading() -> None:
    row = app.render_verdict_row(EMPTY_VERDICT)
    for word in ("高風險", "中風險", "低風險", "紅燈", "黃燈", "綠燈"):
        assert word not in row


def test_verdict_row_shows_probability_when_present() -> None:
    row = app.render_verdict_row(
        verdict_with(scam_probability=0.87, confidence=0.62, scam_type=ScamType.FAKE_AUTHORITY)
    )
    assert "87%" in row
    assert "0.62" in row
    assert "假檢警/假冒公務機關" in row
    assert "未判定" not in row


# ---------------------------------------------------------------------------
# 組裝層
# ---------------------------------------------------------------------------


def test_limits_is_a_single_value_shared_by_detect_and_build_document() -> None:
    assert app.LIMITS is DEFAULT_LIMITS
    assert isinstance(app.LIMITS, Limits)


def test_registry_is_built_once_by_the_assembly_layer() -> None:
    assert isinstance(app.REGISTRY, CheckRegistry)
    rebuilt: list[Check] = app.build_registry().enabled()
    assert [check.name for check in app.REGISTRY.enabled()] == [check.name for check in rebuilt]


def test_panel_row_count_follows_the_registry() -> None:
    """新增一項檢查時面板多一行，呈現邏輯未變。"""
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, app.LIMITS)

    small = CheckRegistry()
    small.register(HardLocalCheck())
    before = detect(request, small, app.TABLE, limits=app.LIMITS)

    bigger = CheckRegistry()
    bigger.register(HardLocalCheck())
    bigger.register(RecordingCheck())
    after = detect(request, bigger, app.TABLE, limits=app.LIMITS)

    panel_before = app.render_panel(before, document, len(small.enabled()), None)
    panel_after = app.render_panel(after, document, len(bigger.enabled()), None)
    assert panel_after.count('class="check ') == panel_before.count('class="check ') + 1
    assert "已註冊 1 個檢查" in panel_before
    assert "已註冊 2 個檢查" in panel_after


def test_domain_age_is_not_injected() -> None:
    assert "domain_age" not in {check.name for check in app.REGISTRY.enabled()}


def test_optional_dependencies_default_to_not_injected() -> None:
    assert app.PII_RECOGNIZER is None
    assert app.POLISHER is None
    assert app.TRANSCRIPT_LOGGER is None


def test_demo_builds_as_blocks_without_flagging() -> None:
    demo = app.build_demo()
    assert isinstance(demo, app.gr.Blocks)
    assert not isinstance(demo, app.gr.Interface)
    assert demo.analytics_enabled is False
