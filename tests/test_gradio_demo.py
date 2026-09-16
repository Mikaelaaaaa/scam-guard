"""Gradio 介面層（`app.py`）的測試。

`pytest.importorskip("gradio")` 在**模組層、檔案頂端**，不是藏在函式裡的
import。本機沒裝 `[demo]` extra 時整個檔案被跳過；CI 安裝 `.[dev,demo]`，
因為 skipped 不會讓 CI 變紅 —— 一個安靜跳過的測試檔等於沒有測試。

**標記層的性質不在這裡驗。** 判定卡、偵測細節、氣泡、排行、受害方回應與樣式表
全部住在 `demo_ui`，它們的測試在 `tests/test_demo_ui.py`，而那個檔案不需要
gradio 也不需要 PSL 快照。這裡只驗這一側獨有的東西：組裝、事件處理、跨輪狀態、
潤飾層的驗證，以及「兩個模式不共用狀態」。
"""

import ast
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("gradio")

# `gradio` 由 `app` 持有。此處**不直接 import 它** —— `pyproject.toml` 的
# `banned-api` 只豁免 `app.py`。
import app  # noqa: E402
import demo_ui  # noqa: E402
from scam_guard.check import Check, CheckRegistry  # noqa: E402
from scam_guard.normalize import DEFAULT_LIMITS, Limits, build_document  # noqa: E402
from scam_guard.pii import find_pii  # noqa: E402
from scam_guard.pipeline import detect  # noqa: E402
from scam_guard.redact import PLACEHOLDERS, RedactedText  # noqa: E402
from scam_guard import pii  # noqa: E402
from scam_guard.types import Message, Request, Verdict  # noqa: E402

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

SCAM_LINE = "請把簡訊驗證碼給我，不要告訴家人。"

MESSAGES, REPLIES, SPOKEN, CONVERSATION, CARD, RANKING, STATUS, BOX = range(8)


def run_practice(
    text: str, state: tuple[list, list, list]
) -> tuple[tuple, tuple[list, list, list]]:
    """跑完一輪對練，回傳最後一次產出與更新後的三份跨輪狀態。"""
    outputs = list(app.practice_submit(text, *state))
    last = outputs[-1]
    return last, (last[MESSAGES], last[REPLIES], last[SPOKEN])


# ---------------------------------------------------------------------------
# persona 驗證 —— 只剩需要 `app.practice_submit()` 的那一條，
# 其餘已隨 `PolishValidator` 一起移到 `tests/test_polish_validator.py`
# ---------------------------------------------------------------------------


def test_persona_cannot_invent_a_percentage_absent_from_the_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """這份 Verdict 的全部確定性素材都沒有百分比，persona 因此不能發明一個。"""
    monkeypatch.setattr(app, "POLISHER", None)
    outputs = list(app.practice_submit(SCAM_LINE, [], [], []))
    victim = outputs[-1][REPLIES][-1]
    assert not re.search(r"\d+%", victim)

    request = Request.from_text(SCAM_LINE)
    verdict = detect(request, app.REGISTRY, app.TABLE, limits=app.LIMITS)
    validator = demo_ui.PolishValidator(demo_ui.verdict_segments(verdict), verdict)
    assert validator.feed("這則訊息有 92% 的可能") is False


# ---------------------------------------------------------------------------
# 模式一：對練
# ---------------------------------------------------------------------------


def test_practice_messages_carry_sender_and_timestamp() -> None:
    before = datetime.now(tz=UTC)
    messages = app.practice_messages([], "你好")
    after = datetime.now(tz=UTC)
    assert len(messages) == 1
    assert messages[0].sender == app.SENDER_THEM
    assert messages[0].sent_at is not None
    assert before <= messages[0].sent_at <= after


def test_practice_request_contains_only_user_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    state: tuple[list, list, list] = ([], [], [])
    for text in ("第一則", "第二則", "第三則"):
        _last, state = run_practice(text, state)
    messages, replies, _spoken = state
    assert [message.text for message in messages] == ["第一則", "第二則", "第三則"]
    assert len(replies) == 3
    for reply in replies:
        assert reply not in [message.text for message in messages]
    assert all(message.sender == app.SENDER_THEM for message in messages)


def test_victim_reply_does_not_change_the_next_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """受害方的台詞含詐騙關鍵詞時，下一輪的檢查結果不因它而改變。"""
    monkeypatch.setattr(app, "POLISHER", None)
    _last, (messages, replies, _spoken) = run_practice(SCAM_LINE, ([], [], []))
    assert replies[0]

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


def test_the_victim_does_not_repeat_itself_on_a_turn_without_new_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本 change 的回歸測試。

    舊實作每輪把整份判定說一次，而 `Verdict` 是對整段對話算的 —— 第二輪打
    「今天天氣不錯對吧」會得到與第一輪逐字相同的八行，畫面上是兩面一樣的文字牆。
    """
    monkeypatch.setattr(app, "POLISHER", None)
    _first, state = run_practice(SCAM_LINE, ([], [], []))
    _second, state = run_practice("今天天氣不錯對吧", state)
    _messages, replies, _spoken = state
    assert replies[1] != replies[0]


def test_spoken_state_grows_at_most_one_line_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    state: tuple[list, list, list] = ([], [], [])
    sizes = []
    for text in (SCAM_LINE, "請至ATM操作解除分期付款。", "今天天氣不錯對吧"):
        _last, state = run_practice(text, state)
        sizes.append(len(state[2]))
    assert sizes == sorted(sizes)
    assert all(later - earlier <= 1 for earlier, later in zip(sizes, sizes[1:]))
    assert len(state[2]) == len(set(state[2]))


def test_ranking_accumulates_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    state: tuple[list, list, list] = ([], [], [])
    first, state = run_practice("請把簡訊驗證碼給我。", state)
    assert "第 1 輪" in first[RANKING]
    second, state = run_practice("不要告訴家人或行員。", state)
    assert "第 2 輪" in second[RANKING]
    assert second[RANKING].count('class="rank-row"') > first[RANKING].count('class="rank-row"')


def test_practice_rejects_empty_input() -> None:
    with pytest.raises(app.gr.Error):
        list(app.practice_submit("   ", [], [], []))


# ---------------------------------------------------------------------------
# 更新順序 —— 判定卡 MUST 早於模型的第一個字元
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


class RecordingPersona:
    def __init__(self) -> None:
        self.segments = None

    def __call__(self, segments):
        self.segments = segments
        yield "我不會照做，這個要求不太對勁。"


class FailingPersona:
    def __call__(self, segments):
        raise RuntimeError("模型生成失敗")
        yield from ()


class FailingSynchronousPersona:
    def __call__(self, segments):
        raise RuntimeError("模型初始化失敗")


def test_gradio_persona_receives_detection_xml_without_the_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona = RecordingPersona()
    monkeypatch.setattr(app, "POLISHER", persona)
    list(app.practice_submit(SCAM_LINE, [], [], []))
    assert persona.segments is not None
    prompt = persona.segments[0]
    assert "善良市民" in prompt
    assert "<detection>" in prompt
    assert SCAM_LINE not in prompt


def test_gradio_persona_failure_falls_back_and_says_it_was_not_generated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "POLISHER", FailingPersona())
    outputs = list(app.practice_submit(SCAM_LINE, [], [], []))
    assert outputs[-1][REPLIES][-1] == outputs[0][REPLIES][-1]
    assert outputs[-1][STATUS] == demo_ui.POLISH_FAILED
    assert "未經模型生成" in outputs[-1][STATUS]


def test_synchronous_persona_setup_failure_also_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "POLISHER", FailingSynchronousPersona())
    outputs = list(app.practice_submit(SCAM_LINE, [], [], []))
    assert outputs[-1][REPLIES][-1] == outputs[0][REPLIES][-1]
    assert outputs[-1][STATUS] == demo_ui.POLISH_FAILED


def test_character_avatar_assets_are_the_user_selected_files() -> None:
    assert (app.ASSETS_PATH / "2.png").is_file()
    assert (app.ASSETS_PATH / "3.png").is_file()
    rendered = app.render_practice_conversation(
        [Message(text="測試")], ["拒絕"], build_document([Message(text="測試")], app.LIMITS)
    )
    assert app.SCAMMER_AVATAR in rendered
    assert app.PERSONA_AVATAR in rendered
    assert "邪惡詐騙犯" in rendered
    assert "善良市民" in rendered


def test_card_completes_before_the_model_produces_any_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generator 的**第一次產出**已含完整的判定卡與排行。

    這條測試會在「等模型跑完再一起更新」的實作下失敗：那種實作的第一次 `yield`
    要等 1 秒，而且 `polisher.called` 在第一次產出時已為 True。
    """
    polisher = SlowPolisher()
    monkeypatch.setattr(app, "POLISHER", polisher)

    stream = app.practice_submit("請把驗證碼告訴我。", [], [], [])
    started = time.monotonic()
    first = next(stream)
    elapsed = time.monotonic() - started

    assert polisher.called is False, "判定卡產出時潤飾層尚未被呼叫"
    assert elapsed < 0.4, f"第一次產出花了 {elapsed:.2f} 秒，代表它等了模型"

    assert demo_ui.TITLE_SCAM in first[CARD]
    assert "solicit_otp" in first[CARD]
    assert "第 1 輪" in first[RANKING]
    assert "s-0-0" in first[CONVERSATION]
    assert first[STATUS] == demo_ui.POLISH_STREAMING

    rest = list(stream)
    assert polisher.called is True
    assert len(rest) >= 2, "潤飾層的輸出 MUST 分多次更新於畫面"
    assert rest[-1][STATUS] == demo_ui.POLISH_ACCEPTED


def test_card_and_ranking_are_identical_with_and_without_polisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    without = next(app.practice_submit("請把驗證碼告訴我。", [], [], []))
    monkeypatch.setattr(app, "POLISHER", SlowPolisher())
    with_polisher = next(app.practice_submit("請把驗證碼告訴我。", [], [], []))
    assert without[CARD] == with_polisher[CARD]
    assert without[RANKING] == with_polisher[RANKING]
    assert without[STATUS] != with_polisher[STATUS]


class LyingPolisher:
    def __call__(self, segments):
        yield "我覺得"
        yield "有 87"
        yield "% 的機率是詐騙"


def test_discarded_polish_is_visible_and_falls_back_to_deterministic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "POLISHER", LyingPolisher())
    outputs = list(app.practice_submit("請把驗證碼告訴我。", [], [], []))
    deterministic = outputs[0][REPLIES][-1]
    assert outputs[-1][STATUS] == demo_ui.POLISH_DISCARDED
    assert outputs[-1][STATUS] != demo_ui.POLISH_NOT_INJECTED
    assert outputs[-1][REPLIES][-1] == deterministic
    assert "87" not in outputs[-1][CONVERSATION]


def test_not_injected_and_discarded_are_different_labels() -> None:
    assert demo_ui.POLISH_NOT_INJECTED != demo_ui.POLISH_DISCARDED
    assert demo_ui.POLISH_NOT_INJECTED != demo_ui.POLISH_ACCEPTED


# ---------------------------------------------------------------------------
# 模式二：這是詐騙嗎
# ---------------------------------------------------------------------------


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
    monkeypatch.setattr(app, "POLISHER", None)
    without = app.inquiry_submit(SCAM_LINE)

    def never_called(segments):
        raise AssertionError("模式二 MUST NOT 呼叫潤飾層")

    monkeypatch.setattr(app, "POLISHER", never_called)
    assert app.inquiry_submit(SCAM_LINE) == without


def test_inquiry_has_no_ranking() -> None:
    """累積命中排行是對練模式的元件。"""
    card, echo = app.inquiry_submit(SCAM_LINE)
    assert "ranking" not in card
    assert "ranking" not in echo


def test_modes_do_not_share_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    baseline = app.inquiry_submit("獨立的一則訊息。")
    state: tuple[list, list, list] = ([], [], [])
    for text in ("模式一的第一則", "模式一的第二則"):
        _last, state = run_practice(text, state)
    assert app.inquiry_submit("獨立的一則訊息。") == baseline
    assert "模式一的第一則" not in baseline[0]


# ---------------------------------------------------------------------------
# 範例庫與範例標籤
# ---------------------------------------------------------------------------


def test_samples_load_with_unique_short_labels() -> None:
    assert 6 <= len(app.SAMPLES) <= 10
    labels = [sample.short_label for sample in app.SAMPLES]
    assert len(labels) == len(set(labels))
    for sample in app.SAMPLES:
        assert sample.text.strip()
        assert sample.label.strip()
        assert 0 < len(sample.short_label) <= app.SHORT_LABEL_MAX


def test_samples_file_declares_its_own_license() -> None:
    with app.SAMPLES_PATH.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["license"] == "MIT"
    assert loaded["attribution"] == "合成範例，MIT 授權。"
    assert all("source_uri" not in sample for sample in loaded["samples"])


def test_every_synthetic_sample_hits_using_rules_only() -> None:
    registry = app.build_registry(None, None, None, None)
    for sample in app.SAMPLES:
        verdict = detect(Request(messages=[Message(text=sample.text)]), registry, app.TABLE)
        assert any(result.hit for result in verdict.checks), sample.short_label
        assert any(result.hit and result.hard for result in verdict.checks), sample.short_label


def test_missing_samples_file_raises_with_filename(tmp_path) -> None:
    missing = tmp_path / "demo_samples.json"
    with pytest.raises(FileNotFoundError, match="demo_samples.json"):
        app.load_samples(missing)


def write_samples(tmp_path, entries) -> object:
    path = tmp_path / "demo_samples.json"
    path.write_text(json.dumps({"samples": entries}, ensure_ascii=False), encoding="utf-8")
    return path


def sample_entry(**overrides) -> dict:
    entry = {
        "label": "假中獎：以抽獎名義索取簡訊認證碼",
        "short_label": "假中獎",
        "text": "你中獎了",
    }
    entry.update(overrides)
    return entry


def test_missing_short_label_raises_and_names_the_entry(tmp_path) -> None:
    entry = sample_entry()
    del entry["short_label"]
    with pytest.raises(ValueError, match="short_label"):
        app.load_samples(write_samples(tmp_path, [entry]))


def test_duplicate_short_label_raises_and_names_the_value(tmp_path) -> None:
    entries = [sample_entry(), sample_entry(label="另一筆")]
    with pytest.raises(ValueError, match="重複"):
        app.load_samples(write_samples(tmp_path, entries))


def test_over_long_short_label_raises_and_names_the_value(tmp_path) -> None:
    entry = sample_entry(short_label="這個標籤實在太長了")
    with pytest.raises(ValueError, match="超過"):
        app.load_samples(write_samples(tmp_path, [entry]))


def test_sample_text_rejects_an_unknown_short_label() -> None:
    with pytest.raises(ValueError, match="沒有這個短標籤"):
        app.sample_text("不存在的標籤")


def test_the_fifth_chip_loads_the_fifth_sample() -> None:
    """捕獲迴圈變數的寫法會讓每一顆按鈕都送出最後一筆，症狀就是這一條。"""
    fifth = app.SAMPLES[4]
    text, card, echo = app.inquiry_from_sample(fifth.short_label)
    assert text == fifth.text
    assert card
    assert echo


def test_a_chip_click_fills_and_judges_in_one_step() -> None:
    sample = app.SAMPLES[0]
    text, card, _echo = app.inquiry_from_sample(sample.short_label)
    assert text == sample.text
    assert "card-title" in card


def test_a_practice_chip_sends_the_sample_as_one_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "POLISHER", None)
    sample = app.SAMPLES[4]
    outputs = list(app.practice_from_sample(sample.short_label, [], [], []))
    assert [message.text for message in outputs[-1][MESSAGES]] == [sample.text]


# ---------------------------------------------------------------------------
# 文案與頁尾
# ---------------------------------------------------------------------------

FORBIDDEN_TERMS = (
    "組裝層",
    "注入",
    "渲染",
    "協定",
    "可記錄投影",
    "確定性渲染層",
    "硬證據",
    "未經文案渲染",
)


def assembled_copy() -> str:
    """畫面上所有由本檔提供的固定文字，兩個記錄狀態各取一份。"""
    written: list[RedactedText] = []
    return "".join(
        [
            app.HEADER,
            app.PRACTICE_NOTE,
            app.SYNTHETIC_SAMPLE_NOTE,
            app.transcript_notice(written.append),
            demo_ui.POLISH_NOT_INJECTED,
            demo_ui.POLISH_STREAMING,
            demo_ui.POLISH_ACCEPTED,
            demo_ui.POLISH_DISCARDED,
            *(reason for _name, _label, reason in app.UNREGISTERED_CHECKS),
        ]
    )


def test_implementation_vocabulary_never_reaches_the_screen() -> None:
    copy = assembled_copy()
    for term in FORBIDDEN_TERMS:
        assert term not in copy


def test_no_design_note_intro_block_survives() -> None:
    """導言整塊被刪掉了：不解釋速度差、不解釋兩個模式的設計立場。"""
    assert not hasattr(app, "INTRO")
    assert not hasattr(app, "EXTERNAL_BLOCK")
    assert "毫秒級" not in app.HEADER
    assert "現場證據" not in app.HEADER


def test_no_fallback_advice_path_exists() -> None:
    """保底建議那條路徑整條刪除，不再有「此建議不基於本次判定」。"""
    assert not hasattr(app, "fallback_advice")
    assert not hasattr(app, "FALLBACK_ACTION")
    assert not hasattr(app, "FALLBACK_NOTE")
    assert not hasattr(app, "RAW_GROUNDS_PREFIX")
    assert "此建議不基於本次判定" not in assembled_copy()


def test_no_advice_at_all_when_nothing_hits() -> None:
    card, _echo = app.inquiry_submit("明天見。")
    assert demo_ui.ACTIONS_HEADING not in card


def test_removed_copy_is_absent_and_synthetic_disclosure_remains() -> None:
    page = (app.SAMPLES_PATH.parent / "docs" / "index.html").read_text(encoding="utf-8")
    copy = assembled_copy() + page
    assert "你的訊息會被怎麼處理" not in copy
    assert "Cofacts，CC BY-SA" not in copy
    assert "合成範例" in app.SYNTHETIC_SAMPLE_NOTE
    assert "真實訊息" in app.SYNTHETIC_SAMPLE_NOTE
    assert "評估報告" in app.SYNTHETIC_SAMPLE_NOTE
    assert "模型由第三方 CDN 取得" in page
    assert "你貼的訊息不離開這台電腦" in page


def test_logging_notices_never_claim_the_projection_is_anonymised() -> None:
    """遮蔽只涵蓋四個辨識類型，姓名與地址原樣留著。"""
    written: list[RedactedText] = []
    for copy in (app.transcript_notice(written.append),):
        for claim in ("去識別化", "匿名化", "不含個人資料", "不含個資"):
            assert claim not in copy
        assert "姓名" in copy
        assert "地址" in copy


def test_transcript_notice_only_when_logger_injected() -> None:
    assert app.transcript_notice(None) == ""
    written: list[RedactedText] = []
    assert "本模式的輸入會被記錄" in app.transcript_notice(written.append)


def test_transcript_logger_receives_only_the_redactable_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """簽章就是許可：`RedactedText` 是唯一可以寫進 log 的型別，
    換了簽章之後「寫原文」在型別上不可表達。"""
    written: list[RedactedText] = []
    monkeypatch.setattr(app, "TRANSCRIPT_LOGGER", written.append)
    monkeypatch.setattr(app, "POLISHER", None)
    text = "我的身分證是 A123456789，請確認。"
    list(app.practice_submit(text, [], [], []))
    app.inquiry_submit(text)

    assert len(written) == 2, "兩個模式各記錄一次，走的是同一個記錄點"
    for projection in written:
        assert isinstance(projection, RedactedText)
        joined = "".join(projection.sentences)
        assert "A123456789" not in joined
        assert PLACEHOLDERS[pii.TW_ID] in joined
        assert projection.coords
        assert set(projection.counts) == set(pii.ENTITY_TYPES)
        assert projection.counts[pii.TW_ID] == 1


def test_dropped_messages_never_reach_the_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """記錄點在 `detect()` 之後，所以被 `Limits` 丟棄的最舊訊息不進 log ——
    沒有遮蔽投影的文字沒有合法的記錄形式。"""
    written: list[RedactedText] = []
    monkeypatch.setattr(app, "TRANSCRIPT_LOGGER", written.append)
    monkeypatch.setattr(app, "LIMITS", Limits(max_messages=2, max_chars=1_000))
    app.inquiry_submit("最舊的一則。\n\n中間的一則。\n\n最新的一則。")
    joined = "".join(written[0].sentences)
    assert "最舊" not in joined
    assert "最新" in joined


def test_the_logger_is_called_with_nothing_but_the_projection() -> None:
    """掃語法樹：記錄器的全部呼叫點只有一個，而且傳的是 `verdict.redacted`。"""
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TRANSCRIPT_LOGGER"
    ]
    assert len(calls) == 1
    [argument] = calls[0].args
    assert isinstance(argument, ast.Attribute)
    assert argument.attr == "redacted"
    assert ast.unparse(argument) == "verdict.redacted"


def test_no_logging_without_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[RedactedText] = []
    monkeypatch.setattr(app, "TRANSCRIPT_LOGGER", None)
    monkeypatch.setattr(app, "POLISHER", None)
    outputs = list(app.practice_submit("第一則", [], [], []))
    assert outputs
    app.inquiry_submit("第二則")
    assert written == []
    assert app.transcript_notice(None) == ""


def test_logging_is_not_a_boolean_switch() -> None:
    """一個預設為 False 的布林是一個可以被貼進設定檔、看起來像有人想過的值。"""
    source = Path(app.__file__).read_text(encoding="utf-8")
    for flag in ("LOG_TRANSCRIPT =", "ENABLE_LOGGING", "LOGGING_ENABLED", "SHOULD_LOG"):
        assert flag not in source


# ---------------------------------------------------------------------------
# 組裝層
# ---------------------------------------------------------------------------


def test_limits_is_a_single_value_shared_by_detect_and_build_document() -> None:
    assert app.LIMITS is DEFAULT_LIMITS
    assert isinstance(app.LIMITS, Limits)


def test_registry_is_built_once_by_the_assembly_layer() -> None:
    assert isinstance(app.REGISTRY, CheckRegistry)
    rebuilt: list[Check] = app.build_registry(
        app.PSL, app.STORE, app.ALLOWLIST, app.LLM_CHECK
    ).enabled()
    assert [check.name for check in app.REGISTRY.enabled()] == [check.name for check in rebuilt]


def test_unregistered_list_holds_only_implemented_checks() -> None:
    """未註冊清單依載入結果算出：基底恆含 `domain_age`；`url_blocklist` 只在黑白名單
    失敗時出現、`llm_scam` 只在 LLM 未載入時出現。三者都是已實作、有「未註冊」語義
    的檢查，不會出現未實作檢查的名稱。"""
    names = {name for name, _label, _reason in app.UNREGISTERED_CHECKS}
    assert "domain_age" in names
    assert names <= {"domain_age", "url_blocklist", "llm_scam"}
    for _name, label, reason in app.UNREGISTERED_CHECKS:
        assert label.strip()
        assert reason.strip()
        assert "\n" not in reason


def test_domain_age_is_not_injected() -> None:
    assert "domain_age" not in {check.name for check in app.REGISTRY.enabled()}


def test_optional_dependencies_default_to_not_injected() -> None:
    """個資辨識器**不在此列**：四條辨識器是純標準庫、就在同一個 wheel 裡，
    沒有任何部署條件擋著它。潤飾層要模型檔，記錄器要一個記錄目的地。"""
    assert app.POLISHER is None
    assert app.TRANSCRIPT_LOGGER is None


def test_the_recognizer_is_mounted_on_this_side_too() -> None:
    """兩份介面層對同一份能力不該給出兩種答案 ——
    `docs/pages_app.py` 早就掛著同一個 `find_pii`。"""
    assert app.PII_RECOGNIZER is not None
    assert app.PII_RECOGNIZER is app.recognize_pii


def recognize_pii_body(path: Path) -> str:
    """兩份介面層的轉接函式最後那一行。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "recognize_pii":
            return ast.unparse(node.body[-1])
    raise AssertionError(f"{path} 裡沒有 recognize_pii")


def test_both_interface_layers_convert_the_spans_the_same_way() -> None:
    """不 import `docs/pages_app.py`：那一側在 import 階段就要 PSL 快照，
    而一個因環境缺件被跳過的測試等於沒有測試。比對的是原始碼。"""
    root = Path(app.__file__).resolve().parent
    assert recognize_pii_body(root / "app.py") == recognize_pii_body(root / "docs" / "pages_app.py")

    sentence = "身分證 A123456789 手機 0912345678。"
    expected = [(span.start, span.end, span.entity_type) for span in find_pii(sentence)]
    assert app.recognize_pii(sentence) == expected
    assert len(expected) == 2


def test_the_recognizer_is_a_module_level_named_function() -> None:
    """以 lambda 或巢狀 `def` 捕獲外層變數是本專案明文禁止的寫法 ——
    同一條規則在八顆範例按鈕上已經吃過一次虧。"""
    assert app.PII_RECOGNIZER.__name__ == "recognize_pii"
    tree = ast.parse(Path(app.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert not any(isinstance(child, ast.FunctionDef) for child in node.body)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign | ast.Assign) and "PII_RECOGNIZER" in ast.unparse(node)
    ]
    assert assignments
    for node in assignments:
        assert not isinstance(node.value, ast.Lambda)


def test_demo_builds_as_blocks_without_flagging() -> None:
    demo = app.build_demo()
    assert isinstance(demo, app.gr.Blocks)
    assert not isinstance(demo, app.gr.Interface)
    assert demo.analytics_enabled is False


def test_the_interface_does_not_try_to_lock_the_colour_scheme() -> None:
    """明暗模式鎖不住，唯一的解法是不寫死顏色 —— 所以這裡不該有任何嘗試。"""
    source = (app.SAMPLES_PATH.parent / "app.py").read_text(encoding="utf-8")
    for attempt in ("__theme", "theme_mode", "color-scheme", "gr.themes"):
        assert attempt not in source


def test_the_interface_layer_owns_no_second_copy_of_the_markup() -> None:
    """判定卡、偵測細節、氣泡與樣式表只有一份實作，住在 `demo_ui`。"""
    for name in (
        "render_verdict_row",
        "render_panel",
        "render_answer",
        "render_lines",
        "render_checks",
        "render_conversation",
        "check_state",
        "escaped",
        "anchor_id",
        "CSS",
        "STATE_LABELS",
    ):
        assert not hasattr(app, name), f"app.{name} 應該已經移入 demo_ui"


def test_document_coords_match_the_detected_document() -> None:
    """渲染用的 `Document` 與 `detect()` 內部的是同一個座標系。"""
    request = app.build_inquiry_request("第一句。第二句。")
    document = build_document(request.messages, app.LIMITS)
    card, echo = app.inquiry_submit("第一句。第二句。")
    assert f'id="{demo_ui.anchor_id(document.coords[1])}"' in echo
    assert card
