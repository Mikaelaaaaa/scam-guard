"""`browser_llm` —— 兩趟 `detect()`、重放式 runtime，與兩個模式的一輪。

**本檔不載入任何模型、不需要 GPU、不需要網路、不需要瀏覽器。**
那正是 `browser_llm.py` 不 import `js` / `pyodide` / `llm_runtime` / `llama_cpp`
的目的：JavaScript 那一側的知識全部留在 `docs/index.html`，這一側可以在一台
從來沒有下載過那 784 MB 的機器上跑完。

註冊表刻意不含 URL 層：那四個檢查需要 PSL 快照，而快照由建置腳本產生 ——
本檔要能在一個 `build/` 目錄不存在的 repo 上跑。
"""

import inspect
import json
import re
from pathlib import Path

import pytest

import browser_llm
from browser_llm import (
    InquiryMode,
    LlmState,
    PracticeMode,
    Presentation,
    ReplayRuntime,
    TwoPass,
)
from scam_guard.check import CheckRegistry, Stage
from scam_guard.llm.check import TIMEOUT_SENTINEL, LlmRuntime
from scam_guard.llm.prompt import DEFAULT_BUDGET, select_window
from scam_guard.llm.schema import LABELS
from scam_guard.llm.validate import SCAM_SIGNAL, LlmOutcome, LlmOutcomeCounter
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.pipeline import detect
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import CheckResult, Message, Request, ScamType
from scam_guard.weights import load_weights

TABLE = load_weights()

PHISHING = (
    "Chunghwa Post：包裹因關稅未繳而暫扣。請支付 159.71 元，付款請於此 ： https://e.vg/post-gov"
)
"""design 的 Context 那一則。規則層對它命中 0 項，而那是結構性的（四個分量跨句）。"""

SOURCE = Path(browser_llm.__file__).read_text(encoding="utf-8")

PRESENTATION = Presentation(
    table=TABLE,
    unregistered=(("domain_age", "網域年齡查詢", "需要向網域註冊局查詢，本頁不對外連線"),),
    sender_label="你貼上的訊息",
    reply_label="對方",
)


def rule_registry() -> CheckRegistry:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    return registry


def a_two_pass(counter: LlmOutcomeCounter | None = None) -> TwoPass:
    return TwoPass(
        registry=rule_registry(),
        table=TABLE,
        limits=DEFAULT_LIMITS,
        counter=LlmOutcomeCounter() if counter is None else counter,
    )


def an_output(
    ids: list[list[int]],
    label: str = LABELS[2],
    category: str | None = ScamType.FAKE_PARCEL.value,
    notes: str = "自稱郵政並要求付款",
) -> str:
    return json.dumps(
        {
            "analysis_notes": notes,
            "evidence_sentence_ids": ids,
            "category_165": category,
            "label": label,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# 重放式 runtime
# ---------------------------------------------------------------------------


def test_replay_runtime_signature_matches_the_protocol() -> None:
    """簽章與 `LlmRuntime.generate` 逐項相同：參數名、順序、keyword-only 與回傳型別。

    這條不是形式主義。`LlmCheck` 是以關鍵字傳 `deadline_s` 的，而參數名一改，
    失敗會發生在**執行期**、在一個已經下載完 784 MB 的瀏覽器裡。
    """
    expected = inspect.signature(LlmRuntime.generate)
    actual = inspect.signature(ReplayRuntime.generate)
    assert list(actual.parameters) == list(expected.parameters)
    for name, parameter in expected.parameters.items():
        assert actual.parameters[name].kind is parameter.kind
    assert inspect.get_annotations(ReplayRuntime.generate) == inspect.get_annotations(
        LlmRuntime.generate
    )


def test_replay_runtime_answers_once() -> None:
    runtime = ReplayRuntime("{}")
    assert runtime.generate("prompt", "grammar", deadline_s=1.0) == "{}"
    with pytest.raises(RuntimeError, match="只回答一次"):
        runtime.generate("prompt", "grammar", deadline_s=1.0)


@pytest.mark.parametrize(
    "raw",
    ['  {"a": 1}  ', '{"analysis_notes":"沒有收尾', "", "\n\n", TIMEOUT_SENTINEL],
)
def test_replay_runtime_returns_the_string_character_for_character(raw: str) -> None:
    """不 strip、不補齊、不修補。修補模型的輸出會讓一次失敗變成一個合法的錯答案。"""
    assert ReplayRuntime(raw).generate("p", "g", deadline_s=0.0) == raw


def test_replay_runtime_does_not_time_itself() -> None:
    """它不 import `time`，也不對 `deadline_s` 做任何比較 —— 生成已經發生完畢了。"""
    assert re.search(r"^\s*(?:import|from)\s+time\b", SOURCE, re.MULTILINE) is None
    assert "deadline_s <" not in SOURCE
    assert "deadline_s >" not in SOURCE


def test_ignoring_grammar_carries_an_explanation() -> None:
    """忽略 `grammar` 的那一段帶有說明為何忽略的註解。"""
    body = inspect.getsource(ReplayRuntime.generate)
    assert "沒有文法約束解碼" in body
    assert "grammar" in body


# ---------------------------------------------------------------------------
# 模組的界線
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["js", "pyodide", "llm_runtime", "llama_cpp"])
def test_browser_llm_does_not_import_the_browser_or_an_inference_engine(module: str) -> None:
    assert re.search(rf"^\s*(?:import|from)\s+{module}\b", SOURCE, re.MULTILINE) is None


def test_the_prefill_prefix_comes_from_the_schema_constant() -> None:
    """前綴由 `FIELD_NAMES` 產生，檔案裡沒有手寫的欄位名字面值。"""
    assert browser_llm.ASSISTANT_PREFILL == '{"analysis_notes":"'
    assert '"analysis_notes"' not in SOURCE
    assert "FIELD_NAMES[0]" in SOURCE


def test_the_first_pass_registry_may_not_contain_an_expensive_check() -> None:
    """第一趟不呼叫模型，而「不呼叫」由註冊表的形狀保證，不由執行順序保證。"""

    class Expensive:
        name = "llm_scam"
        stage = Stage.EXPENSIVE

        def __call__(self, req: Request, doc: object, *, prior: object = ()) -> list[CheckResult]:
            return []

    registry = rule_registry()
    registry.register(Expensive())
    with pytest.raises(ValueError, match="Stage.LOCAL"):
        TwoPass(registry=registry, table=TABLE, counter=LlmOutcomeCounter())


# ---------------------------------------------------------------------------
# 兩趟流程
# ---------------------------------------------------------------------------


def test_the_first_pass_matches_a_plain_detect() -> None:
    """模型未載入時的 `Verdict` 與直接呼叫 `detect()` 逐欄位相同。

    這是「模型層的存在 MUST NOT 改變純規則版的判定結果」那條 requirement 的形狀：
    第一趟就是本 change 之前的那一次 `detect()`，一個位元組都沒有多。
    """
    request = Request.from_text(PHISHING)
    registry = rule_registry()
    expected = detect(request, registry, TABLE, limits=DEFAULT_LIMITS)
    actual = a_two_pass().first_pass(request).verdict

    assert actual.scam_probability == expected.scam_probability
    assert actual.confidence == expected.confidence
    assert actual.scam_type == expected.scam_type
    assert actual.evidence == expected.evidence
    assert {result.name for result in actual.checks if result.hit} == {
        result.name for result in expected.checks if result.hit
    }


def test_both_passes_see_the_same_window() -> None:
    """兩趟的 `PromptWindow.coords` 逐項相同。

    兩段 prompt 文字**不**相同（nonce 每次重新產生），而那沒有關係：
    證據座標是拿視窗驗的，而視窗只由 `Document` 與 `PromptBudget` 決定。
    """
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    second = two_pass.second_pass(first.token, an_output([[0, 0]]))
    assert first.window_coords == tuple(select_window(second.document, DEFAULT_BUDGET).coords)


def test_a_coordinate_in_the_first_window_is_accepted_by_the_second_pass() -> None:
    """視窗一致的行為證據：第一趟視窗裡的座標，第二趟驗得過。"""
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    ids = [list(coord) for coord in first.window_coords]
    assert two_pass.second_pass(first.token, an_output(ids)).outcome is LlmOutcome.OK


def test_a_coordinate_outside_the_window_is_semantic_failure() -> None:
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    assert two_pass.second_pass(first.token, an_output([[99, 7]])).outcome is LlmOutcome.SEMANTIC


def test_the_second_pass_only_adds() -> None:
    """單調升級：信心不降低，且第一趟命中的每個訊號都還在第二趟的命中集合裡。"""
    two_pass = a_two_pass()
    request = Request(
        messages=[
            Message(text="我是檢察官，你的帳戶涉及洗錢。"),
            Message(text="請把簡訊驗證碼給我，不要告訴家人。"),
        ]
    )
    first = two_pass.first_pass(request)
    second = two_pass.second_pass(first.token, an_output([[0, 0]], category=None))

    assert second.verdict.confidence >= first.verdict.confidence
    before = {result.name for result in first.verdict.checks if result.hit}
    after = {result.name for result in second.verdict.checks if result.hit}
    assert before <= after


def test_a_broken_object_is_structure_and_leaves_the_verdict_alone() -> None:
    """模型少寫一個 `}`，串接結果照原樣送進驗證 —— 不補括號。"""
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    truncated = browser_llm.ASSISTANT_PREFILL + "包裹詐騙"
    second = two_pass.second_pass(first.token, truncated)
    assert second.outcome is LlmOutcome.STRUCTURE
    assert second.verdict.confidence == first.verdict.confidence


def test_timeout_sentinel_is_recorded_as_timeout_and_changes_nothing() -> None:
    counter = LlmOutcomeCounter()
    two_pass = a_two_pass(counter)
    request = Request.from_text(PHISHING)
    first = two_pass.first_pass(request)
    second = two_pass.second_pass(first.token, TIMEOUT_SENTINEL)

    assert second.outcome is LlmOutcome.TIMEOUT
    assert counter.counts()[LlmOutcome.TIMEOUT] == 1
    assert counter.counts()[LlmOutcome.STRUCTURE] == 0

    plain = detect(request, rule_registry(), TABLE, limits=DEFAULT_LIMITS)
    assert second.verdict.scam_probability == plain.scam_probability
    assert second.verdict.confidence == plain.confidence
    assert second.verdict.scam_type == plain.scam_type
    assert second.verdict.evidence == plain.evidence


def test_without_model_records_nothing() -> None:
    """一次沒有發生的判讀不該出現在失敗率的分母裡。"""
    counter = LlmOutcomeCounter()
    two_pass = a_two_pass(counter)
    first = two_pass.first_pass(Request.from_text(PHISHING))
    second = two_pass.without_model(first.token)
    assert second.outcome is None
    assert counter.total() == 0
    assert second.verdict.confidence == first.verdict.confidence


# ---------------------------------------------------------------------------
# 一次性識別字
# ---------------------------------------------------------------------------


def test_the_token_expires_after_one_use() -> None:
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    two_pass.second_pass(first.token, an_output([[0, 0]]))
    with pytest.raises(ValueError, match=re.escape(first.token)):
        two_pass.second_pass(first.token, an_output([[0, 0]]))


def test_resubmitting_invalidates_the_previous_token() -> None:
    """使用者改了輸入重新送出：先前那次的生成回來時撞上一個已作廢的識別字。"""
    two_pass = a_two_pass()
    stale = two_pass.first_pass(Request.from_text(PHISHING))
    two_pass.first_pass(Request.from_text("明天見"))
    with pytest.raises(ValueError, match=re.escape(stale.token)):
        two_pass.second_pass(stale.token, an_output([[0, 0]]))


def test_an_unknown_token_raises_instead_of_passing_off_the_rule_layer_result() -> None:
    two_pass = a_two_pass()
    two_pass.first_pass(Request.from_text(PHISHING))
    with pytest.raises(ValueError, match="deadbeef"):
        two_pass.second_pass("deadbeef", an_output([[0, 0]]))


def test_the_second_pass_signature_carries_no_message_text() -> None:
    """待判讀的文字 MUST NOT 第二次過界：簽章裡只有識別字與模型輸出。"""
    for method in (TwoPass.second_pass, TwoPass.without_model):
        names = list(inspect.signature(method).parameters)
        assert "text" not in names
        assert "request" not in names
        assert "messages" not in names
    assert list(inspect.signature(TwoPass.second_pass).parameters) == ["self", "token", "raw"]
    assert list(inspect.signature(TwoPass.without_model).parameters) == ["self", "token"]


# ---------------------------------------------------------------------------
# 掛上這一層之後，那則釣魚訊息**仍然**不會被判為詐騙
# ---------------------------------------------------------------------------


def test_the_model_layer_supplies_a_type_and_confidence_but_not_a_decision() -> None:
    """這條測試存在的目的是釘住一件事：**掛上模型也不會讓那則訊息被判為詐騙**。

    `weights.toml` 的 `base_single_group = 0.35` 低於 `confidence_floor = 0.40`，
    單一群組命中依設計必定拒答。LLM 這一層在今天的權重表下補的是
    **類型、依據與信心，不是判定**。要跨過門檻需要第二個群組命中，
    或需要 `add-testset` / `add-metrics` 取代那張表裡的佔位值 ——
    兩者都不在 `add-browser-llm` 的範圍內。

    這條測試若有一天紅了，先確認是不是權重表改了，不要改這條斷言。
    """
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    assert first.verdict.scam_probability is None
    assert first.verdict.scam_type is None
    assert [result for result in first.verdict.checks if result.hit] == []

    second = two_pass.second_pass(first.token, an_output([[0, 0]]))
    assert second.outcome is LlmOutcome.OK
    assert second.verdict.scam_probability is None
    assert second.verdict.confidence == pytest.approx(0.35)
    assert second.verdict.scam_type is ScamType.FAKE_PARCEL
    assert [result.name for result in second.verdict.checks if result.hit] == [SCAM_SIGNAL]


# ---------------------------------------------------------------------------
# 畫面上的狀態
# ---------------------------------------------------------------------------


def test_the_four_states_are_four_different_sentences() -> None:
    texts = [browser_llm.first_pass_status(state) for state in LlmState]
    assert len(set(texts)) == len(texts)


def test_no_state_can_be_read_as_the_model_found_nothing() -> None:
    """三種失敗、三種沒有跑，與「模型判為無話術」是六段互不相同的文字。"""
    failures = [
        browser_llm.STATUS_STRUCTURE,
        browser_llm.STATUS_SEMANTIC,
        browser_llm.STATUS_TIMEOUT,
        browser_llm.STATUS_NOT_LOADED,
        browser_llm.STATUS_LOADING,
        browser_llm.STATUS_LOAD_FAILED,
        browser_llm.STATUS_SKIPPED,
    ]
    assert browser_llm.STATUS_NO_SIGNAL not in failures
    assert len(set([*failures, browser_llm.STATUS_NO_SIGNAL, browser_llm.STATUS_HIT])) == 9
    for text in failures:
        assert "沒有詐騙話術" not in text
        assert "判為" not in text


def test_a_successful_reading_leaves_the_unregistered_list_alone() -> None:
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    second = two_pass.second_pass(first.token, an_output([[0, 0]]))

    listed = browser_llm.first_pass_unregistered(PRESENTATION, LlmState.READY)
    assert [row[0] for row in listed] == ["domain_age", SCAM_SIGNAL]

    after = browser_llm.second_pass_unregistered(PRESENTATION, LlmState.READY, second)
    assert after == PRESENTATION.unregistered


def test_a_failed_reading_stays_on_the_unregistered_list_with_its_own_reason() -> None:
    two_pass = a_two_pass()
    first = two_pass.first_pass(Request.from_text(PHISHING))
    second = two_pass.second_pass(first.token, "不是 JSON")
    rows = browser_llm.second_pass_unregistered(PRESENTATION, LlmState.READY, second)
    assert rows[-1][0] == SCAM_SIGNAL
    assert "JSON" in rows[-1][2]


# ---------------------------------------------------------------------------
# 模式一
# ---------------------------------------------------------------------------


def test_the_inquiry_card_is_complete_before_any_generation() -> None:
    mode = InquiryMode(a_two_pass(), PRESENTATION)
    first = mode.first(Request.from_text(PHISHING), LlmState.READY)
    assert '<div class="card">' in first["card"]
    assert first["prompt"]
    assert first["status"] == browser_llm.STATUS_RUNNING


def test_an_empty_document_yields_no_prompt() -> None:
    """貼圖、純空白：正規化後一句都不剩，沒有東西可以判讀，也就不要生成。"""
    mode = InquiryMode(a_two_pass(), PRESENTATION)
    first = mode.first(Request.from_text("　"), LlmState.NOT_LOADED)
    assert first["prompt"] == ""
    assert first["status"] == browser_llm.STATUS_NOT_LOADED


# ---------------------------------------------------------------------------
# 模式二：對練
# ---------------------------------------------------------------------------


def a_practice() -> PracticeMode:
    return PracticeMode(
        a_two_pass(),
        Presentation(
            table=TABLE,
            unregistered=(),
            sender_label="你（扮演詐騙方）",
            reply_label="對方",
        ),
    )


def test_the_first_pass_of_a_round_cannot_contain_a_victim_line() -> None:
    """判定卡與排行在受害方台詞之前完成更新 —— 而這由**回傳值的形狀**保證。

    驗證順序而不是最終畫面：一個「等模型跑完再一起更新」的實作在最終畫面上
    仍然正確，只是慢，而那種實作在這裡會紅，因為 `first()` 根本沒有台詞可以給。
    """
    practice = a_practice()
    first = practice.first("請把簡訊驗證碼給我，不要告訴家人。", LlmState.READY)
    assert "polish_prompt" not in first
    assert "polish_status" not in first
    assert '<div class="bubble me">' not in first["conversation"]
    assert "第 1 輪" in first["ranking"]

    second = practice.second(first["token"], an_output([[0, 0]], category=None))
    assert '<div class="bubble me">' in second["conversation"]


def test_the_victim_line_and_the_validator_share_one_verdict() -> None:
    """同源的行為證據：驗證器不可能拒絕 `victim_reply()` 自己產出的那一句。

    用兩個不同的 `Verdict` 時這條會紅 —— 那正是「判定卡說是某個類型、
    而潤飾層禁止提到該類型」那個矛盾的形狀。
    """
    practice = a_practice()
    first = practice.first("我是檢察官，你的帳戶涉及洗錢，請領錢交給書記官。", LlmState.READY)
    practice.second(first["token"], an_output([[0, 0]], category=ScamType.FAKE_AUTHORITY.value))
    baseline = practice._replies[-1]
    assert baseline
    assert practice.polish_feed(baseline)["ok"] is True


def test_without_model_the_two_still_share_one_verdict() -> None:
    practice = a_practice()
    first = practice.first("請把簡訊驗證碼給我。", LlmState.NOT_LOADED)
    second = practice.without_model(first["token"], LlmState.NOT_LOADED)
    assert second["polish_status"] == browser_llm.demo_ui.POLISH_NOT_INJECTED
    assert practice.polish_feed(practice._replies[-1])["ok"] is True


def test_a_violating_chunk_aborts_and_the_whole_line_falls_back() -> None:
    """違規即中止，整句退回，狀態文字是「已整句丟棄」那一句。"""
    practice = a_practice()
    first = practice.first("請把簡訊驗證碼給我，不要告訴家人。", LlmState.READY)
    second = practice.second(first["token"], an_output([[0, 0]], category=None))
    baseline = practice._replies[-1]

    assert practice.polish_feed("我覺得")["ok"] is True
    violating = practice.polish_feed("有 87% 的機率是詐騙")
    assert violating["ok"] is False
    assert violating["polish_status"] == browser_llm.demo_ui.POLISH_DISCARDED
    assert practice._replies[-1] == baseline
    assert "87" not in violating["conversation"]
    assert second["polish_status"] == browser_llm.demo_ui.POLISH_STREAMING

    with pytest.raises(ValueError, match="沒有進行中的潤飾"):
        practice.polish_feed("後面還有")


def test_an_accepted_polish_replaces_the_line() -> None:
    practice = a_practice()
    first = practice.first("請把簡訊驗證碼給我，不要告訴家人。", LlmState.READY)
    practice.second(first["token"], an_output([[0, 0]], category=None))
    assert practice.polish_feed("你叫我不要跟家人講，我還是會講。")["ok"] is True
    ended = practice.polish_end()
    assert ended["polish_status"] == browser_llm.demo_ui.POLISH_ACCEPTED
    assert "你叫我不要跟家人講" in ended["conversation"]


def test_the_polish_prompt_carries_the_victim_line_and_nothing_else() -> None:
    """潤飾的輸入只有受害方那一句，**沒有訊息原文**。"""
    practice = a_practice()
    spoken = "請把簡訊驗證碼給我，不要告訴家人。"
    first = practice.first(spoken, LlmState.READY)
    second = practice.second(first["token"], an_output([[0, 0]], category=None))
    assert spoken not in second["polish_prompt"]
    assert practice._replies[-1] in second["polish_prompt"]


def test_a_round_accumulates_messages() -> None:
    practice = a_practice()
    for text in ("你好。", "我是檢察官。", "請把驗證碼給我。"):
        first = practice.first(text, LlmState.NOT_LOADED)
        practice.without_model(first["token"], LlmState.NOT_LOADED)
    assert len(practice._messages) == 3
    assert len(practice._replies) == 3
    assert "第 4 輪" in practice.first("再一則。", LlmState.NOT_LOADED)["ranking"]


def test_an_empty_turn_is_refused() -> None:
    with pytest.raises(ValueError, match="輸入為空"):
        a_practice().first("   ", LlmState.NOT_LOADED)


# ---------------------------------------------------------------------------
# 計數器
# ---------------------------------------------------------------------------


def test_outcome_counts_has_all_four_keys_before_anything_has_run() -> None:
    """畫面在第一次判讀之前就要能顯示這一塊，而 `failure_rate()` 在那時會拋例外。"""
    assert set(browser_llm.outcome_counts()) == {outcome.value for outcome in LlmOutcome}


def test_the_registry_the_second_pass_builds_keeps_every_rule_check() -> None:
    """第二趟的註冊表是規則層加一個 `LlmCheck`，不是換掉一批。"""
    registry = rule_registry()
    two_pass = TwoPass(registry=registry, table=TABLE, counter=LlmOutcomeCounter())
    first = two_pass.first_pass(Request.from_text(PHISHING))
    second = two_pass.second_pass(first.token, an_output([[0, 0]]))
    rule_names = {check.name for check in registry.enabled()}
    reported = {result.name for result in second.verdict.checks}
    assert rule_names <= reported


def test_build_document_is_redone_in_the_second_pass() -> None:
    """第二趟重新建立 `Document`，不沿用第一趟的物件。"""
    two_pass = a_two_pass()
    request = Request.from_text(PHISHING)
    first = two_pass.first_pass(request)
    second = two_pass.second_pass(first.token, an_output([[0, 0]]))
    assert second.document is not first.document
    assert second.document.coords == build_document(request.messages, DEFAULT_LIMITS).coords
