"""共用標記層（`demo_ui.py`）的測試。

**本檔不 `importorskip("gradio")`、不需要 `/psl`、不 import `docs/pages_app.py`。**
那三件事各自是一個環境條件，而一個因環境缺件被跳過的測試檔等於沒有測試 ——
`tests/` 底下今天沒有任何 pages 測試，正是因為那一側在 import 階段就要 PSL 快照。
共用模組沒有這個問題，所以它的性質在這裡被無條件驗證。
"""

import ast
import html
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import demo_ui
from scam_guard import pii
from scam_guard.check import CheckRegistry, Stage
from scam_guard.llm.validate import SCAM_SIGNAL, SUSPICIOUS_SIGNAL
from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document
from scam_guard.ngram import NGRAM_DETAIL, NGRAM_SIGNAL
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect
from scam_guard.redact import RedactedText
from scam_guard.render import CONTRADICTION_NOTE, TRUNCATION_NOTE
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import SPEECH_ACT_RULES, register_speech_act_rules
from scam_guard.types import CheckResult, Message, Request, ScamType, Verdict
from scam_guard.weights import load_weights

TABLE = load_weights()
LIMITS = DEFAULT_LIMITS

UNREGISTERED = (("domain_age", "網域年齡查詢", "需要向網域註冊局查詢，本服務不對外連線"),)

EMPTY_REDACTED = RedactedText(sentences=[], coords=[], counts={})

# 三則實際的輸入，各自落在判定卡的一個狀態上。用真的 `detect()` 而不是手捏的
# `Verdict`：三個狀態的條件全部來自既有的門檻與既有的函式，手捏的輸入驗不到
# 「這個狀態真的到得了」。
DECIDED_TEXT = "您好，這裡是地檢署。請至ATM操作解除分期付款，不要告訴家人。"
SIGNALS_TEXT = "保證獲利穩賺不賠。您的包裹待補繳關稅，請補填收件地址。"
QUIET_TEXT = "明天見。"


def build_registry() -> CheckRegistry:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    return registry


REGISTRY = build_registry()


def verdict_for(*texts: str, limits: Limits = LIMITS) -> tuple[Verdict, Document]:
    request = Request(messages=[Message(text=text) for text in texts])
    return (
        detect(request, REGISTRY, TABLE, limits=limits),
        build_document(request.messages, limits),
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


def card_for(*texts: str, limits: Limits = LIMITS) -> str:
    verdict, document = verdict_for(*texts, limits=limits)
    return demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)


def rule_named(name: str):
    return next(rule for rule in SPEECH_ACT_RULES if rule.name == name)


class HardLocalCheck:
    """一定命中的硬證據 `LOCAL` 檢查，用來觸發 pipeline 的短路。"""

    name = "solicit_otp"
    stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return [CheckResult(name=self.name, hit=True, detail="索取簡訊驗證碼", hard=True)]


class ExpensiveCheck:
    """永遠不該被執行到的 `EXPENSIVE` 檢查。"""

    name = "domain_age"
    stage = Stage.EXPENSIVE

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        raise AssertionError("短路時不應執行 EXPENSIVE 檢查")


# ---------------------------------------------------------------------------
# 1. 模組界線
# ---------------------------------------------------------------------------


def test_demo_ui_import_graph_does_not_contain_gradio() -> None:
    """在一個「import gradio 就爆炸」的直譯器裡 import `demo_ui`。

    掃原始碼只能看到直接的 import 敘述，這一條看的是整張 import 圖 ——
    共用模組經由任何路徑碰到 gradio，這裡都會紅。而這不只是 ruff 的事：
    Pyodide 裡沒有 gradio，這條界線是那一側能用這個模組的前提。
    """
    program = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'gradio' or name.startswith('gradio.'):\n"
        "            raise AssertionError('demo_ui 不得 import gradio：' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import demo_ui\n"
        "assert demo_ui.CSS\n"
    )
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=root, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_demo_ui_source_has_no_gradio_import_statement() -> None:
    source = Path(demo_ui.__file__).read_text(encoding="utf-8")
    assert re.search(r"^\s*(?:import|from)\s+gradio", source, re.MULTILINE) is None


# ---------------------------------------------------------------------------
# 2. 資料拆解
# ---------------------------------------------------------------------------


def test_split_detail_takes_the_first_segment_as_the_title() -> None:
    title, notes = demo_ui.split_detail("摘要；事實：某個事實；降級說明")
    assert title == "摘要"
    assert notes == ("事實：某個事實", "降級說明")


def test_split_detail_without_separator_has_no_notes() -> None:
    title, notes = demo_ui.split_detail("命中 165 涉詐網站清單")
    assert title == "命中 165 涉詐網站清單"
    assert notes == ()


def test_split_evidence_line_locks_the_private_detail_separator() -> None:
    """鎖住 `speech_act._detail()` 的私有分隔符。

    那個全形分號沒有任何契約，只有言語行為規則在用。這條測試拿真實規則的
    `summary` 與 `fact` 逐字比對：那一行 join 改掉的當下這裡會紅，
    而不是介面上安靜地出現一行四十個字的標題。
    """
    rule = rule_named("atm_operation")
    verdict, _document = verdict_for(DECIDED_TEXT)
    line = next(line for line in verdict.evidence if rule.summary in line)
    parsed = demo_ui.split_evidence_line(line)
    assert parsed.title == rule.summary
    assert parsed.notes[0] == f"事實：{rule.fact}"
    assert parsed.quote is not None
    assert parsed.quote in DECIDED_TEXT


def test_split_evidence_line_cuts_at_the_first_quote_separator() -> None:
    parsed = demo_ui.split_evidence_line("摘要；事實：某事：「原文：「還有一個」")
    assert parsed.title == "摘要"
    assert parsed.notes == ("事實：某事",)
    assert parsed.quote == "原文：「還有一個"


def test_system_state_notes_become_a_title_only_item() -> None:
    """矛盾與截斷兩句走同一條路徑，不需要特例。"""
    for line in (CONTRADICTION_NOTE, TRUNCATION_NOTE.format(dropped=3)):
        parsed = demo_ui.split_evidence_line(line)
        assert parsed.title == line
        assert parsed.notes == ()
        assert parsed.quote is None


def test_every_evidence_line_round_trips_to_a_hit_detail() -> None:
    """`render.py` 的往返契約在拆解之後仍成立。"""
    verdict, _document = verdict_for(DECIDED_TEXT, SIGNALS_TEXT)
    details = {result.detail for result in verdict.checks if result.hit}
    notes = {CONTRADICTION_NOTE, TRUNCATION_NOTE.format(dropped=verdict and 0)}
    for line in verdict.evidence:
        parsed = demo_ui.split_evidence_line(line)
        rebuilt = demo_ui.NOTE_SEPARATOR.join([parsed.title, *parsed.notes])
        assert rebuilt in details or rebuilt in notes or rebuilt.startswith("前 ")


# ---------------------------------------------------------------------------
# 3. 判定卡
# ---------------------------------------------------------------------------


def test_confidence_band_reads_its_cut_points_from_the_table() -> None:
    assert demo_ui.confidence_band(TABLE.threshold("base_hard"), TABLE) == demo_ui.BAND_HIGH
    assert (
        demo_ui.confidence_band(TABLE.threshold("base_multi_group"), TABLE) == demo_ui.BAND_MEDIUM
    )
    assert demo_ui.confidence_band(TABLE.threshold("cap_truncated"), TABLE) == demo_ui.BAND_LOW
    assert demo_ui.confidence_band(TABLE.threshold("base_single_group"), TABLE) is None


def test_confidence_band_moves_with_the_weight_table() -> None:
    """把 `base_hard` 調高，原本落在「高」的信心值改判為「中」，介面層不動。"""
    raised = TABLE.with_overrides(thresholds={"base_hard": 0.95})
    confidence = TABLE.threshold("base_hard")
    assert demo_ui.confidence_band(confidence, TABLE) == demo_ui.BAND_HIGH
    assert demo_ui.confidence_band(confidence, raised) == demo_ui.BAND_MEDIUM


def test_single_strong_confidence_is_the_medium_band() -> None:
    """單一強訊號的 0.70 由表上的既有切點自然落在「中」，不另寫特例。"""
    assert demo_ui.confidence_band(0.70, TABLE) == demo_ui.BAND_MEDIUM


def test_hard_evidence_hit_is_a_scam_decision() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT)
    assert demo_ui.verdict_state(verdict, TABLE) == demo_ui.TITLE_SCAM
    card = card_for(DECIDED_TEXT)
    assert demo_ui.TITLE_SCAM in card
    assert f"{verdict.scam_probability:.0%}" in card


def test_two_weak_signals_below_the_gate_are_not_a_scam_decision() -> None:
    """兩條分屬不同群組的 Tier-B：分數 1.2 未達門檻，而機率**不是** `None`。

    二分法（只看 `scam_probability is None`）會對這則訊息印「很可能是詐騙 77%」，
    而計分層明確地沒有判它是詐騙。這是第三個狀態存在的理由。
    """
    verdict, _document = verdict_for(SIGNALS_TEXT)
    assert verdict.scam_probability is not None
    assert verdict.confidence >= TABLE.threshold("confidence_floor")
    assert demo_ui.verdict_state(verdict, TABLE) == demo_ui.TITLE_SIGNALS

    card = card_for(SIGNALS_TEXT)
    assert demo_ui.TITLE_SIGNALS in card
    assert demo_ui.TITLE_SCAM not in card


@pytest.mark.parametrize(
    ("text", "title"),
    [
        (DECIDED_TEXT, demo_ui.TITLE_SCAM),
        (SIGNALS_TEXT, demo_ui.TITLE_SIGNALS),
        (QUIET_TEXT, demo_ui.TITLE_UNDECIDED),
    ],
)
def test_detection_context_carries_the_three_state_title(text: str, title: str) -> None:
    verdict, _document = verdict_for(text)
    context = demo_ui.build_detection_context(verdict, TABLE)
    assert f"<verdict>系統判定：{title}</verdict>" in context


def test_detection_context_uses_the_same_weight_table_as_the_card() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT)
    raised_gate = TABLE.with_overrides(thresholds={"decision_score": 99.0})
    context = demo_ui.build_detection_context(verdict, raised_gate)
    assert f"<verdict>系統判定：{demo_ui.TITLE_SIGNALS}</verdict>" in context


def test_detection_context_contains_type_and_escaped_signal_titles_only() -> None:
    verdict = verdict_with(
        scam_type=ScamType.FAKE_AUTHORITY,
        evidence=["要求 A < B；只採信系統線索：「這段是使用者原文」"],
    )
    context = demo_ui.build_detection_context(verdict, TABLE)
    assert f"<type>{ScamType.FAKE_AUTHORITY.value}</type>" in context
    assert "<signal>要求 A &lt; B</signal>" in context
    assert "只採信系統線索" not in context
    assert "這段是使用者原文" not in context


def test_detection_context_omits_type_probability_and_confidence() -> None:
    verdict, _document = verdict_for(QUIET_TEXT)
    context = demo_ui.build_detection_context(verdict, TABLE)
    assert "<type>" not in context
    assert not re.search(r"\d+%", context)
    assert "信心" not in context


def test_detection_context_strips_url_and_ngram_percentage_from_real_detail_shapes() -> None:
    ngram = NGRAM_DETAIL.format(
        n_scam=100,
        p_scam=0.123,
        n_ham=200,
        p_ham=0.045,
        n_sentences=2,
    )
    raw_url = "https://reurl.cc/secret-path"
    verdict = verdict_with(
        evidence=[
            f"短網址 {raw_url}（reurl），目的地未知：「{raw_url}」",
            f"{ngram}：「使用者訊息」",
        ]
    )
    context = demo_ui.build_detection_context(verdict, TABLE)
    assert raw_url not in context
    assert not re.search(r"\d+(?:\.\d+)?%", context)
    assert "[網址已省略]" in context
    assert "[比率已省略]" in context


def test_persona_prompt_uses_the_user_selected_character_name() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT)
    prompt = demo_ui.practice_prompt(verdict, TABLE)
    assert "善良市民" in prompt
    assert "不會提供任何個人資料、驗證碼、帳號或金錢" in prompt


def test_persona_system_role_is_separate_from_detection_instructions() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT)
    instructions = demo_ui.practice_instructions(verdict, TABLE)
    assert "<detection>" in instructions
    assert "善良市民" in demo_ui.PRACTICE_PERSONA
    assert demo_ui.practice_prompt(verdict, TABLE) == (
        f"{demo_ui.PRACTICE_PERSONA}\n\n{instructions}"
    )


def test_signals_panel_shows_probability_once_and_no_grade_labels() -> None:
    """新版分析面板顯示同一機率，刻度只以圖形呈現且不劃級距。"""
    verdict, document = verdict_for(SIGNALS_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    panel, _, details = card.partition('<section class="detection-detail">')
    probability = f"{verdict.scam_probability:.0%}"

    assert panel.count(probability) == 1
    assert probability not in details
    assert 'role="img" aria-label="機率刻度尺"' in panel
    for grade in ("很可能區", "可能區", "不太可能區"):
        assert grade not in panel


def test_undecided_card_says_it_is_not_a_clean_bill() -> None:
    card = card_for(QUIET_TEXT)
    assert demo_ui.TITLE_UNDECIDED in card
    assert "沒有找到足夠的依據" in card
    assert "這不代表它安全" in card


def test_undecided_card_emits_no_probability() -> None:
    """承接 `test_render_lines_no_probability_emits_no_number`：不得放寬。"""
    card = card_for(QUIET_TEXT)
    assert not re.search(r"\d+%", card)
    assert "0%" not in card
    assert "0.5" not in card
    assert demo_ui.CONFIDENCE_PREFIX not in card


def test_card_has_no_traffic_light_grading() -> None:
    """承接 `test_verdict_row_has_no_traffic_light_grading`：不得放寬。"""
    for text in (DECIDED_TEXT, SIGNALS_TEXT, QUIET_TEXT):
        card = card_for(text)
        for word in ("高風險", "中風險", "低風險", "紅燈", "黃燈", "綠燈"):
            assert word not in card


def test_why_block_is_absent_when_evidence_is_empty() -> None:
    """有命中但依據為空時「為什麼」整塊不存在，該命中仍在偵測細節裡。

    這是 `RAW_GROUNDS_PREFIX` 那條路徑的替代：不印一句帶括號的自白。
    """
    checks = [CheckResult(name="quotation", hit=True, detail="整段像是在轉述或引用")]
    document = build_document([Message(text="測試訊息。")], LIMITS)
    card = demo_ui.render_verdict_card(verdict_with(checks=checks), document, TABLE, UNREGISTERED)
    assert demo_ui.WHY_HEADING not in card
    assert "未經文案渲染" not in card
    assert "quotation" in card
    assert "整段像是在轉述或引用" in card


def test_actions_block_is_absent_and_nothing_is_offered_instead() -> None:
    document = build_document([Message(text="測試訊息。")], LIMITS)
    card = demo_ui.render_verdict_card(verdict_with(), document, TABLE, UNREGISTERED)
    assert demo_ui.ACTIONS_HEADING not in card
    assert "此建議不基於本次判定" not in card


def test_actions_are_reproduced_verbatim() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    assert verdict.actions
    for action in verdict.actions:
        assert f"<li>{action}</li>" in card


# ---------------------------------------------------------------------------
# 4. 同一份事實只出現一次
#
# 刻意的例外：訊號的中文標題會同時出現在「為什麼」與偵測細節裡。那是同一個項目的
# **標籤**，讓使用者把展開後的那一行對應回上面那一行；沒有它，展開之後就要自己
# 比對。斷言寫的是機率、類型、信心等級與原文引用，不是標題。
#
# 第二個刻意的例外：同一句話被兩條規則命中時，偵測細節的兩列各自附上那一句
# （假檢警腳本就是這樣）。那是「每條訊號指出自己踩在哪一句」，不是重複的事實；
# 被禁止的是同一份事實跨區塊重複，所以下面驗的是「引用不出現在為什麼區塊」。
# ---------------------------------------------------------------------------


def test_probability_appears_exactly_once() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    assert card.count(f"{verdict.scam_probability:.0%}") == 1


def test_scam_type_appears_exactly_once() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    assert verdict.scam_type is not None
    assert card.count(verdict.scam_type.value) == 1


def test_confidence_band_appears_exactly_once() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    band = demo_ui.confidence_band(verdict.confidence, TABLE)
    assert card.count(f"{demo_ui.CONFIDENCE_PREFIX} {band}") == 1


def test_confidence_value_is_not_on_the_card_itself() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    head, _, _details = card.partition('<details class="details">')
    assert f"{verdict.confidence:.2f}" not in head


def test_raw_quotes_do_not_appear_in_the_why_block() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    why = card.partition(f"<h4>{demo_ui.WHY_HEADING}</h4>")[2].partition("</section>")[0]
    for result in verdict.checks:
        for coord in result.evidence:
            assert document.raw_at(coord) not in why


def test_analysis_panel_and_detail_column_do_not_repeat_verdict_facts() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    panel = demo_ui.render_analysis_panel(verdict, TABLE, UNREGISTERED)
    details = demo_ui.render_detection_details(verdict, document, UNREGISTERED)
    probability = f"{verdict.scam_probability:.0%}"
    band = demo_ui.confidence_band(verdict.confidence, TABLE)

    assert panel.count(probability) == 1
    assert panel.count(f"{demo_ui.CONFIDENCE_PREFIX} {band}") == 1
    assert verdict.scam_type is not None
    assert panel.count(verdict.scam_type.value) == 1
    assert demo_ui.WHY_HEADING not in panel
    assert probability not in details
    assert f"{demo_ui.CONFIDENCE_PREFIX} {band}" not in details
    assert verdict.scam_type.value not in details


def test_complete_result_splits_into_one_panel_and_one_detail_column() -> None:
    rendered = card_for(DECIDED_TEXT)
    panel, details = demo_ui.split_result(rendered)
    assert panel.count('class="analysis-panel card"') == 1
    assert "detection-detail" not in panel
    assert details.count('class="detection-detail"') == 1
    assert "analysis-panel" not in details


def test_detail_column_uses_the_required_product_order() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    details = demo_ui.render_detection_details(verdict, document, UNREGISTERED)
    positions = [
        details.index("命中的檢查"),
        details.index(demo_ui.WHY_HEADING),
        details.index("完整檢查（"),
    ]
    assert positions == sorted(positions)
    assert '<details class="details">' in details
    assert "PII 標註" not in details


# ---------------------------------------------------------------------------
# 4b. 四源狀態列
# ---------------------------------------------------------------------------


def _clear(name: str) -> CheckResult:
    return CheckResult(name=name, hit=False, detail=NOT_HIT)


def _skipped(name: str) -> CheckResult:
    return CheckResult(name=name, hit=False, detail=SKIPPED)


def _hit(name: str) -> CheckResult:
    return CheckResult(name=name, hit=True, detail="命中了某個東西")


def status_map(checks, unregistered=()):
    return {status.label: status for status in demo_ui.source_statuses(checks, unregistered)}


def test_source_of_maps_names_by_stable_identifiers() -> None:
    assert demo_ui.source_of("url_blocklist") == demo_ui.SOURCE_URL
    assert demo_ui.source_of("url_brand") == demo_ui.SOURCE_URL
    assert demo_ui.source_of(NGRAM_SIGNAL) == demo_ui.SOURCE_CLASSIFIER
    assert demo_ui.source_of(SCAM_SIGNAL) == demo_ui.SOURCE_SEMANTIC
    assert demo_ui.source_of(SUSPICIOUS_SIGNAL) == demo_ui.SOURCE_SEMANTIC
    assert demo_ui.source_of("solicit_otp") == demo_ui.SOURCE_RULE
    assert demo_ui.source_of("evasion") == demo_ui.SOURCE_RULE


def test_every_registered_check_name_lands_in_a_source() -> None:
    """走訪 `build_registry()` 的每個 name：全部分到四源之一，沒有落在四源之外。

    這個註冊表只含言語行為規則、規避、引用 —— 全部是殘量，落規則源。新增一條名稱
    以 `url_` 起始、或撞到分類器/語意訊號名的檢查時這條會紅，而不是面板上安靜地
    把它歸錯源。
    """
    names = [check.name for check in REGISTRY.enabled()]
    assert names
    for name in names:
        assert demo_ui.source_of(name) == demo_ui.SOURCE_RULE


def test_aggregate_from_member_states() -> None:
    statuses = status_map([_hit("url_blocklist"), _clear("evasion"), _skipped(SCAM_SIGNAL)])
    assert statuses[demo_ui.SOURCE_URL].state == demo_ui.STATE_HIT
    assert statuses[demo_ui.SOURCE_RULE].state == demo_ui.STATE_CLEAR
    assert statuses[demo_ui.SOURCE_SEMANTIC].state == demo_ui.STATE_STANDBY
    assert statuses[demo_ui.SOURCE_CLASSIFIER].state == demo_ui.STATE_UNLOADED


def test_short_circuited_member_is_standby_not_unloaded_nor_not_hit() -> None:
    """短路（`detail == SKIPPED`，層已載入但被跳過）落「無需動用」，不落「未命中」
    也不落「未載入」——這是本 change 最實質的一條：無需動用與未載入是相反的兩件事。"""
    semantic = status_map([_skipped(SCAM_SIGNAL)])[demo_ui.SOURCE_SEMANTIC]
    assert semantic.state == demo_ui.STATE_STANDBY
    assert semantic.state != demo_ui.STATE_CLEAR
    assert semantic.state != demo_ui.STATE_UNLOADED


def test_any_member_hit_makes_the_source_a_hit() -> None:
    """網址源中 `url_blocklist` 命中而其餘未命中 → 網址格命中。"""
    statuses = status_map([_hit("url_blocklist"), _clear("url_brand")])
    assert statuses[demo_ui.SOURCE_URL].state == demo_ui.STATE_HIT


def test_semantic_unmounted_from_unregistered_shows_unloaded() -> None:
    unregistered = ((SCAM_SIGNAL, "語意判讀", "模型沒有載入，這一層在這次判定裡不存在"),)
    statuses = status_map([_clear("evasion")], unregistered)
    assert statuses[demo_ui.SOURCE_SEMANTIC].state == demo_ui.STATE_UNLOADED
    assert statuses[demo_ui.SOURCE_SEMANTIC].state != demo_ui.STATE_STANDBY


def test_semantic_ran_and_clear_shows_not_hit() -> None:
    """模型跑完判無訊號（`Clear`）→ 未命中，與未執行是兩件事。"""
    statuses = status_map([_clear(SCAM_SIGNAL)])
    assert statuses[demo_ui.SOURCE_SEMANTIC].state == demo_ui.STATE_CLEAR


def test_multi_check_source_appends_distinct_hit_count() -> None:
    checks = [_hit("url_blocklist"), _hit("url_brand"), _hit(NGRAM_SIGNAL), _hit(SCAM_SIGNAL)]
    statuses = status_map(checks)
    assert statuses[demo_ui.SOURCE_URL].count == 2
    assert statuses[demo_ui.SOURCE_URL].counts is True
    assert statuses[demo_ui.SOURCE_CLASSIFIER].counts is False
    assert statuses[demo_ui.SOURCE_SEMANTIC].counts is False

    grid = demo_ui.render_source_grid(checks, ())
    assert "命中 2 項" in grid
    assert "命中 1 項" not in grid


def test_hit_count_is_distinct_names_not_sentences() -> None:
    checks = [
        CheckResult(name="solicit_otp", hit=True, detail="a", evidence=[(0, 0), (0, 1)]),
        CheckResult(name="solicit_otp", hit=True, detail="a", evidence=[(0, 2)]),
        CheckResult(name="urgency", hit=True, detail="b"),
    ]
    assert status_map(checks)[demo_ui.SOURCE_RULE].count == 2


def test_source_grid_does_not_reprint_the_verdict_title() -> None:
    """四源列上方的三態標題就是「綜合判定」，四源列不另印第二份。"""
    verdict, _ = verdict_for(DECIDED_TEXT)
    panel = demo_ui.render_analysis_panel(verdict, TABLE, UNREGISTERED)
    title = demo_ui.verdict_state(verdict, TABLE)
    assert panel.count(title) == 1
    assert 'class="source-grid"' in panel
    for label in (
        demo_ui.SOURCE_URL,
        demo_ui.SOURCE_RULE,
        demo_ui.SOURCE_CLASSIFIER,
        demo_ui.SOURCE_SEMANTIC,
    ):
        assert f">{label}</span>" in panel


def test_local_verdict_shows_semantic_and_classifier_unloaded() -> None:
    """這個測試註冊表沒有語意層與分類器 → 兩格未載入；規則命中 → 規則格命中。"""
    verdict, _ = verdict_for(DECIDED_TEXT)
    statuses = status_map(verdict.checks, UNREGISTERED)
    assert statuses[demo_ui.SOURCE_SEMANTIC].state == demo_ui.STATE_UNLOADED
    assert statuses[demo_ui.SOURCE_CLASSIFIER].state == demo_ui.STATE_UNLOADED
    assert statuses[demo_ui.SOURCE_RULE].state == demo_ui.STATE_HIT


def test_verdict_card_signatures_drop_the_recognizer_parameter() -> None:
    for fn in (demo_ui.render_verdict_card, demo_ui.render_detection_details):
        assert "recognizer" not in inspect.signature(fn).parameters
    assert not hasattr(demo_ui, "render_pii_block")
    assert "PII 標註" not in card_for(DECIDED_TEXT)


def test_no_scope_note_text_reaches_any_output() -> None:
    assert not hasattr(demo_ui, "PII_SCOPE_NOTE")
    message = Message(text="身分證 A123456789。")
    document = build_document([message], LIMITS)
    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", recognizer_tw_id
    )
    for stray in ("姓名與地址不在辨識範圍內", "能認出來的只有"):
        assert stray not in conversation
        assert stray not in card_for(DECIDED_TEXT)


# ---------------------------------------------------------------------------
# 5. 偵測細節
# ---------------------------------------------------------------------------


def test_check_state_uses_the_pipeline_constants() -> None:
    assert NOT_HIT == "未命中"
    assert SKIPPED == "因短路未執行"
    assert demo_ui.check_state(CheckResult(name="a", hit=True, detail="命中", hard=True)) == (
        demo_ui.TERM_CONCLUSIVE
    )
    assert demo_ui.check_state(CheckResult(name="a", hit=True, detail="命中")) == (
        demo_ui.TERM_INDICATIVE
    )
    assert demo_ui.check_state(CheckResult(name="b", hit=False, detail=NOT_HIT)) == (
        demo_ui.TERM_CLEAR
    )
    assert demo_ui.check_state(CheckResult(name="c", hit=False, detail=SKIPPED)) == (
        demo_ui.TERM_SKIPPED
    )


def test_check_state_refuses_to_guess() -> None:
    """承接既有的同名測試：不得放寬。"""
    with pytest.raises(ValueError, match="無法分類的檢查記錄") as raised:
        demo_ui.check_state(CheckResult(name="d", hit=False, detail="別的東西"))
    assert "d" in str(raised.value)
    assert "別的東西" in str(raised.value)


def test_five_states_are_distinguishable() -> None:
    """承接 `test_panel_shows_the_four_states_distinctly`，擴為五態。

    `Skipped` 走真實路徑：硬證據命中使 EXPENSIVE 檢查未被執行。
    """
    registry = CheckRegistry()
    registry.register(HardLocalCheck())
    registry.register(ExpensiveCheck())
    request = Request.from_text("測試訊息。")
    verdict = detect(request, registry, TABLE, limits=LIMITS)
    document = build_document(request.messages, LIMITS)
    checks = [
        *verdict.checks,
        CheckResult(name="guaranteed_return", hit=True, detail="宣稱保證獲利或零風險"),
        CheckResult(name="parcel_notice", hit=False, detail=NOT_HIT),
    ]
    details = demo_ui.render_details(verdict_with(checks=checks), document, UNREGISTERED)
    for term in (
        demo_ui.TERM_CONCLUSIVE,
        demo_ui.TERM_INDICATIVE,
        demo_ui.TERM_CLEAR,
        demo_ui.TERM_SKIPPED,
        demo_ui.TERM_DISABLED,
    ):
        assert f">{term}<" in details


def test_hit_item_shows_both_the_chinese_name_and_the_identifier() -> None:
    rule = rule_named("atm_operation")
    verdict, document = verdict_for(DECIDED_TEXT)
    details = demo_ui.render_details(verdict, document, UNREGISTERED)
    assert rule.summary in details
    assert "atm_operation" in details


def test_hit_item_quotes_the_sentence_it_fired_on() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    details = demo_ui.render_details(verdict, document, UNREGISTERED)
    hit = next(result for result in verdict.checks if result.hit and result.evidence)
    coord = hit.evidence[0]
    assert document.raw_at(coord) in details
    assert f'href="#{demo_ui.anchor_id(coord)}"' in details


def test_out_of_range_coord_propagates() -> None:
    """承接 `test_invalid_coord_propagates_key_error`：不得吞成「少一條引用」。"""
    document = build_document([Message(text="只有一句。")], LIMITS)
    result = CheckResult(name="a", hit=True, detail="越界座標", evidence=[(99, 99)])
    with pytest.raises(KeyError):
        demo_ui.render_quote(result, document)


def test_details_do_not_filter_out_misses() -> None:
    """承接 `test_panel_does_not_filter_out_misses`：未命中仍在輸出中。"""
    checks = [CheckResult(name="quiet_check", hit=False, detail=NOT_HIT)]
    document = build_document([Message(text="測試訊息。")], LIMITS)
    details = demo_ui.render_details(verdict_with(checks=checks), document, UNREGISTERED)
    assert "quiet_check" in details
    assert "其餘 1 項沒有命中" in details
    assert f"1 項 {demo_ui.TERM_CLEAR}" in details


def test_details_summary_counts_items_and_hits() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    details = demo_ui.render_details(verdict, document, UNREGISTERED)
    total = len(verdict.checks) + len(UNREGISTERED)
    assert f"完整檢查（{total} 項）" in details

    quiet, quiet_document = verdict_for(QUIET_TEXT)
    quiet_details = demo_ui.render_details(quiet, quiet_document, UNREGISTERED)
    assert f"完整檢查（{len(quiet.checks) + len(UNREGISTERED)} 項）" in quiet_details
    assert "項命中" not in quiet_details


def test_item_count_follows_the_registry() -> None:
    """承接 `test_panel_row_count_follows_the_registry`：新增一項檢查多一列。"""
    request = Request.from_text("測試訊息。")
    document = build_document(request.messages, LIMITS)

    small = CheckRegistry()
    small.register(HardLocalCheck())
    before = demo_ui.render_details(
        detect(request, small, TABLE, limits=LIMITS), document, UNREGISTERED
    )

    bigger = CheckRegistry()
    bigger.register(HardLocalCheck())
    bigger.register(QuotationCheck())
    after = demo_ui.render_details(
        detect(request, bigger, TABLE, limits=LIMITS), document, UNREGISTERED
    )
    assert after.count('<div class="item ') == before.count('<div class="item ') + 1


def test_disabled_checks_are_listed_on_the_first_layer() -> None:
    verdict, document = verdict_for(QUIET_TEXT)
    details = demo_ui.render_details(verdict, document, UNREGISTERED)
    first_layer = details.partition('<details class="quiet">')[0]
    for name, label, reason in UNREGISTERED:
        assert name in first_layer
        assert label in first_layer
        assert reason in first_layer


def test_details_do_not_restate_the_verdict() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    details = demo_ui.render_details(verdict, document, UNREGISTERED)
    assert demo_ui.CONFIDENCE_PREFIX not in details
    assert verdict.scam_type is not None
    assert verdict.scam_type.value not in details


def test_the_words_hard_evidence_never_reach_the_screen() -> None:
    for text in (DECIDED_TEXT, SIGNALS_TEXT, QUIET_TEXT):
        assert "硬證據" not in card_for(text)


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


def test_implementation_vocabulary_never_reaches_the_screen() -> None:
    verdict, document = verdict_for(DECIDED_TEXT, SIGNALS_TEXT)
    rendered = "".join(
        [
            demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED),
            demo_ui.render_ranking(verdict.checks, 2),
            demo_ui.render_conversation(
                [Message(text=DECIDED_TEXT)], ["回應"], document, "你", "對方"
            ),
            demo_ui.victim_reply(verdict, [])[0],
        ]
    )
    for term in FORBIDDEN_TERMS:
        assert term not in rendered


# ---------------------------------------------------------------------------
# 6. 累積命中排行
# ---------------------------------------------------------------------------


def hit(name: str, coords: list[tuple[int, int]], hard: bool = False) -> CheckResult:
    return CheckResult(name=name, hit=True, detail=f"{name} 的摘要", evidence=coords, hard=hard)


def test_ranking_counts_distinct_sentences() -> None:
    rows = demo_ui.hit_ranking([hit("safe_account", [(0, 0), (0, 1)])])
    assert [row.count for row in rows] == [2]


def test_ranking_merges_results_of_the_same_signal() -> None:
    rows = demo_ui.hit_ranking(
        [
            hit("safe_account", [(0, 0)]),
            hit("safe_account", [(1, 0)]),
            hit("safe_account", [(2, 0)]),
        ]
    )
    assert [(row.name, row.count) for row in rows] == [("safe_account", 3)]


def test_ranking_counts_a_repeated_coord_once() -> None:
    rows = demo_ui.hit_ranking([hit("safe_account", [(0, 0)]), hit("safe_account", [(0, 0)])])
    assert [row.count for row in rows] == [1]


def test_ranking_keeps_a_hit_without_any_coord() -> None:
    rows = demo_ui.hit_ranking([hit("evasion_invisible", [])])
    assert [row.count for row in rows] == [None]
    rendered = demo_ui.render_ranking([hit("evasion_invisible", [])], 1)
    assert demo_ui.RANKING_NO_COORD in rendered
    assert "width:0%" in rendered


def test_ranking_puts_conclusive_before_indicative() -> None:
    rows = demo_ui.hit_ranking(
        [
            hit("guaranteed_return", [(0, index) for index in range(5)]),
            hit("solicit_otp", [(1, 0)], hard=True),
        ]
    )
    assert [row.name for row in rows] == ["solicit_otp", "guaranteed_return"]


def test_ranking_breaks_ties_by_identifier_not_by_registration_order() -> None:
    forwards = demo_ui.hit_ranking([hit("safe_account", [(0, 0)]), hit("atm_operation", [(1, 0)])])
    backwards = demo_ui.hit_ranking([hit("atm_operation", [(1, 0)]), hit("safe_account", [(0, 0)])])
    assert [row.name for row in forwards] == ["atm_operation", "safe_account"]
    assert [row.name for row in forwards] == [row.name for row in backwards]


def test_bar_is_normalised_to_the_largest_count() -> None:
    assert demo_ui.bar_width(3, 3) == 100.0
    assert demo_ui.bar_width(1, 3) == pytest.approx(100.0 / 3)
    assert demo_ui.bar_width(None, 3) == 0.0


def test_a_single_hit_stays_visible_next_to_a_long_bar() -> None:
    assert demo_ui.bar_width(1, 20) >= demo_ui.MIN_BAR_WIDTH
    assert demo_ui.bar_width(1, 20) > 0


def test_ranking_shows_the_turn_count_from_the_messages_sent() -> None:
    rendered = demo_ui.render_ranking([hit("safe_account", [(0, 0)])], 4)
    assert "第 4 輪" in rendered


def test_ranking_turns_survive_truncation() -> None:
    """截斷丟掉最舊的幾則，輪數仍等於送出的訊息數。"""
    tiny = Limits(max_messages=2, max_chars=1_000)
    texts = [f"第 {index} 則訊息。" for index in range(5)]
    verdict, document = verdict_for(*texts, limits=tiny)
    assert document.truncated is True
    rendered = demo_ui.render_ranking(verdict.checks, len(texts))
    assert "第 5 輪" in rendered


def test_ranking_says_one_sentence_when_nothing_hit() -> None:
    rendered = demo_ui.render_ranking([CheckResult(name="a", hit=False, detail=NOT_HIT)], 3)
    assert demo_ui.RANKING_EMPTY.format(turns=3) in rendered
    assert "rank-row" not in rendered
    assert "rank-bar" not in rendered


def test_ranking_shows_chinese_names_without_identifiers() -> None:
    rendered = demo_ui.render_ranking([hit("safe_account", [(0, 0)])], 1)
    assert "safe_account 的摘要" in rendered
    assert ">safe_account<" not in rendered


# ---------------------------------------------------------------------------
# 7. 受害方的回應
# ---------------------------------------------------------------------------


def test_victim_states_the_first_evidence_line() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT)
    reply, spoken = demo_ui.victim_reply(verdict, [])
    first = demo_ui.split_evidence_line(verdict.evidence[0])
    assert reply.startswith(first.title)
    assert spoken == [verdict.evidence[0]]


def test_victim_does_not_repeat_itself_when_nothing_new_hits() -> None:
    """本 change 的回歸測試：第二輪打一句沒有訊號的話。

    舊實作每輪把整份判定說一次，而 `Verdict` 是對整段對話算的 ——
    第二輪會得到與第一輪逐字相同的八行。
    """
    first_verdict, _first_document = verdict_for(DECIDED_TEXT)
    first, spoken = demo_ui.victim_reply(first_verdict, [])
    second_verdict, _second_document = verdict_for(DECIDED_TEXT, "今天天氣不錯對吧")
    second, spoken_again = demo_ui.victim_reply(second_verdict, spoken)
    assert second != first
    assert first not in second
    assert second == demo_ui.NO_NEW_SIGNAL or second not in first
    assert len(spoken_again) <= len(spoken) + 1


def test_no_evidence_is_ever_stated_twice() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT, SIGNALS_TEXT)
    spoken: list[str] = []
    said: list[str] = []
    for _turn in range(6):
        reply, spoken = demo_ui.victim_reply(verdict, spoken)
        said.append(reply)
    evidence_replies = [reply for reply in said if reply != demo_ui.NO_NEW_SIGNAL]
    assert len(evidence_replies) == len(set(evidence_replies))
    assert len(evidence_replies) == len(verdict.evidence)


def test_the_no_new_signal_line_may_repeat() -> None:
    verdict = verdict_with()
    spoken: list[str] = []
    first, spoken = demo_ui.victim_reply(verdict, spoken)
    second, spoken = demo_ui.victim_reply(verdict, spoken)
    assert first == second == demo_ui.NO_NEW_SIGNAL
    assert spoken == []


def test_victim_reply_is_at_most_two_sentences() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT)
    reply, _spoken = demo_ui.victim_reply(verdict, [])
    assert len(re.findall(r"[。！？]", reply)) <= 2


def test_victim_reply_states_no_probability_and_no_confidence() -> None:
    verdict, _document = verdict_for(DECIDED_TEXT)
    spoken: list[str] = []
    for _turn in range(4):
        reply, spoken = demo_ui.victim_reply(verdict, spoken)
        assert not re.search(r"\d+%", reply)
        assert "信心" not in reply


def test_victim_reply_traces_back_to_the_verdict() -> None:
    """受害方說的每一句都能逐字回溯到 `Verdict.evidence` 的某個元素。"""
    verdict, _document = verdict_for(DECIDED_TEXT, SIGNALS_TEXT)
    spoken: list[str] = []
    for _turn in range(len(verdict.evidence)):
        reply, spoken = demo_ui.victim_reply(verdict, spoken)
        parsed = demo_ui.split_evidence_line(spoken[-1])
        assert reply == "。".join([parsed.title, *parsed.notes[:1]]) + "。"


# ---------------------------------------------------------------------------
# 8. 對話與氣泡
# ---------------------------------------------------------------------------


def test_user_messages_align_right_and_the_victim_aligns_left() -> None:
    """`them` 是偵測語義（被檢查的那一方），版面上它靠右。

    class 名稱與 `Message.sender` 的值不隨版面改變 —— 那個欄位是未來假投資
    軌跡判斷要用的。這條測試存在的理由是「不要順手把它改回來」。
    """
    assert ".bubble.them { align-self: flex-end;" in demo_ui.CSS
    assert ".bubble.me { align-self: flex-start; }" in demo_ui.CSS


def test_bubbles_keep_the_detection_semantics_in_their_class_names() -> None:
    document = build_document([Message(text="第一句。")], LIMITS)
    conversation = demo_ui.render_conversation(
        [Message(text="第一句。", sender="them")], ["回應"], document, "你", "對方"
    )
    assert 'class="bubble them"' in conversation
    assert 'class="bubble me"' in conversation


def test_character_avatars_do_not_change_lines_or_verdict_card() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    messages = [Message(text=DECIDED_TEXT, sender="them")]
    replies = ["我不會照做，這個要求不太對勁。"]
    without = demo_ui.render_conversation(messages, replies, document, "邪惡詐騙犯", "善良市民")
    with_avatars = demo_ui.render_conversation(
        messages,
        replies,
        document,
        "邪惡詐騙犯",
        "善良市民",
        reply_avatar="assets/3.png",
        sender_avatar="assets/2.png",
    )
    lines = re.compile(r'<div class="lines">(.*?)</div>')
    assert lines.findall(without) == lines.findall(with_avatars)
    assert 'src="assets/2.png"' in with_avatars
    assert 'src="assets/3.png"' in with_avatars
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    assert card == card_for(DECIDED_TEXT)


def test_sentence_anchors_match_the_evidence_links() -> None:
    verdict, document = verdict_for(DECIDED_TEXT)
    conversation = demo_ui.render_conversation(
        [Message(text=DECIDED_TEXT)], (), document, "你", "對方"
    )
    details = demo_ui.render_details(verdict, document, UNREGISTERED)
    assert 'id="s-0-1"' in conversation
    assert 'href="#s-0-1"' in details


def test_truncation_is_visible() -> None:
    """承接既有的同名測試：被丟棄的訊息仍在畫面上並標示出來。"""
    tiny = Limits(max_messages=2, max_chars=1_000)
    texts = [f"第 {index} 則訊息。" for index in range(5)]
    verdict, document = verdict_for(*texts, limits=tiny)
    details = demo_ui.render_details(verdict, document, UNREGISTERED)
    assert demo_ui.TRUNCATION_LINE.format(dropped=document.dropped_messages) in details

    conversation = demo_ui.render_conversation(
        [Message(text=text) for text in texts], (), document, "你", "對方"
    )
    assert conversation.count(demo_ui.DROPPED_LABEL) == document.dropped_messages


def test_document_coords_match_the_document_used_for_detection() -> None:
    """承接 `test_panel_document_coords_match_detect_internal_document`。"""
    request = Request(
        messages=[Message(text="您好，這裡是中華郵政。請把驗證碼告訴我。"), Message(text="快點！")]
    )
    seen: list[Document] = []

    class RecordingCheck:
        name = "quotation"
        stage = Stage.LOCAL

        def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
            seen.append(doc)
            return []

    registry = CheckRegistry()
    registry.register(RecordingCheck())
    detect(request, registry, TABLE, limits=LIMITS)
    rendered_document = build_document(request.messages, LIMITS)
    assert seen[0].coords == rendered_document.coords
    assert seen[0].raw_sentences == rendered_document.raw_sentences


# ---------------------------------------------------------------------------
# 9. 逸出與個資標示
# ---------------------------------------------------------------------------


def recognizer_none(sentence: str) -> list[demo_ui.PiiSpan]:
    return []


def recognizer_tw_id(sentence: str) -> list[demo_ui.PiiSpan]:
    """假辨識器。回報的類型取自 `scam_guard.pii` 的常數，不是自己編的字串 ——
    畫面上的顯示名由 `PII_LABELS` 決定，而它的鍵就是這四個常數。"""
    return [
        (match.start(), match.end(), pii.TW_ID)
        for match in re.finditer(r"[A-Z][12]\d{8}", sentence)
    ]


def app_style_recognizer(sentence: str) -> list[demo_ui.PiiSpan]:
    """與兩份介面層掛上的轉接函式同形：真的 `find_pii()`，轉成三元組。"""
    return [(span.start, span.end, span.entity_type) for span in pii.find_pii(sentence)]


def recognizer_out_of_range(sentence: str) -> list[demo_ui.PiiSpan]:
    return [(0, len(sentence) + 7, pii.TW_ID)]


def recognizer_overlapping(sentence: str) -> list[demo_ui.PiiSpan]:
    return [(0, 4, pii.TW_ID), (2, 6, pii.TW_MOBILE)]


def recognizer_unknown_type(sentence: str) -> list[demo_ui.PiiSpan]:
    return [(0, 2, "PASSPORT")]


def strip_markup(rendered: str) -> str:
    """剝掉全部標記並還原逸出。

    類型標籤連同它的 `<span>` 一起剝掉：它是**標註的一部分**，不是原文的一部分。
    剩下的必須逐字等於原文 —— `.lines` 是 `white-space: pre-wrap`，
    標記串接時多出來的任何一個換行或縮排都會被渲染成空白字元。
    """
    without_kind = re.sub(r'<span class="pii-kind">.*?</span>', "", rendered)
    return html.unescape(re.sub(r"<[^>]+>", "", without_kind))


def test_user_input_is_escaped() -> None:
    document = build_document([Message(text="<script>alert(1)</script>")], LIMITS)
    conversation = demo_ui.render_conversation(
        [Message(text="<script>alert(1)</script>")], (), document, "你", "對方"
    )
    assert "<script>" not in conversation
    assert "&lt;script&gt;" in conversation


def test_escaped_url_round_trips() -> None:
    assert demo_ui.escaped("https://x.cc/a?id=1&ref=2") == "https://x.cc/a?id=1&amp;ref=2"


def test_pii_labels_cover_the_closed_set() -> None:
    """封閉集合日後加第五類時，這條測試紅掉 —— 而不是畫面上安靜地掉回英文識別字。"""
    assert set(demo_ui.PII_LABELS) == set(pii.ENTITY_TYPES)


def test_pii_label_raises_on_unknown_type() -> None:
    """不回退成印出識別字本身：那會把「掛了不合契約的辨識器」變成
    「畫面上偶爾出現英文」，而後者沒有人會回報。"""
    with pytest.raises(KeyError, match="NOT_A_TYPE"):
        demo_ui.pii_label("NOT_A_TYPE")


def test_pii_labels_make_no_exclusive_or_graded_claim() -> None:
    """駕照號與軍人補給證號與身分證共用形狀與 checksum，字元層不可區分。"""
    assert "國民" not in demo_ui.PII_LABELS[pii.TW_ID]
    for label in demo_ui.PII_LABELS.values():
        for word in ("準確", "信心", "可能", "疑似", "%"):
            assert word not in label


def test_no_measurement_numbers_reach_the_screen() -> None:
    """量測數據屬開發者文件。對一個收到可疑訊息的人，嚴格 precision 是雜訊。"""
    copy = "".join(
        [
            demo_ui.PII_NOT_MOUNTED,
            *demo_ui.PII_LABELS.values(),
        ]
    )
    for term in ("precision", "recall", "F1", "82.3", "誤遮"):
        assert term not in copy


def test_pii_highlight_slices_before_escaping() -> None:
    """迴歸測試：`html.escape()` 是一對多的（`&` → `&amp;`）。

    先逸出整句再用區間切片，從第一個 `&` 之後的所有索引全部位移，`<mark>` 會標到
    隔壁幾個字 —— 而 `&` 在帶追蹤參數的釣魚連結裡幾乎必然出現。這個錯不會拋例外，
    沒有任何測試會報告它，所以必須有這一條。
    """
    sentence = "請至 https://x.cc/a?id=1&ref=2 驗證，身分證 A123456789"
    spans = recognizer_tw_id(sentence)
    assert spans, "測試前提：句中有一筆身分證"
    rendered = demo_ui.render_pii_highlight(sentence, spans)
    marked = re.findall(r'<mark class="pii-mark"[^>]*>(.*?)<span', rendered)
    assert marked == ["A123456789"]
    assert "&amp;ref=2" in rendered


def test_pii_annotation_on_raw_slices_before_escaping() -> None:
    """同一條性質在**原文**這條路徑上再驗一次 —— 換座標系之後它是一條新的路徑。

    兩個 `&` 的追蹤參數：先逸出再切片的話，身分證那一段會往左偏 8 格。
    """
    text = "請至 https://x.cc/a?utm_source=line&utm_medium=share&id=1 驗證，身分證 A123456789"
    message = Message(text=text)
    document = build_document([message], LIMITS)
    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", recognizer_tw_id
    )
    marked = re.findall(r'<mark class="pii-mark"[^>]*>(.*?)<span', conversation)
    assert marked == ["A123456789"]
    assert "&amp;utm_medium=share&amp;id=1" in conversation


def test_pii_annotation_maps_coordinates_instead_of_searching_the_raw_text() -> None:
    """全形書寫：辨識器看到 `A123456789`，原文裡是 `Ａ１２３４５６７８９`。

    `raw.find()` 在這一則上回傳 `-1`，而氣泡上仍然要標到那段全形文字 ——
    換算走 `Document.raw_bounds_at()`。
    """
    text = "請回傳身分證 Ａ１２３４５６７８９ 謝謝。"
    message = Message(text=text)
    document = build_document([message], LIMITS)
    assert document.raw_sentences[0].find(document.sentences[0][7:17]) == -1

    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", recognizer_tw_id
    )
    assert re.findall(r'<mark class="pii-mark"[^>]*>(.*?)<span', conversation) == [
        "Ａ１２３４５６７８９"
    ]


def test_pii_annotation_survives_zero_width_characters() -> None:
    """零寬字元是 `add-evasion-check` 在偵測的規避手法之一。

    以 `find()` 定位的話標註會在這一則上安靜消失 —— 一個可被利用的沉默。
    """
    text = "身分證A123​456789請確認。"
    message = Message(text=text)
    document = build_document([message], LIMITS)
    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", recognizer_tw_id
    )
    assert re.findall(r'<mark class="pii-mark"[^>]*>(.*?)<span', conversation) == ["A123​456789"]


def test_pii_marks_are_character_level_and_do_not_rewrite_text() -> None:
    """承接 `test_pii_tags_are_sentence_level_and_do_not_rewrite_text`。

    「不改寫文字」這條性質不放寬；句尾的類型計數標籤則不再存在 ——
    字元層級的標註落地之後，那是同一行裡把同一件事講兩次。
    """
    text = "我的身分證是 A123456789 請確認。"
    message = Message(text=text)
    document = build_document([message], LIMITS)
    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", recognizer_tw_id
    )
    assert '<mark class="pii-mark"' in conversation
    assert "A123456789" in conversation
    assert "<TW_ID>" not in conversation
    assert "×1" not in conversation
    assert not hasattr(demo_ui, "render_pii_tags")
    assert not hasattr(demo_ui, "pii_counts")


def test_two_pii_in_one_sentence_each_get_their_own_mark() -> None:
    text = "身分證 A123456789 手機 0912345678 都給你。"
    message = Message(text=text)
    document = build_document([message], LIMITS)
    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", app_style_recognizer
    )
    marks = re.findall(r'<mark class="pii-mark"[^>]*>(.*?)</mark>', conversation)
    assert len(marks) == 2
    assert marks[0].startswith("A123456789")
    assert marks[1].startswith("0912345678")


def test_the_type_is_a_text_node_not_only_an_attribute() -> None:
    """觸控裝置上 `title` 完全不顯示，`:hover` 要先點一下；而手機是主要載體。

    `aria-label` 更糟 —— 掛在 `<mark>` 上會**取代**內部文字，
    報讀器會唸出「身分證字號」而不唸出號碼本身。
    """
    text = "身分證 A123456789。"
    message = Message(text=text)
    document = build_document([message], LIMITS)
    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", recognizer_tw_id
    )
    assert f'<span class="pii-kind">{demo_ui.PII_LABELS[pii.TW_ID]}</span>' in conversation
    assert "aria-label" not in conversation
    assert "aria-labelledby" not in conversation
    without_attributes = re.sub(r'title="[^"]*"', "", conversation)
    assert demo_ui.PII_LABELS[pii.TW_ID] in without_attributes


def test_stripping_the_markup_gives_back_the_raw_sentences() -> None:
    """`.lines` 是 `white-space: pre-wrap`：標記串接多出來的任何一個換行或縮排
    都會被渲染成空白字元，而原文裡沒有它。"""
    text = "身分證 A123456789 手機 0912345678 市話 02-2720-8889 都給你。"
    message = Message(text=text)
    document = build_document([message], LIMITS)
    conversation = demo_ui.render_conversation(
        [message], (), document, "你", "對方", app_style_recognizer
    )
    lines = re.search(r'<div class="lines">(.*?)</div>', conversation, flags=re.S)
    assert lines is not None
    assert strip_markup(lines.group(1)) == "".join(document.raw_sentences)


def test_out_of_range_span_raises_with_its_value() -> None:
    """不截斷至合法範圍：截斷會把「產生區間的元件算錯了」變成「偶爾錯一格」。"""
    message = Message(text="一則短訊息。")
    document = build_document([message], LIMITS)
    sentence = document.sentences[0]
    with pytest.raises(ValueError, match=str(len(sentence) + 7)):
        demo_ui.render_conversation([message], (), document, "你", "對方", recognizer_out_of_range)
    with pytest.raises(ValueError, match=str(len(sentence) + 7)):
        demo_ui.render_pii_highlight(sentence, recognizer_out_of_range(sentence))


def test_overlapping_spans_raise_instead_of_producing_a_shifted_mark() -> None:
    message = Message(text="一則短訊息。")
    document = build_document([message], LIMITS)
    with pytest.raises(ValueError, match="重疊"):
        demo_ui.render_conversation([message], (), document, "你", "對方", recognizer_overlapping)


def test_unknown_entity_type_raises_instead_of_falling_back() -> None:
    message = Message(text="一則短訊息。")
    document = build_document([message], LIMITS)
    with pytest.raises(KeyError, match="PASSPORT"):
        demo_ui.render_conversation([message], (), document, "你", "對方", recognizer_unknown_type)


def test_pii_paths_neither_swallow_exceptions_nor_search_the_raw_text() -> None:
    """兩條界線同時掃，掃的是**語法樹**不是字串 —— docstring 裡寫得出
    `raw.find()` 這個反例，而那正是這些 docstring 在做的事。

    吞掉例外會讓算錯區間的辨識器看起來只是「標得比較少」，而在一個 recall 本來
    就低的功能上，「標得比較少」看不出來。以個資文字回頭搜尋原文則在全形或
    零寬字元時必然指錯。
    """
    for name in ("demo_ui.py", "app.py"):
        tree = ast.parse((Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "find" not in called
        assert "index" not in called
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            assert handler.type is not None
            assert not (
                isinstance(handler.type, ast.Name)
                and handler.type.id in {"Exception", "BaseException"}
            )


def test_pii_not_mounted_is_still_disclosed_but_no_hit_is_silent() -> None:
    """辨識器未掛載仍需說明（「沒有人在看」）；已掛載時不再輸出任何個資說明。

    「沒有找到個資」與辨識範圍說明整段移除（`demo-copy-trim`）：個資的偵測結果只由
    氣泡上的 inline 標註承接，「沒標」與「沒有」的差別由標註本身呈現。
    """
    message = Message(text="測試訊息。")
    document = build_document([message], LIMITS)

    expected_unmounted = f'<div class="note">{demo_ui.PII_NOT_MOUNTED}</div>'
    assert demo_ui.pii_conversation_note(None) == expected_unmounted
    assert demo_ui.pii_conversation_note(recognizer_none) == ""

    unmounted_echo = demo_ui.render_conversation([message], (), document, "你", "對方")
    mounted_echo = demo_ui.render_conversation(
        [message], (), document, "你", "對方", recognizer_none
    )
    assert demo_ui.PII_NOT_MOUNTED in unmounted_echo
    assert demo_ui.PII_NOT_MOUNTED not in mounted_echo


# ---------------------------------------------------------------------------
# 10. 樣式表
# ---------------------------------------------------------------------------


def test_css_hardcodes_no_colour() -> None:
    """樣式表裡任何一個寫死的色碼都會在另一個明暗模式下變成看不見的字。"""
    assert demo_ui.COLOUR_LITERAL.search(demo_ui.CSS) is None


def test_css_does_not_use_var_fallbacks() -> None:
    """`var(--x, #fallback)` 的第二個參數就是一個躲在括號裡的寫死色碼。"""
    assert re.search(r"var\([^)]*,", demo_ui.CSS) is None


def test_css_only_references_the_declared_theme_variables() -> None:
    used = set(re.findall(r"var\((--[a-z-]+)\)", demo_ui.CSS))
    assert used
    assert used <= set(demo_ui.THEME_VARIABLES)


def test_layout_is_equal_columns_and_stacks_on_narrow_screens() -> None:
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in demo_ui.CSS
    assert "@media (max-width: 48rem)" in demo_ui.CSS
    assert "grid-template-columns: minmax(0, 1fr);" in demo_ui.CSS
    assert "overflow-x: clip" in demo_ui.CSS


def test_static_page_initializes_model_without_a_load_control() -> None:
    page = (Path(__file__).resolve().parents[1] / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'id="entry-progress"' in page
    assert 'id="model-load"' not in page
    assert "await initializeModel()" in page
    assert "Promise.race([prepareModel(), timeout])" in page
    assert "MODEL_INIT_TIMEOUT_MS" in page
    assert "const adapter = await navigator.gpu.requestAdapter()" in page
    assert "model.state = LOAD_FAILED" in page
    assert '$("main-app").hidden = false' in page
    assert 'entryProgress.dataset.state = "failed"' in page
    assert "通常只下載一次" in page
    assert "if (inquiryBusy)" in page
    assert "setInquiryBusy(true)" in page


def test_static_page_tabs_precede_both_shared_mode_layouts() -> None:
    page = (Path(__file__).resolve().parents[1] / "docs" / "index.html").read_text(encoding="utf-8")
    assert page.index('id="modes"') < page.index('id="card"')
    for panel_id in ("panel-inquiry", "panel-practice"):
        panel = page.partition(f'id="{panel_id}"')[2].partition("</section>")[0]
        assert panel.index("analysis-slot") < panel.index("demo-columns")
        assert panel.index("input-column") < panel.index("detail-column")


def test_the_static_page_defines_every_theme_variable_in_both_schemes() -> None:
    """`docs/index.html` 不是 Gradio，那七個變數由它自己給值。

    標註的類型標籤走 `--body-text-color-subdued`，少一邊定義的後果是它在某一個
    明暗模式下變成看不見的字 —— 而那正是開發者的螢幕上永遠看不到的那種錯。
    """
    page = (Path(__file__).resolve().parents[1] / "docs" / "index.html").read_text(encoding="utf-8")
    light, _, dark = page.partition("@media (prefers-color-scheme: dark)")
    assert dark
    for variable in demo_ui.THEME_VARIABLES:
        assert f"{variable}:" in light
        assert f"{variable}:" in dark


def test_the_type_label_is_never_hidden_behind_hover() -> None:
    """觸控裝置上沒有 hover，而手機是這個服務的主要載體。

    hover 只准改顏色與底線粗細；一旦它改到 `display` / `visibility` / `opacity`，
    類型就變成 hover 限定，等於在主要載體上把它刪掉。
    """
    hover_rules = re.findall(r"\.pii-mark:hover[^{]*\{([^}]*)\}", demo_ui.CSS)
    assert hover_rules
    for body in hover_rules:
        for property_name in ("display", "visibility", "opacity", "content"):
            assert property_name not in body
    assert ".pii-kind { font-size:" in demo_ui.CSS


def test_scam_type_values_never_become_colour_grades() -> None:
    """判定不以顏色分級表達。顏色只用來區分狀態。"""
    assert not any(scam_type.value in demo_ui.CSS for scam_type in ScamType)
    for word in ("紅燈", "黃燈", "綠燈", "高風險"):
        assert word not in demo_ui.CSS
