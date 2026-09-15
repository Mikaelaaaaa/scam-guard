"""共用標記層（`demo_ui.py`）的測試。

**本檔不 `importorskip("gradio")`、不需要 `/psl`、不 import `docs/pages_app.py`。**
那三件事各自是一個環境條件，而一個因環境缺件被跳過的測試檔等於沒有測試 ——
`tests/` 底下今天沒有任何 pages 測試，正是因為那一側在 import 階段就要 PSL 快照。
共用模組沒有這個問題，所以它的性質在這裡被無條件驗證。
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import demo_ui
from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import DEFAULT_LIMITS, Document, Limits, build_document
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


def test_signals_card_shows_the_gate_instead_of_the_probability() -> None:
    """第三態的卡上是「分數與門檻的距離」，不是一個未達門檻的百分比。

    `77%` 與判定成立時的 `92%` 長得一模一樣，使用者分不出哪一個是系統認可的
    判定。百分比移進偵測細節，在那裡它旁邊就是分數、門檻與逐項訊號。
    """
    verdict, document = verdict_for(SIGNALS_TEXT)
    card = demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED)
    head, _, details = card.partition('<details class="details">')
    probability = f"{verdict.scam_probability:.0%}"

    assert probability not in head
    assert not re.search(r"\d+%", re.sub(r'style="width:[^"]*"|style="left:[^"]*"', "", head))
    assert f"{TABLE.threshold('decision_score'):.2f}" in head
    assert probability in details
    assert "/ 1.50" not in card


def test_score_scale_keeps_the_gate_away_from_the_right_edge() -> None:
    """刻度尺的右界超過門檻 —— 它讀起來不能像「進度條快滿了」。"""
    gate = TABLE.threshold("decision_score")
    assert demo_ui.scale_bound(1.2, gate) > gate
    assert demo_ui.scale_bound(5.0, gate) > 5.0


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
    hits = sum(1 for result in verdict.checks if result.hit)
    assert f"共 {total} 項，{hits} 項命中" in details

    quiet, quiet_document = verdict_for(QUIET_TEXT)
    quiet_details = demo_ui.render_details(quiet, quiet_document, UNREGISTERED)
    assert f"共 {len(quiet.checks) + len(UNREGISTERED)} 項" in quiet_details
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
    return [
        (match.start(), match.end(), "身分證") for match in re.finditer(r"[A-Z][12]\d{8}", sentence)
    ]


def test_user_input_is_escaped() -> None:
    document = build_document([Message(text="<script>alert(1)</script>")], LIMITS)
    conversation = demo_ui.render_conversation(
        [Message(text="<script>alert(1)</script>")], (), document, "你", "對方"
    )
    assert "<script>" not in conversation
    assert "&lt;script&gt;" in conversation


def test_escaped_url_round_trips() -> None:
    assert demo_ui.escaped("https://x.cc/a?id=1&ref=2") == "https://x.cc/a?id=1&amp;ref=2"


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
    marked = re.findall(r'<mark class="pii-mark"[^>]*>(.*?)</mark>', rendered)
    assert marked == ["A123456789"]
    assert "&amp;ref=2" in rendered


def test_pii_not_mounted_is_not_the_same_as_no_pii() -> None:
    """承接既有的同名測試：兩個狀態仍然可分辨。"""
    document = build_document([Message(text="測試訊息。")], LIMITS)
    not_mounted = demo_ui.render_pii_block(document, None)
    mounted = demo_ui.render_pii_block(document, recognizer_none)
    assert demo_ui.PII_NOT_MOUNTED in not_mounted
    assert demo_ui.PII_NO_HIT not in not_mounted
    assert demo_ui.PII_NO_HIT in mounted
    assert demo_ui.PII_NOT_MOUNTED not in mounted


def test_pii_block_discloses_its_scope() -> None:
    document = build_document([Message(text="測試訊息。")], LIMITS)
    block = demo_ui.render_pii_block(document, None)
    for label in ("身分證", "手機", "市話", "信用卡"):
        assert label in block
    assert "姓名與地址不在辨識範圍內" in block


def test_pii_tags_are_sentence_level_and_do_not_rewrite_text() -> None:
    document = build_document([Message(text="我的身分證是 A123456789 請確認。")], LIMITS)
    conversation = demo_ui.render_conversation(
        [Message(text="我的身分證是 A123456789 請確認。")],
        (),
        document,
        "你",
        "對方",
        recognizer_tw_id,
    )
    assert "身分證 ×1" in conversation
    assert "A123456789" in conversation
    assert "<TW_ID>" not in conversation


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


def test_scam_type_values_never_become_colour_grades() -> None:
    """判定不以顏色分級表達。顏色只用來區分狀態。"""
    assert not any(scam_type.value in demo_ui.CSS for scam_type in ScamType)
    for word in ("紅燈", "黃燈", "綠燈", "高風險"):
        assert word not in demo_ui.CSS
