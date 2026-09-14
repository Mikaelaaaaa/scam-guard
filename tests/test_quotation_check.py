"""引述偵測 —— 名稱契約、四類標記、輸出形狀與短路否決。"""

import ast
import inspect

from scam_guard import pipeline
from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document, build_document
from scam_guard.pipeline import SKIPPED, detect
from scam_guard.rules import quotation
from scam_guard.rules.quotation import (
    ATTRIBUTION_CATEGORY,
    AWARENESS_CATEGORY,
    FORWARD_CATEGORY,
    MIN_QUOTE_LEN,
    NAME,
    QUOTE_CATEGORY,
    QUOTE_WEIGHT,
    QuotationCheck,
    long_quotes,
    quoted_spans,
)
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import CheckResult, Message, Request

CONVERSATION_LOG = (
    "上午 10:23 小明 你好\n上午 10:24 小明 幫我看一下這個\n上午 10:25 小明 請至 ATM 解除分期"
)

AWARENESS_POST = "最近很多假檢警詐騙，會叫你把錢匯到監管帳戶，千萬不要相信"


class FakeLlm:
    name = "llm"
    stage = Stage.EXPENSIVE

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        self.calls += 1
        return []


def run(*texts: str) -> list[CheckResult]:
    req = Request(messages=[Message(text=text) for text in texts])
    return QuotationCheck()(req, build_document(req.messages))


def categories(*texts: str) -> str:
    results = run(*texts)
    return results[0].detail if results else ""


# --- 名稱與契約 -------------------------------------------------------------


def test_name_matches_the_pipeline_constant() -> None:
    """名稱漂移時這一行立刻紅 —— 契約由測試綁定，不由 import 綁定。"""
    assert NAME == pipeline.QUOTATION_CHECK
    assert QuotationCheck().name == pipeline.QUOTATION_CHECK


def test_module_does_not_import_pipeline() -> None:
    """反向 import（`rules/` → `pipeline`）是一條由下往上的邊，不建立它。"""
    tree = ast.parse(inspect.getsource(quotation))
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]

    assert imported != []
    assert all("pipeline" not in module for module in imported)


def test_stage_is_local() -> None:
    assert QuotationCheck().stage is Stage.LOCAL


# --- 四類標記 ---------------------------------------------------------------


def test_attribution_marker_hits() -> None:
    assert ATTRIBUTION_CATEGORY in categories("有人傳這個給我媽，請至 ATM 解除分期")


def test_awareness_frame_hits() -> None:
    assert AWARENESS_CATEGORY in categories(AWARENESS_POST)


def test_bank_reminder_is_an_accepted_false_positive() -> None:
    """真實銀行簡訊的標準句。方向是把正常訊息判得更正常，代價是零。"""
    assert AWARENESS_CATEGORY in categories("本行絕不會以電話要求您操作 ATM")


def test_plain_scam_instruction_does_not_hit() -> None:
    assert run("請至 ATM 依語音指示操作解除分期") == []


def test_short_quote_is_emphasis_not_quotation() -> None:
    assert long_quotes("『限時』特價中") == []
    assert run("『限時』特價中") == []


def test_long_quote_counts() -> None:
    sentence = "他說『請你到提款機把設定解除』"

    assert long_quotes(sentence) == ["請你到提款機把設定解除"]
    assert QUOTE_CATEGORY in categories(sentence)


def test_unpaired_quote_is_skipped() -> None:
    assert quoted_spans("他說「請你到提款機把設定解除") == []
    assert QUOTE_CATEGORY not in categories("「請你到提款機把設定解除")


def test_nested_quotes_take_the_outer_pair() -> None:
    assert long_quotes("「他說『到提款機解除設定』就對了」") == ["他說『到提款機解除設定』就對了"]


def test_quote_threshold_is_a_module_constant() -> None:
    assert MIN_QUOTE_LEN == 8


def test_three_timestamp_lines_hit_forward_format() -> None:
    assert FORWARD_CATEGORY in categories(CONVERSATION_LOG)


def test_single_date_does_not_hit() -> None:
    assert run("請於 09/20 前完成付款") == []


# --- 輸出形狀 ---------------------------------------------------------------


def test_multiple_categories_still_produce_one_result() -> None:
    results = run("有人傳這個給我，他說「請你到提款機把設定解除」，千萬不要相信")

    assert len(results) == 1
    assert QUOTE_CATEGORY in results[0].detail
    assert ATTRIBUTION_CATEGORY in results[0].detail
    assert AWARENESS_CATEGORY in results[0].detail
    assert "命中 3 類引述標記" in results[0].detail


def test_output_shape() -> None:
    results = run(AWARENESS_POST)

    assert results[0].hit is True
    assert results[0].hard is False
    assert results[0].scam_types == []
    assert results[0].weight == QUOTE_WEIGHT == -1.5


def test_evidence_covers_every_matching_sentence() -> None:
    req = Request(messages=[Message(text="有人傳這個給我。幫我看看。千萬不要相信。")])
    doc = build_document(req.messages)

    results = QuotationCheck()(req, doc)

    assert results[0].evidence == [(0, 0), (0, 2)]
    assert all(doc.index_of(coord) >= 0 for coord in results[0].evidence)


def test_no_marker_returns_empty_list() -> None:
    assert run("今天天氣不錯，我們晚點約在捷運站見。") == []


# --- 與 pipeline 的整合 ------------------------------------------------------


def registry_with_rules() -> tuple[CheckRegistry, FakeLlm]:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    registry.register(QuotationCheck())
    llm = FakeLlm()
    registry.register(llm)
    return registry, llm


def test_quotation_vetoes_the_short_circuit_despite_hard_evidence() -> None:
    registry, llm = registry_with_rules()

    checks = detect(Request.from_text(AWARENESS_POST), registry).checks

    hard_hits = [r for r in checks if r.hit and r.hard]
    assert [r.name for r in hard_hits] == ["safe_account"]
    assert llm.calls == 1
    assert all(r.detail != SKIPPED for r in checks)


def test_quotation_alone_runs_expensive_checks() -> None:
    registry, llm = registry_with_rules()

    detect(Request.from_text("有人傳這個給我，幫我看看"), registry)

    assert llm.calls == 1


def test_rules_only_mode_does_not_raise() -> None:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    registry.register(QuotationCheck())

    checks = detect(Request.from_text(AWARENESS_POST), registry).checks

    assert any(r.name == NAME and r.hit for r in checks)


def test_quotation_does_not_change_other_checks() -> None:
    """引述不抑制其他規則的命中 —— 硬證據原封不動，只是不短路。"""
    without = CheckRegistry()
    register_speech_act_rules(without)
    with_quotation = CheckRegistry()
    register_speech_act_rules(with_quotation)
    with_quotation.register(QuotationCheck())

    plain = [r for r in detect(Request.from_text(AWARENESS_POST), without).checks]
    quoted = [
        r
        for r in detect(Request.from_text(AWARENESS_POST), with_quotation).checks
        if r.name != NAME
    ]

    assert plain == quoted


# --- 對抗樣本 ---------------------------------------------------------------


def test_scam_disguised_as_awareness_still_keeps_its_hard_evidence() -> None:
    """已知的對抗樣本：前半是宣導框架，後半是真的索取。

    引述命中的後果有兩個 —— 否決短路（多跑一次 LLM，無害）與負權重（扣分，有害）。
    「引述命中但同時有硬證據時不套用負權重」這條規則的正確落點是
    `add-score-compute`：`Check` 協定只收 `req` 與 `doc`，檢查看不到彼此的結果。
    本檢查的責任到「把判斷所需的資訊放進 `detail`」為止。
    """
    text = "近期常見假冒銀行詐騙，為保障您的權益請把網銀帳號密碼提供給我"
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    registry.register(QuotationCheck())

    checks = detect(Request.from_text(text), registry).checks
    hit = {r.name: r for r in checks if r.hit}

    assert NAME in hit
    assert AWARENESS_CATEGORY in hit[NAME].detail
    assert hit["solicit_bank_credentials"].hard is True
    assert hit["solicit_bank_credentials"].weight == 2.5
    assert hit["solicit_bank_credentials"].evidence != []
