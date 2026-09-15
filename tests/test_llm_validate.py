"""模型輸出的解析、七條語意驗證、fail-closed、可觀測性與單調升級。

需要 `LlmCheck` 或 `pipeline` 的那幾條（建構時缺 counter、injection 回歸、
短路時留下 `SKIPPED` 記錄）在 `tests/test_llm_check.py` ——
`LlmCheck` 屬 `add-llm-client`。
"""

import ast
import json
import random
from pathlib import Path

import pytest

from scam_guard.llm import validate as validate_module
from scam_guard.llm.prompt import PromptWindow
from scam_guard.llm.schema import LABELS
from scam_guard.llm.validate import (
    SCAM_SIGNAL,
    SUSPICIOUS_SIGNAL,
    LlmOutcome,
    LlmOutcomeCounter,
    parse_and_validate,
    to_check_results,
)
from scam_guard.render import SPECULATIVE_TERMS, VERDICT_CLAIMS
from scam_guard.scoring import compute_score
from scam_guard.types import CheckResult, Coord, ScamType
from scam_guard.weights import load_weights

TABLE = load_weights()

VALIDATE_SOURCE = Path(validate_module.__file__).read_text(encoding="utf-8")

WINDOW = PromptWindow(coords=[(0, 0), (0, 1), (1, 0), (1, 1)], dropped_before=0)

# 兩張禁用詞表，複製自 `scam_guard/render.py`；理由與 `tests/test_llm_schema.py` 同。
SPECULATIVE_TERMS_COPY = frozenset(
    {"可疑", "危險", "不明", "很可能", "應該是", "一定是", "肯定", "小心", "注意"}
)
VERDICT_CLAIMS_COPY = frozenset({"是詐騙", "為詐騙", "詐騙訊息", "確定是"})


def an_output(
    notes: str = "第 1 句自稱郵局",
    ids: list[Coord] | None = None,
    category: str | None = "假投資",
    label: str = LABELS[2],
) -> str:
    # `ids or [(0, 0)]` 會把「刻意給空陣列」變成「用預設值」—— 正是本專案
    # 禁止的那種安靜 fallback，在測試輔助函式裡一樣不行。
    if ids is None:
        ids = [(0, 0)]
    return json.dumps(
        {
            "analysis_notes": notes,
            "evidence_sentence_ids": [list(coord) for coord in ids],
            "category_165": category,
            "label": label,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def imported_modules(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


# --- 解析（STRUCTURE） --------------------------------------------------


def test_a_complete_output_parses() -> None:
    outcome, output = parse_and_validate(an_output(), WINDOW)

    assert outcome is LlmOutcome.OK
    assert output is not None
    assert output.label == LABELS[2]
    assert output.category_165 is ScamType.FAKE_INVESTMENT


def test_a_truncated_output_is_a_structure_failure() -> None:
    outcome, output = parse_and_validate(an_output()[:-1], WINDOW)

    assert outcome is LlmOutcome.STRUCTURE
    assert output is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"analysis_notes":"x","evidence_sentence_ids":[[0,0]],"category_165":null}',
        '{"analysis_notes":"x","evidence_sentence_ids":[[0,0]],"category_165":null,'
        '"label":"完整詐騙話術","confidence":0.9}',
        '["完整詐騙話術"]',
        '{"analysis_notes":3,"evidence_sentence_ids":[[0,0]],"category_165":null,'
        '"label":"完整詐騙話術"}',
    ],
)
def test_a_wrong_shape_is_a_structure_failure(raw: str) -> None:
    assert parse_and_validate(raw, WINDOW)[0] is LlmOutcome.STRUCTURE


# --- 座標（SEMANTIC） ---------------------------------------------------


def test_a_coordinate_inside_the_window_passes() -> None:
    window = PromptWindow(coords=[(37, 0), (38, 1), (40, 2)], dropped_before=12)

    outcome, output = parse_and_validate(an_output(ids=[(38, 1)]), window)

    assert outcome is LlmOutcome.OK
    assert output is not None
    assert output.evidence_sentence_ids == ((38, 1),)


def test_a_coordinate_in_the_document_but_not_in_the_window_is_semantic() -> None:
    """`Document.index_of()` 對 `(0,0)` 不拋例外，而 `(0,0)` 正是 1B 模型最愛的值。"""
    window = PromptWindow(coords=[(37, 0), (37, 1)], dropped_before=95)

    assert parse_and_validate(an_output(ids=[(0, 0)]), window)[0] is LlmOutcome.SEMANTIC


def test_a_coordinate_that_exists_nowhere_is_semantic() -> None:
    assert parse_and_validate(an_output(ids=[(99, 7)]), WINDOW)[0] is LlmOutcome.SEMANTIC


def test_duplicate_coordinates_are_semantic_and_are_not_deduplicated() -> None:
    outcome, output = parse_and_validate(an_output(ids=[(0, 0), (0, 0)]), WINDOW)

    assert outcome is LlmOutcome.SEMANTIC
    assert output is None


def test_coordinates_are_sorted_after_validation() -> None:
    outcome, output = parse_and_validate(an_output(ids=[(1, 1), (0, 1)]), WINDOW)

    assert outcome is LlmOutcome.OK
    assert output is not None
    assert output.evidence_sentence_ids == ((0, 1), (1, 1))


@pytest.mark.parametrize(
    "ids", ["[[0]]", "[[0,0,0]]", '[[0,"0"]]', "[[true,false]]", '"[[0,0]]"', "[0,0]"]
)
def test_a_malformed_coordinate_shape_is_semantic(ids: str) -> None:
    raw = (
        '{"analysis_notes":"x","evidence_sentence_ids":' + ids + ","
        '"category_165":null,"label":"完整詐騙話術"}'
    )

    assert parse_and_validate(raw, WINDOW)[0] is LlmOutcome.SEMANTIC


# --- 跨欄位（SEMANTIC） -------------------------------------------------


def test_the_lowest_label_with_a_type_is_semantic() -> None:
    raw = an_output(ids=[], category="假投資", label=LABELS[0])

    assert parse_and_validate(raw, WINDOW)[0] is LlmOutcome.SEMANTIC


def test_a_non_lowest_label_without_a_type_passes() -> None:
    outcome, output = parse_and_validate(an_output(category=None, label=LABELS[1]), WINDOW)

    assert outcome is LlmOutcome.OK
    assert output is not None
    assert output.category_165 is None
    assert to_check_results(outcome, output)[0].scam_types == []


def test_a_non_lowest_label_without_evidence_is_semantic() -> None:
    assert parse_and_validate(an_output(ids=[]), WINDOW)[0] is LlmOutcome.SEMANTIC


def test_the_lowest_label_without_a_type_and_without_evidence_passes() -> None:
    raw = an_output(ids=[], category=None, label=LABELS[0])

    assert parse_and_validate(raw, WINDOW)[0] is LlmOutcome.OK


@pytest.mark.parametrize("category", ["網路購物", "其他", "", "假投資 "])
def test_a_type_outside_the_vocabulary_is_semantic(category: str) -> None:
    assert parse_and_validate(an_output(category=category), WINDOW)[0] is LlmOutcome.SEMANTIC


def test_a_label_outside_the_vocabulary_is_semantic() -> None:
    assert parse_and_validate(an_output(label="可疑"), WINDOW)[0] is LlmOutcome.SEMANTIC


# --- fail-closed --------------------------------------------------------


@pytest.mark.parametrize("outcome", [LlmOutcome.STRUCTURE, LlmOutcome.SEMANTIC, LlmOutcome.TIMEOUT])
def test_every_failure_yields_no_results(outcome: LlmOutcome) -> None:
    assert to_check_results(outcome, None) == []


def test_the_lowest_label_yields_no_results_even_when_the_outcome_is_ok() -> None:
    """單調升級的條件 A。"""
    outcome, output = parse_and_validate(an_output(ids=[], category=None, label=LABELS[0]), WINDOW)

    assert outcome is LlmOutcome.OK
    assert to_check_results(outcome, output) == []


# --- CheckResult --------------------------------------------------------


def test_the_highest_label_becomes_one_soft_result() -> None:
    outcome, output = parse_and_validate(an_output(), WINDOW)

    results = to_check_results(outcome, output)

    assert len(results) == 1
    assert results[0].name == SCAM_SIGNAL
    assert results[0].hit is True
    assert results[0].hard is False
    assert results[0].scam_types == [ScamType.FAKE_INVESTMENT]


def test_the_middle_label_becomes_the_other_signal() -> None:
    outcome, output = parse_and_validate(an_output(label=LABELS[1]), WINDOW)

    assert to_check_results(outcome, output)[0].name == SUSPICIOUS_SIGNAL


def test_the_detail_matches_character_for_character() -> None:
    raw = an_output(ids=[(0, 0), (0, 1), (1, 0)], category="假檢警/假冒公務機關")

    outcome, output = parse_and_validate(raw, WINDOW)

    assert to_check_results(outcome, output)[0].detail == (
        "語意判讀：完整詐騙話術（類型：假檢警/假冒公務機關，依據 3 句）"
    )


def test_the_detail_without_a_type_still_carries_the_sentence_count() -> None:
    outcome, output = parse_and_validate(an_output(category=None, label=LABELS[1]), WINDOW)

    assert to_check_results(outcome, output)[0].detail == (
        "語意判讀：部分詐騙話術（未判定類型，依據 1 句）"
    )


def test_every_possible_detail_passes_both_banned_word_tables() -> None:
    assert SPECULATIVE_TERMS_COPY == SPECULATIVE_TERMS
    assert VERDICT_CLAIMS_COPY == VERDICT_CLAIMS
    details = []
    for label in LABELS[1:]:
        outcome, output = parse_and_validate(an_output(category=None, label=label), WINDOW)
        details.append(to_check_results(outcome, output)[0].detail)
        for scam_type in ScamType:
            outcome, output = parse_and_validate(
                an_output(category=scam_type.value, label=label), WINDOW
            )
            details.append(to_check_results(outcome, output)[0].detail)
    for detail in details:
        assert not [term for term in SPECULATIVE_TERMS_COPY if term in detail]
        assert not [claim for claim in VERDICT_CLAIMS_COPY if claim in detail]
        assert any(character.isdigit() for character in detail)


def test_the_detail_never_carries_the_analysis_notes() -> None:
    marker = "MODEL-FREE-TEXT-MARKER"

    outcome, output = parse_and_validate(an_output(notes=marker), WINDOW)

    assert output is not None
    assert output.analysis_notes == marker
    assert marker not in to_check_results(outcome, output)[0].detail


# --- 計數器 -------------------------------------------------------------


def test_a_fresh_counter_has_no_failure_rate() -> None:
    with pytest.raises(ValueError):
        LlmOutcomeCounter().failure_rate()


def test_three_readings_are_counted_separately() -> None:
    counter = LlmOutcomeCounter()

    counter.record(LlmOutcome.OK)
    counter.record(LlmOutcome.SEMANTIC)
    counter.record(LlmOutcome.TIMEOUT)

    assert counter.counts() == {
        LlmOutcome.OK: 1,
        LlmOutcome.STRUCTURE: 0,
        LlmOutcome.SEMANTIC: 1,
        LlmOutcome.TIMEOUT: 1,
    }
    assert counter.total() == 3
    assert counter.failure_rate() == pytest.approx(2 / 3)


def test_the_counter_does_no_io() -> None:
    source = VALIDATE_SOURCE
    for forbidden in ("open(", "urllib", "subprocess", "socket", "Path("):
        assert forbidden not in source


def test_validate_imports_no_pipeline_or_runtime() -> None:
    assert imported_modules(VALIDATE_SOURCE).isdisjoint(
        {"llama_cpp", "llm_runtime", "scam_guard.pipeline"}
    )


# --- 單調升級 -----------------------------------------------------------


def test_both_llm_signals_are_soft_and_non_negative() -> None:
    """單調升級的條件 B。"""
    for name in (SCAM_SIGNAL, SUSPICIOUS_SIGNAL):
        signal = TABLE.signals[name]
        assert signal.hard_capable is False
        assert signal.weight_soft.value >= 0.0
        assert signal.weight_hard is None
        assert "add-llm-validate" in signal.weight_soft.inherited_from
    assert (
        TABLE.signals[SCAM_SIGNAL].weight_soft.value
        == TABLE.signals[SUSPICIOUS_SIGNAL].weight_soft.value
    )


def test_the_llm_signals_hold_no_role() -> None:
    """單調升級的條件 C —— 擔任引述角色就能觸發矛盾而讓系統拒答。"""
    assert set(TABLE.roles.values()).isdisjoint({SCAM_SIGNAL, SUSPICIOUS_SIGNAL})


def test_adding_an_llm_result_never_lowers_the_score() -> None:
    generator = random.Random(20260915)
    names = [name for name in TABLE.signals if name not in (SCAM_SIGNAL, SUSPICIOUS_SIGNAL)]
    labels = list(LABELS)
    categories = [None] + [scam_type.value for scam_type in ScamType]
    for _ in range(200):
        results = [
            CheckResult(
                name=name,
                hit=True,
                detail="0.87/0.85",
                hard=TABLE.signals[name].hard_capable and generator.random() < 0.5,
            )
            for name in generator.sample(names, generator.randint(0, 6))
        ]
        label = generator.choice(labels)
        category = None if label == LABELS[0] else generator.choice(categories)
        ids = [] if label == LABELS[0] else [list(generator.choice(list(WINDOW.coords)))]
        raw = json.dumps(
            {
                "analysis_notes": "x",
                "evidence_sentence_ids": ids,
                "category_165": category,
                "label": label,
            },
            ensure_ascii=False,
        )
        outcome, output = parse_and_validate(raw, WINDOW)
        assert outcome is LlmOutcome.OK

        before = compute_score(results, TABLE)
        after = compute_score(results + to_check_results(outcome, output), TABLE)

        assert after.value >= before.value


def test_an_unmounted_llm_and_a_silent_llm_score_the_same() -> None:
    results = [CheckResult(name="solicit_otp", hit=True, detail="第 2 句索取驗證碼", hard=True)]
    outcome, output = parse_and_validate(an_output(ids=[], category=None, label=LABELS[0]), WINDOW)

    unmounted = compute_score(results, TABLE)
    silent = compute_score(results + to_check_results(outcome, output), TABLE)

    assert silent.value == unmounted.value
    assert silent.probability == unmounted.probability
    assert silent.group_contributions == unmounted.group_contributions
