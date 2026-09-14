"""信心值：四級基準、三個上限、取 min，以及八個門檻之間的六條關係。"""

import inspect
import random
from pathlib import Path

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.confidence import compute_confidence
from scam_guard.normalize import Document, build_document
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect
from scam_guard.scoring import compute_score
from scam_guard.types import CheckResult, Message, Request, ScamType
from scam_guard.weights import load_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE = load_weights()

EVASION_CHECKS = (
    "evasion_invisible",
    "evasion_width_mix",
    "evasion_split_word",
    "evasion_homophone",
    "evasion_zhuyin",
)


def a_document() -> Document:
    return build_document([Message(text="請將款項匯入監管帳戶")])


def truncated_document() -> Document:
    return build_document(
        [Message(text=f"第 {index} 則") for index in range(101)],
    )


def hit(name: str, *, hard: bool = False, types: tuple[ScamType, ...] = ()) -> CheckResult:
    return CheckResult(name=name, hit=True, detail="命中", scam_types=list(types), hard=hard)


def miss(name: str) -> CheckResult:
    return CheckResult(name=name, hit=False, detail=NOT_HIT)


def skipped(name: str) -> CheckResult:
    return CheckResult(name=name, hit=False, detail=SKIPPED)


AUTHORITY = (ScamType.FAKE_AUTHORITY,)


# --- 四級基準值 ---------------------------------------------------------


def test_no_hit_is_the_fourth_level_and_abstains() -> None:
    confidence = compute_confidence([miss("solicit_otp")], a_document(), False, TABLE)

    assert confidence == TABLE.threshold("base_no_hit")
    assert confidence < TABLE.threshold("confidence_floor")


def test_hard_evidence_is_the_first_level() -> None:
    results = [hit("safe_account", hard=True, types=AUTHORITY)]

    assert compute_confidence(results, a_document(), False, TABLE) == 0.90


def test_two_groups_is_the_second_level() -> None:
    results = [
        hit("guaranteed_return", types=(ScamType.FAKE_INVESTMENT,)),
        hit("parcel_notice", types=(ScamType.FAKE_PARCEL,)),
    ]

    assert compute_confidence(results, a_document(), False, TABLE) == 0.55


def test_four_rules_in_one_group_is_still_one_group() -> None:
    """同一段腳本命中四條規則仍是一件事，不因命中筆數變成多個依據。"""
    results = [
        hit(name, types=AUTHORITY)
        for name in ("safe_account", "atm_operation", "secrecy_demand", "remote_control_tool")
    ]

    assert compute_confidence(results, a_document(), False, TABLE) == 0.35


def test_five_evasion_checks_are_one_group_in_both_layers() -> None:
    """跨模組一致性：信心層數群組與計分層取 max 必須用同一份分群。"""
    results = [hit(name) for name in EVASION_CHECKS]

    score = compute_score(results, TABLE)

    assert len(score.group_contributions) == 1
    assert score.value == 0.6
    assert compute_confidence(results, a_document(), False, TABLE) == min(
        TABLE.threshold("base_single_group"), TABLE.threshold("cap_unseen_pattern")
    )


# --- 上限一：矛盾 -------------------------------------------------------


def test_contradiction_cap_dominates_hard_evidence() -> None:
    results = [hit("safe_account", hard=True, types=AUTHORITY)]

    confidence = compute_confidence(results, a_document(), True, TABLE)

    assert confidence == 0.30
    assert confidence < TABLE.threshold("confidence_floor")


def test_no_contradiction_keeps_the_base() -> None:
    results = [hit("url_blocklist", hard=True, types=(ScamType.PHISHING_LINK,))]

    assert compute_confidence(results, a_document(), False, TABLE) == 0.90


def test_implementation_never_looks_at_the_quotation_signal() -> None:
    source = (REPO_ROOT / "scam_guard" / "confidence.py").read_text(encoding="utf-8")
    code = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    body = "\n".join(code)

    assert 'roles["quotation"]' not in body
    assert "QUOTATION" not in body


# --- 上限二：未見型態 ---------------------------------------------------


def test_only_evasion_triggers_unseen_pattern() -> None:
    results = [hit("evasion_invisible"), hit("evasion_width_mix")]

    assert compute_confidence(results, a_document(), False, TABLE) <= 0.35


def test_only_shortener_triggers_unseen_pattern() -> None:
    results = [hit("url_shortener")]

    assert compute_confidence(results, a_document(), False, TABLE) <= 0.35


def test_only_relationship_building_triggers_unseen_pattern() -> None:
    results = [hit("relationship_building")]

    assert compute_confidence(results, a_document(), False, TABLE) <= 0.35


def test_any_typed_hit_clears_unseen_pattern() -> None:
    results = [
        hit("evasion_invisible"),
        hit("safe_account", hard=True, types=AUTHORITY),
    ]

    assert compute_confidence(results, a_document(), False, TABLE) == 0.90


def test_no_hit_does_not_trigger_unseen_pattern() -> None:
    """完全無命中由基準值第四級處理，不是未見型態。"""
    confidence = compute_confidence([miss("evasion_invisible")], a_document(), False, TABLE)

    assert confidence == TABLE.threshold("base_no_hit")


# --- 上限三：截斷 -------------------------------------------------------


def test_truncated_with_only_weak_signals() -> None:
    results = [
        hit("guaranteed_return", types=(ScamType.FAKE_INVESTMENT,)),
        hit("parcel_notice", types=(ScamType.FAKE_PARCEL,)),
    ]

    confidence = compute_confidence(results, truncated_document(), False, TABLE)

    assert confidence == 0.50
    assert confidence > TABLE.threshold("confidence_floor")


def test_truncated_with_hard_evidence_is_not_capped() -> None:
    """硬證據是單句可指認的事實，不依賴被丟棄的前文。"""
    results = [hit("safe_account", hard=True, types=AUTHORITY)]

    assert compute_confidence(results, truncated_document(), False, TABLE) == 0.90


def test_not_truncated_is_not_capped() -> None:
    results = [
        hit("guaranteed_return", types=(ScamType.FAKE_INVESTMENT,)),
        hit("parcel_notice", types=(ScamType.FAKE_PARCEL,)),
    ]
    doc = a_document()

    assert doc.truncated is False
    assert compute_confidence(results, doc, False, TABLE) == 0.55


def test_truncation_alone_does_not_cause_abstention() -> None:
    results = [
        hit("guaranteed_return", types=(ScamType.FAKE_INVESTMENT,)),
        hit("parcel_notice", types=(ScamType.FAKE_PARCEL,)),
    ]

    confidence = compute_confidence(results, truncated_document(), False, TABLE)

    assert confidence > TABLE.threshold("confidence_floor")


# --- 取 min 與值域 ------------------------------------------------------


def test_multiple_caps_take_the_smallest() -> None:
    results = [hit("evasion_invisible")]

    confidence = compute_confidence(results, truncated_document(), True, TABLE)

    assert confidence == 0.30


def test_confidence_is_always_within_zero_and_one() -> None:
    rng = random.Random(20260914)
    names = list(TABLE.signals)
    for _ in range(200):
        chosen = rng.sample(names, rng.randint(0, 6))
        results = [
            hit(
                name,
                hard=TABLE.signals[name].hard_capable and rng.random() < 0.5,
                types=(ScamType.FAKE_AUTHORITY,) if rng.random() < 0.5 else (),
            )
            for name in chosen
        ]
        doc = truncated_document() if rng.random() < 0.5 else a_document()

        confidence = compute_confidence(results, doc, rng.random() < 0.5, TABLE)

        assert 0.0 <= confidence <= 1.0


# --- 未執行的檢查不是證據 -----------------------------------------------


def test_unmounted_and_mounted_but_unhit_give_the_same_confidence() -> None:
    results = [hit("safe_account", hard=True, types=AUTHORITY)]

    assert compute_confidence(results, a_document(), False, TABLE) == compute_confidence(
        [*results, miss("domain_age")], a_document(), False, TABLE
    )


def test_skipped_records_do_not_lower_confidence() -> None:
    results = [hit("safe_account", hard=True, types=AUTHORITY)]

    assert compute_confidence(results, a_document(), False, TABLE) == compute_confidence(
        [*results, skipped("domain_age")], a_document(), False, TABLE
    )


def test_url_layer_silence_under_a_shortener_does_not_lower_confidence() -> None:
    """`url_shortener` 命中時 URL 層其餘四個檢查回空陣列 —— 不需要任何特例。"""
    quiet = ("url_blocklist", "url_tld_risk", "url_host_shape", "url_brand")
    silent = [miss(name) for name in quiet]
    results = [hit("url_shortener"), *silent]

    assert compute_confidence(results, a_document(), False, TABLE) == compute_confidence(
        [hit("url_shortener")], a_document(), False, TABLE
    )


def test_implementation_does_not_parse_detail_strings() -> None:
    source = (REPO_ROOT / "scam_guard" / "confidence.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))

    assert ".detail" not in code


# --- 簽章與界線 ---------------------------------------------------------


def test_signature_does_not_accept_a_score() -> None:
    """拿不到分數就不可能讀分數大小。"""
    parameters = inspect.signature(compute_confidence).parameters

    assert list(parameters) == ["results", "doc", "contradicted", "table"]
    assert parameters["contradicted"].annotation is bool
    assert not any("Score" in str(parameter.annotation) for parameter in parameters.values())


def test_module_does_not_import_pipeline_or_rules() -> None:
    source = (REPO_ROOT / "scam_guard" / "confidence.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert not any("scam_guard.pipeline" in line for line in imports)
    assert not any("scam_guard.rules" in line for line in imports)


# --- 八個門檻之間的六條關係 ---------------------------------------------


def test_no_hit_base_is_below_the_floor() -> None:
    assert TABLE.threshold("base_no_hit") < TABLE.threshold("confidence_floor")


def test_single_group_base_is_below_the_floor() -> None:
    assert TABLE.threshold("base_single_group") < TABLE.threshold("confidence_floor")


def test_hard_base_is_above_the_floor() -> None:
    assert TABLE.threshold("base_hard") > TABLE.threshold("confidence_floor")


def test_contradiction_cap_is_below_the_floor() -> None:
    assert TABLE.threshold("cap_contradiction") < TABLE.threshold("confidence_floor")


def test_unseen_pattern_cap_is_below_the_floor() -> None:
    assert TABLE.threshold("cap_unseen_pattern") < TABLE.threshold("confidence_floor")


def test_truncated_cap_is_above_the_floor() -> None:
    assert TABLE.threshold("cap_truncated") > TABLE.threshold("confidence_floor")


def test_floor_can_be_swept_without_touching_the_original() -> None:
    derived = TABLE.with_overrides(thresholds={"confidence_floor": 0.95})

    assert derived.threshold("confidence_floor") == 0.95
    assert TABLE.threshold("confidence_floor") == 0.40


# --- 端到端 -------------------------------------------------------------


class StaticCheck:
    """回傳固定結果的假檢查。"""

    stage = Stage.LOCAL

    def __init__(self, name: str, results: list[CheckResult]) -> None:
        self.name = name
        self.results = results

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return list(self.results)


def test_no_signal_request_abstains_with_near_zero_confidence() -> None:
    registry = CheckRegistry()
    registry.register(StaticCheck("solicit_otp", []))

    verdict = detect(Request.from_text("明天見"), registry, TABLE)

    assert verdict.scam_probability is None
    assert verdict.confidence == TABLE.threshold("base_no_hit")


def test_awareness_post_abstains_at_the_contradiction_cap() -> None:
    registry = CheckRegistry()
    registry.register(
        StaticCheck("safe_account", [hit("safe_account", hard=True, types=AUTHORITY)])
    )
    registry.register(StaticCheck("quotation", [hit("quotation")]))

    verdict = detect(Request.from_text("有人傳這個給我，說要匯到監管帳戶"), registry, TABLE)

    assert verdict.scam_probability is None
    assert verdict.confidence == TABLE.threshold("cap_contradiction")


def test_blocklist_exact_hit_with_quotation_still_gives_a_number() -> None:
    registry = CheckRegistry()
    registry.register(
        StaticCheck(
            "url_blocklist",
            [hit("url_blocklist", hard=True, types=(ScamType.PHISHING_LINK,))],
        )
    )
    registry.register(StaticCheck("quotation", [hit("quotation")]))

    verdict = detect(Request.from_text("有人傳這個給我 https://evil.com"), registry, TABLE)

    assert verdict.scam_probability == pytest.approx(0.9241418199787566)
    assert verdict.confidence == TABLE.threshold("base_hard")


def test_confidence_is_reported_even_when_abstaining() -> None:
    registry = CheckRegistry()
    registry.register(StaticCheck("evasion_invisible", [hit("evasion_invisible")]))

    verdict = detect(Request.from_text("匯 款"), registry, TABLE)

    assert verdict.scam_probability is None
    assert verdict.confidence > 0.0
