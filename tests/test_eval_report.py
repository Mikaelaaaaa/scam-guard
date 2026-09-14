"""報告層 —— 誤判率與棄權率的綁定、ham 子集不得合併、類型無 ground truth。

大部分測試以合成的 `RunRecord` 執行，不吃資料：這一層問的是「報告的形狀對不對」，
而形狀不該依賴任何一份語料。最後一節的驗收測試吃資料，缺檔時 skip。
"""

import csv
import math
from pathlib import Path

import pytest

from scam_guard.normalize import build_document
from scam_guard.types import Message, Request
from scam_guard.weights import load_weights
from tools.eval import report as report_module
from tools.eval import stats as stats_module
from tools.eval.calibration import MIN_BIN_SIZE, MIN_SUBSET_SIZE, calibrate
from tools.eval.report import (
    MULTI_MESSAGE_MINIMUM,
    Report,
    SubsetReport,
    build_report,
    render_report,
    subset_report,
)
from tools.eval.run import NO_CLAUSE_MARKER, RunId, RunRecord
from tools.eval.selectors import (
    COFACTS_HAM_AD,
    COFACTS_HAM_SUSPECTED,
    COFACTS_SCAM,
    HOLDOUT,
    LABEL_CASE_ONLY,
    MULTI_MESSAGE_HAM,
    LABEL_HAM,
    LABEL_SCAM,
    SELF_SMS_HAM,
    SELF_SMS_SCAM,
)
from tools.eval.signals import LOWER_BOUND_ONLY, NOT_ESTIMABLE, estimate_signal
from tools.eval.stats import Rate, wilson_upper
from tools.eval.sweep import SweepPoint, confidence_cutpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE = load_weights()
RUN_ID = RunId(
    testset_manifest_sha256="a" * 64,
    weights_sha256="b" * 64,
    blocklist_data_through="2025-12-31",
    blocklist_manifest_sha256="c" * 64,
    git_commit="d" * 40,
)


def make_record(
    identifier: str,
    subset: str,
    label: str,
    *,
    score: float = 0.0,
    confidence: float = 0.05,
    decided: bool = False,
    abstained: bool = True,
    scam_type: str | None = None,
    hits: tuple[str, ...] = (),
    split: str = HOLDOUT,
) -> RunRecord:
    return RunRecord(
        id=identifier,
        subset=subset,
        split=split,
        label=label,
        score=score,
        confidence=confidence,
        probability=None if abstained else 0.9,
        abstained=abstained,
        decided=decided,
        scam_type=scam_type,
        contradicted=False,
        quotation_hit=False,
        blocklist_hit="url_blocklist" in hits,
        simplified=False,
        hit_signals=hits,
        hard_signals=(),
        shadowed_signals=(),
        speech_act_hits=1,
        no_clause_hits=0,
        evidence=("依據一行",),
        actions=("建議一行",),
    )


def baseline_records() -> list[RunRecord]:
    """三個子集各湊出足以計算比率的樣本，誤判分布使頭條必然落在 self_sms_ham。"""
    records: list[RunRecord] = []
    records += [make_record(f"ad-{i}", COFACTS_HAM_AD.name, LABEL_HAM) for i in range(400)]
    records += [make_record(f"sus-{i}", COFACTS_HAM_SUSPECTED.name, LABEL_HAM) for i in range(400)]
    records += [make_record(f"self-{i}", SELF_SMS_HAM.name, LABEL_HAM) for i in range(189)]
    records += [make_record(f"scam-{i}", COFACTS_SCAM.name, LABEL_SCAM) for i in range(600)]
    records[800] = make_record(
        "self-0", SELF_SMS_HAM.name, LABEL_HAM, decided=True, abstained=False, confidence=0.9
    )
    return records


def _baseline() -> tuple[list[RunRecord], dict[str, str]]:
    """基準樣本與它們的原文，成對取用 —— 逐則案例兩者都要。"""
    records = baseline_records()
    return records, texts_for(records)


def texts_for(records: list[RunRecord]) -> dict[str, str]:
    """逐則案例需要原文。合成資料一律給同一段占位文字。"""
    return {record.id: "內文" for record in records}


# --- 誤判率與棄權率的綁定 -------------------------------------------------


def test_subset_report_cannot_be_built_without_the_abstention_rate() -> None:
    with pytest.raises(TypeError, match="abstention_rate"):
        SubsetReport(  # type: ignore[call-arg]
            name="x",
            ham_count=10,
            scam_count=0,
            false_positive_rate=Rate(numerator=1, denominator=10),
            recall=None,
        )


def test_subset_report_has_no_item_access_and_no_single_metric_accessor() -> None:
    """取不到單獨的誤判率，就不可能寫出一個只有誤判率的句子。"""
    report = subset_report("x", [make_record("a", COFACTS_HAM_AD.name, LABEL_HAM)])
    assert not hasattr(report, "__getitem__")
    names = [name for name in dir(report) if not name.startswith("_")]
    assert "abstention_rate" in names
    assert not [name for name in names if name.startswith("fpr")]


def test_no_module_level_false_positive_rate_function() -> None:
    """`dir()` 掃描兩個模組的公開名稱，確認沒有一個獨立的誤判率入口。"""
    for module in (report_module, stats_module):
        public = [name for name in dir(module) if not name.startswith("_")]
        assert "fpr" not in public
        assert "false_positive_rate" not in public


def test_markdown_row_always_carries_both_columns() -> None:
    report = subset_report("x", [make_record("a", COFACTS_HAM_AD.name, LABEL_HAM)])
    row = report.to_markdown()
    assert row.count("|") == 7
    assert "to_markdown" in dir(report)
    with pytest.raises(TypeError):
        report.to_markdown(include_abstention=False)  # type: ignore[call-arg]


# --- ham 子集分開，頭條取最差 ---------------------------------------------


def test_report_has_no_pooled_false_positive_rate() -> None:
    fields = set(Report.__dataclass_fields__) | {
        name for name in dir(Report) if not name.startswith("_")
    }
    assert not [name for name in fields if "pool" in name or "combined" in name]


def test_headline_takes_the_worst_upper_bound() -> None:
    report = build_report(*_baseline(), TABLE, RUN_ID)
    name, rate = report.headline_false_positive_rate
    assert name == SELF_SMS_HAM.name
    assert rate.numerator == 1
    assert rate.denominator == 189
    for other in (COFACTS_HAM_AD.name, COFACTS_HAM_SUSPECTED.name):
        other_rate = report.subsets[other].false_positive_rate
        assert other_rate is not None
        assert other_rate.upper < rate.upper


def test_headline_raises_when_no_ham_subset_survived() -> None:
    records = [make_record(f"s-{i}", COFACTS_SCAM.name, LABEL_SCAM) for i in range(10)]
    records += [make_record(f"m-{i}", MULTI_MESSAGE_HAM.name, LABEL_HAM) for i in range(10)]
    report = build_report(
        records,
        texts_for(records),
        TABLE,
        RUN_ID,
        missing_subsets=(SELF_SMS_HAM.name, COFACTS_HAM_AD.name, COFACTS_HAM_SUSPECTED.name),
    )
    with pytest.raises(ValueError, match="沒有任何 ham 子集"):
        report.headline_false_positive_rate


def test_subset_report_rejects_a_rate_that_contradicts_its_sample_count() -> None:
    with pytest.raises(ValueError, match="不一致"):
        SubsetReport(
            name="x",
            ham_count=0,
            scam_count=5,
            abstention_rate=Rate(numerator=1, denominator=5),
            false_positive_rate=Rate(numerator=0, denominator=5),
            recall=Rate(numerator=1, denominator=5),
        )


# --- case_only 不進任何比率 -----------------------------------------------


def test_case_only_records_never_enter_any_numerator_or_denominator() -> None:
    records = baseline_records()
    polluted = records + [
        make_record(
            f"case-{i}",
            SELF_SMS_SCAM.name,
            LABEL_CASE_ONLY,
            decided=True,
            abstained=False,
            confidence=0.9,
        )
        for i in range(30)
    ]
    clean = build_report(records, texts_for(records), TABLE, RUN_ID)
    with_cases = build_report(polluted, texts_for(polluted), TABLE, RUN_ID)
    assert SELF_SMS_SCAM.name not in with_cases.subsets
    for name, subset in clean.subsets.items():
        assert with_cases.subsets[name] == subset
    assert with_cases.by_message_count.single.abstention_rate == (
        clean.by_message_count.single.abstention_rate
    )


# --- 門檻掃描 -------------------------------------------------------------


def test_confidence_floor_has_exactly_five_meaningful_cutpoints() -> None:
    """六個離散信心值之間的五個間隙。細粒度取樣只會產生大量重複的點。"""
    cutpoints = confidence_cutpoints(TABLE)
    assert len(cutpoints) == 5
    assert sorted(cutpoints) == list(cutpoints)


def test_sweep_point_requires_all_three_rates() -> None:
    with pytest.raises(TypeError, match="abstention_rate"):
        SweepPoint(  # type: ignore[call-arg]
            confidence_floor=0.4,
            decision_score=1.5,
            false_positive_rate=Rate(numerator=1, denominator=10),
            recall=Rate(numerator=1, denominator=10),
        )


def test_sweep_output_has_five_columns_and_no_roc() -> None:
    report = build_report(*_baseline(), TABLE, RUN_ID)
    assert report.sweep_points
    assert all(len(point.as_row()) == 5 for point in report.sweep_points)
    public = " ".join(dir(report_module) + dir(stats_module)).lower()
    assert "roc" not in public
    assert "tpr" not in public


# --- 類型判定：沒有 ground truth ------------------------------------------


def test_no_type_metric_requires_ground_truth(tmp_path: Path) -> None:
    records = baseline_records()
    records[1200] = make_record(
        "scam-0",
        COFACTS_SCAM.name,
        LABEL_SCAM,
        decided=True,
        abstained=False,
        confidence=0.9,
        scam_type="FAKE_INVESTMENT",
    )
    report = render_report(records, texts_for(records), TABLE, RUN_ID, tmp_path)
    forbidden = ("macro_f1", "macro-f1", "f1", "accuracy", "precision")
    fields = " ".join(Report.__dataclass_fields__).lower()
    assert not [word for word in forbidden if word in fields]
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    for line in markdown.splitlines():
        if "準確率" in line or "正確率" in line:
            assert "不" in line or "沒有" in line or "無法" in line, (
                f"報告出現一句宣稱準確率的文字：{line!r}"
            )
    assert "類型產出率" in markdown
    assert report.type_output_rate is not None
    with (tmp_path / "signals.csv").open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert not [word for word in forbidden if word in " ".join(header).lower()]


def test_merged_members_are_aligned_by_summing_their_case_titles() -> None:
    report = build_report(*_baseline(), TABLE, RUN_ID)
    merged = {
        share.scam_type: share
        for share in report.type_marginal_comparison
        if len(share.case_titles) > 1
    }
    assert set(merged) == {"INSTALLMENT_CANCEL", "ORDER_ANOMALY"}
    assert merged["INSTALLMENT_CANCEL"].official_cases == 1589 + 997
    assert merged["ORDER_ANOMALY"].official_cases == 839 + 266


# --- 多則情境樣本不足 -----------------------------------------------------


def test_multi_message_metrics_are_withheld_when_samples_are_scarce() -> None:
    report = build_report(*_baseline(), TABLE, RUN_ID)
    assert report.by_message_count.multi_count < MULTI_MESSAGE_MINIMUM
    assert report.by_message_count.sufficient is False
    assert report.by_message_count.multi is None


# --- 訊號：零命中不平滑 ---------------------------------------------------


def test_zero_ham_hits_are_not_smoothed_and_yield_a_weight_lower_bound() -> None:
    records = [
        make_record(f"s-{i}", COFACTS_SCAM.name, LABEL_SCAM, hits=("solicit_otp",))
        for i in range(100)
    ] + [make_record(f"h-{i}", COFACTS_HAM_AD.name, LABEL_HAM) for i in range(400)]
    estimate = estimate_signal("solicit_otp", records)
    assert estimate.p_hit_given_ham.numerator == 0
    assert estimate.p_hit_given_ham.value == 0.0
    assert estimate.estimability == LOWER_BOUND_ONLY
    assert estimate.weight_lower_bound == pytest.approx(math.log(1.0 / wilson_upper(0, 400)))


def test_a_signal_with_no_scam_hits_is_not_estimable_and_has_no_bound() -> None:
    records = [make_record(f"s-{i}", COFACTS_SCAM.name, LABEL_SCAM) for i in range(100)] + [
        make_record(f"h-{i}", COFACTS_HAM_AD.name, LABEL_HAM, hits=("solicit_otp",))
        for i in range(400)
    ]
    estimate = estimate_signal("solicit_otp", records)
    assert estimate.estimability == NOT_ESTIMABLE
    assert estimate.weight_lower_bound is None


# --- 校準 -----------------------------------------------------------------


def test_calibration_is_withheld_on_a_small_decided_subset() -> None:
    records = [
        make_record(f"s-{i}", COFACTS_SCAM.name, LABEL_SCAM, abstained=False, decided=True)
        for i in range(60)
    ]
    calibration = calibrate(records)
    assert calibration.bins == ()
    assert "樣本不足" in calibration.note
    assert calibration.decided_count == 60


def test_calibration_bin_count_follows_the_sample_size_not_a_fixed_ten() -> None:
    records = [
        make_record(f"s-{i}", COFACTS_SCAM.name, LABEL_SCAM, abstained=False, decided=True)
        for i in range(500)
    ]
    calibration = calibrate(records)
    assert len(calibration.bins) == 500 // MIN_BIN_SIZE
    assert len(calibration.bins) != 10
    assert all(bin_.observed.denominator >= MIN_BIN_SIZE for bin_ in calibration.bins)
    assert sum(bin_.observed.denominator for bin_ in calibration.bins) == 500
    assert MIN_SUBSET_SIZE == 100


def test_calibration_never_touches_the_sigmoid_offset() -> None:
    source = (REPO_ROOT / "tools" / "eval" / "calibration.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if "sigmoid_offset" in line:
            assert line.lstrip().startswith(("#", "`", "*", "-")) or '"' in line or "校準" in line


# --- 輸出的形狀 -----------------------------------------------------------


def test_render_report_writes_five_files_and_no_images(tmp_path: Path) -> None:
    render_report(*_baseline(), TABLE, RUN_ID, tmp_path)
    produced = sorted(path.name for path in tmp_path.iterdir())
    assert produced == ["cases.md", "rates.csv", "report.md", "signals.csv", "sweep.csv"]
    assert not [name for name in produced if name.endswith((".png", ".svg", ".jpg", ".pdf"))]


def test_run_id_carries_all_four_reproducibility_fields() -> None:
    text = str(RUN_ID)
    assert "testset=" in text
    assert "weights=" in text
    assert "blocklist=2025-12-31" in text
    assert "code=" in text


def test_report_always_states_the_abstention_rate_next_to_the_headline(tmp_path: Path) -> None:
    render_report(*_baseline(), TABLE, RUN_ID, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    headline_section = markdown.split("## 各子集")[0]
    assert "誤判率" in headline_section
    assert "棄權率" in headline_section


# --- 未切出子句的標記必須真的存在 ------------------------------------------


def test_no_clause_marker_still_matches_what_the_rules_emit() -> None:
    """標記是 `speech_act._detail()` 組出來的，它不匯出這個字串。

    這條測試用一次**真實命中**釘住它 —— 標記改了而 `run.py` 沒改時，
    這裡會失敗，而不是那個比率悄悄變成 0。
    """
    from scam_guard.check import CheckRegistry
    from scam_guard.rules.speech_act import register_speech_act_rules

    registry = CheckRegistry()
    register_speech_act_rules(registry)
    rules = {check.name: check for check in registry.enabled()}
    request = Request(messages=[Message(text="請至ATM依指示操作解除設定")])
    document = build_document(request.messages)
    results = rules["atm_operation"](request, document)
    assert results and results[0].hit
    assert NO_CLAUSE_MARKER in results[0].detail


def test_blocklist_type_claim_is_a_per_record_list_not_a_rate(tmp_path: Path) -> None:
    """160055 的類型宣稱只得逐則呈現 —— 用 `url_blocklist` 的輸出當標籤，
    再去量一個含 `url_blocklist` 的系統，結果恆為完美。"""
    records = baseline_records()
    records[1200] = make_record(
        "scam-0",
        COFACTS_SCAM.name,
        LABEL_SCAM,
        decided=True,
        abstained=False,
        confidence=0.9,
        scam_type="FAKE_INVESTMENT",
        hits=("url_blocklist",),
    )
    report = render_report(records, texts_for(records), TABLE, RUN_ID, tmp_path)
    assert [case.id for case in report.blocklist_type_cases] == ["scam-0"]
    fields = " ".join(Report.__dataclass_fields__)
    assert "blocklist_type_rate" not in fields
    with (tmp_path / "rates.csv").open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert not [row for row in rows if "blocklist_type" in ",".join(row)]
    assert "blocklist_type_claim" in (tmp_path / "cases.md").read_text(encoding="utf-8")
