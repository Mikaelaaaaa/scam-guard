"""權重校準 —— 四個出口的歸屬、下界不進表、`weights.py` 不改、`decision_score` 重算。

大部分測試以合成的 `RunRecord` 執行，不吃資料：校準的邏輯是「給定命中數，這個
條目走哪個出口」，而那不該依賴任何一份語料。吃資料的端到端行為由
`python -m tools.eval.calibrate` 在開發時執行，其產物是 `weights.toml` 本身，
而本檔的「表通過驗證」測試就是對那份產物的驗收。
"""

import hashlib
import math
import tomllib
from pathlib import Path

import pytest

from scam_guard.weights import (
    BASIS_MEASURED,
    BASIS_PLACEHOLDER,
    MEASURED_TOLERANCE,
    load_weights,
)
from tools.eval.calibrate import (
    EXIT_ESTIMABLE_BUT_WIDE,
    EXIT_LOWER_BOUND_ONLY,
    EXIT_MEASURED,
    EXIT_NOT_ESTIMABLE,
    EXITS,
    WEIGHT_HARD,
    WEIGHT_SOFT,
    CalibrationReport,
    DecisionScore,
    EntryExit,
    _measure_entry,
    calibrate,
    render_weights_toml,
)
from tools.eval.run import RunRecord
from tools.eval.selectors import LABEL_HAM, LABEL_SCAM, SELF_SMS_HAM, TUNE
from tools.eval.dataset import Sample
from tools.eval.signals import estimate_signal

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PY = REPO_ROOT / "scam_guard" / "weights.py"
CALIBRATE_PY = REPO_ROOT / "tools" / "eval" / "calibrate.py"

WEIGHTS_PY_SHA256 = "c51bcedec1f9be78f48690a904f0a090ac03bf4b9a0002627c55f00a4deb305a"
"""`scam_guard/weights.py` 的雜湊；由 add-strong-signal-tier 合法更新。

若日後有 change 合法修改 `weights.py`，那個 change MUST 明確更新這個常數 ——
它擋的是本 change（以及未來聲稱不動驗證邏輯的 change）悄悄改到表的驗證規則。
"""


def make_record(
    identifier: str,
    label: str,
    *,
    hit_signals: tuple[str, ...] = (),
    hard_signals: tuple[str, ...] = (),
) -> RunRecord:
    """一則只填了校準會讀的欄位的合成紀錄，其餘給不影響判斷的佔位值。"""
    return RunRecord(
        id=identifier,
        subset="cofacts_scam" if label == LABEL_SCAM else "cofacts_ham_ad",
        split=TUNE,
        label=label,
        score=0.0,
        confidence=0.05,
        probability=None,
        abstained=True,
        decided=False,
        scam_type=None,
        contradicted=False,
        quotation_hit=False,
        blocklist_hit=False,
        simplified=False,
        hit_signals=hit_signals,
        hard_signals=hard_signals,
        typed_signals=(),
        shadowed_signals=(),
        speech_act_hits=0,
        no_clause_hits=0,
        evidence=(),
        actions=(),
    )


def records_with(
    name: str,
    *,
    n_scam: int,
    scam_hits: int,
    n_ham: int,
    ham_hits: int,
    hard: bool = False,
) -> list[RunRecord]:
    """湊出一批紀錄，讓 `name` 在指定側各命中指定次數。

    `hard=True` 時命中寫進 `hard_signals`（也寫進 `hit_signals`，真實紀錄如此）；
    `hard=False` 時只寫 `hit_signals`。這對應 `weight_hard` 與 `weight_soft` 的
    兩種投影。
    """
    records: list[RunRecord] = []
    for index in range(n_scam):
        hit = index < scam_hits
        records.append(_record_for(name, LABEL_SCAM, f"s{index}", hit, hard))
    for index in range(n_ham):
        hit = index < ham_hits
        records.append(_record_for(name, LABEL_HAM, f"h{index}", hit, hard))
    return records


def _record_for(name: str, label: str, identifier: str, hit: bool, hard: bool) -> RunRecord:
    if not hit:
        return make_record(identifier, label)
    if hard:
        return make_record(identifier, label, hit_signals=(name,), hard_signals=(name,))
    return make_record(identifier, label, hit_signals=(name,))


# --- 量測只用 tune -------------------------------------------------------


def test_holdout_sample_raises_and_names_id() -> None:
    """混入 holdout 樣本即拋例外，訊息含該樣本 id。`self_sms_ham` 恆為 holdout。"""
    holdout = Sample(id="holdout-x", text="內文", subset=SELF_SMS_HAM.name)
    with pytest.raises(ValueError, match="holdout-x"):
        calibrate([holdout], registry=None, table=load_weights())


# --- 四個出口 ------------------------------------------------------------


def test_not_estimable_when_scam_side_zero() -> None:
    records = records_with("x", n_scam=572, scam_hits=0, n_ham=824, ham_hits=3)
    entry = _measure_entry("x", WEIGHT_SOFT, False, False, records)
    assert entry.exit_name == EXIT_NOT_ESTIMABLE
    assert entry.measured_value is None


def test_lower_bound_only_when_ham_side_zero() -> None:
    records = records_with("x", n_scam=572, scam_hits=7, n_ham=824, ham_hits=0)
    entry = _measure_entry("x", WEIGHT_SOFT, False, False, records)
    assert entry.exit_name == EXIT_LOWER_BOUND_ONLY
    assert entry.measured_value is None
    assert entry.estimate.weight_lower_bound is not None


def test_estimable_but_wide_when_interval_too_wide() -> None:
    """兩側皆命中但稀疏 —— 區間寬度 > 1.9，維持 placeholder。"""
    records = records_with("x", n_scam=572, scam_hits=2, n_ham=824, ham_hits=2)
    entry = _measure_entry("x", WEIGHT_SOFT, False, False, records)
    assert entry.exit_name == EXIT_ESTIMABLE_BUT_WIDE
    assert entry.interval_width > 1.9
    assert entry.measured_value is None


def test_measured_when_both_sides_hit_and_interval_narrow() -> None:
    """quotation 的真實形狀：26/572 與 180/824，寬度約 1.0 ≤ 1.9。"""
    records = records_with("x", n_scam=572, scam_hits=26, n_ham=824, ham_hits=180)
    entry = _measure_entry("x", WEIGHT_SOFT, False, False, records)
    assert entry.exit_name == EXIT_MEASURED
    assert entry.interval_width <= 1.9
    assert entry.measured_value is not None


def test_measured_value_equals_log_ratio() -> None:
    records = records_with("x", n_scam=572, scam_hits=26, n_ham=824, ham_hits=180)
    entry = _measure_entry("x", WEIGHT_SOFT, False, False, records)

    expected = math.log(entry.p_scam / entry.p_ham)
    assert abs(entry.measured_value - expected) <= MEASURED_TOLERANCE


def test_every_entry_takes_exactly_one_exit() -> None:
    """四個出口的條目數加總等於條目總數 —— 每個條目恰好走一個出口。"""
    entries = tuple(
        EntryExit(
            signal_name=f"s{i}",
            weight_key=WEIGHT_SOFT,
            hard=False,
            hard_capable=False,
            estimate=estimate_signal(
                f"s{i}", records_with(f"s{i}", n_scam=10, scam_hits=i % 3, n_ham=10, ham_hits=1)
            ),
            exit_name=exit_name,
            interval_width=None,
            measured_value=None,
            p_scam=None,
            p_ham=None,
        )
        for i, exit_name in enumerate(EXITS)
    )
    report = CalibrationReport(
        entries=entries, decision_score=_dummy_decision(), n_scam=10, n_ham=10
    )
    counts = report.exit_counts()
    assert sum(counts.values()) == len(entries)
    assert all(entry.exit_name in EXITS for entry in entries)


# --- weight_hard 與 weight_soft 分開量測 ---------------------------------


def test_hard_and_soft_measured_independently() -> None:
    """同一訊號的硬命中與軟命中數不同時，兩個條目走不同出口。"""
    records: list[RunRecord] = []
    # scam：1 則硬命中、其餘無；ham：多則軟命中
    records.append(make_record("s0", LABEL_SCAM, hit_signals=("g",), hard_signals=("g",)))
    for i in range(571):
        records.append(make_record(f"s{i + 1}", LABEL_SCAM))
    for i in range(180):
        records.append(make_record(f"h{i}", LABEL_HAM, hit_signals=("g",)))
    for i in range(644):
        records.append(make_record(f"hz{i}", LABEL_HAM))
    # scam 側也給軟命中，使 weight_soft 兩側皆命中
    for i in range(26):
        records[1 + i] = make_record(f"s{i + 1}", LABEL_SCAM, hit_signals=("g",))

    hard_entry = _measure_entry("g", WEIGHT_HARD, True, True, records)
    soft_entry = _measure_entry("g", WEIGHT_SOFT, False, True, records)
    # 硬命中：scam 1、ham 0 → lower_bound_only；軟命中：scam 26、ham 180 → measured
    assert hard_entry.exit_name == EXIT_LOWER_BOUND_ONLY
    assert soft_entry.exit_name == EXIT_MEASURED
    assert hard_entry.exit_name != soft_entry.exit_name


# --- 下界不進表；render 只動該動的 --------------------------------------

MINIMAL_TABLE = """\
[[signal]]
name = "alpha"
group = "g_alpha"
hard_capable = false
[signal.weight_soft]
value = -1.5
basis = "placeholder"
inherited_from = "來源"
blocked_on = "add-testset"

[[signal]]
name = "beta"
group = "g_beta"
hard_capable = false
[signal.weight_soft]
value = 0.6
basis = "placeholder"
inherited_from = "來源"
blocked_on = "add-testset"

[thresholds.decision_score]
value = 1.5
basis = "placeholder"
rationale = "舊 rationale"
blocked_on = "add-metrics"
"""


def test_render_converts_measured_and_leaves_lower_bound_untouched() -> None:
    """`measured` 條目被改寫；`lower_bound_only` 的條目一位元組都不動。"""
    measured = EntryExit(
        signal_name="alpha",
        weight_key=WEIGHT_SOFT,
        hard=False,
        hard_capable=False,
        estimate=estimate_signal(
            "alpha", records_with("alpha", n_scam=572, scam_hits=26, n_ham=824, ham_hits=180)
        ),
        exit_name=EXIT_MEASURED,
        interval_width=1.0,
        measured_value=-1.569821,
        p_scam=0.045455,
        p_ham=0.218447,
    )
    lower = EntryExit(
        signal_name="beta",
        weight_key=WEIGHT_SOFT,
        hard=False,
        hard_capable=False,
        estimate=estimate_signal(
            "beta", records_with("beta", n_scam=572, scam_hits=7, n_ham=824, ham_hits=0)
        ),
        exit_name=EXIT_LOWER_BOUND_ONLY,
        interval_width=None,
        measured_value=None,
        p_scam=None,
        p_ham=None,
    )
    report = CalibrationReport(
        entries=(measured, lower), decision_score=_dummy_decision(), n_scam=572, n_ham=824
    )
    rendered = render_weights_toml(MINIMAL_TABLE, report)
    document = tomllib.loads(rendered)

    alpha = _signal_named(document, "alpha")["weight_soft"]
    assert alpha["basis"] == BASIS_MEASURED
    assert "p_hit_given_scam" in alpha
    assert "inherited_from" not in alpha

    beta = _signal_named(document, "beta")["weight_soft"]
    assert beta["basis"] == BASIS_PLACEHOLDER
    assert beta["value"] == 0.6
    assert beta["inherited_from"] == "來源"
    # 逐位元組：beta 的四行原封不動
    assert 'value = 0.6\nbasis = "placeholder"\ninherited_from = "來源"' in rendered


def test_render_rewrites_decision_score_rationale_but_keeps_placeholder() -> None:
    report = CalibrationReport(
        entries=(),
        decision_score=DecisionScore(
            value=1.5,
            recomputed=False,
            lower_bound=4.467675,
            lower_bound_signal="ngram_classifier",
            upper_bound=None,
            upper_bound_signal=None,
            rationale="新 rationale 指名 ngram_classifier",
        ),
        n_scam=0,
        n_ham=0,
    )
    document = tomllib.loads(render_weights_toml(MINIMAL_TABLE, report))
    decision = document["thresholds"]["decision_score"]
    assert decision["value"] == 1.5
    assert decision["basis"] == BASIS_PLACEHOLDER
    assert "ngram_classifier" in decision["rationale"]


# --- 產出的表通過驗證，且七個信心門檻不動 -------------------------------


def test_produced_table_loads() -> None:
    """本 change 寫出的 `weights.toml` 通過 `load_weights()` 的全部驗證。"""
    load_weights()


def test_seven_confidence_thresholds_unchanged() -> None:
    table = load_weights()
    expected = {
        "base_hard": 0.90,
        "base_multi_group": 0.55,
        "base_single_group": 0.35,
        "base_no_hit": 0.05,
        "cap_contradiction": 0.30,
        "cap_unseen_pattern": 0.35,
        "cap_truncated": 0.50,
        "confidence_floor": 0.40,
    }
    for key, value in expected.items():
        assert table.threshold(key) == value
        assert table.thresholds[key].basis == BASIS_PLACEHOLDER


def test_sigmoid_offset_and_prior_unchanged() -> None:
    table = load_weights()
    assert table.threshold("sigmoid_offset") == 0.0
    assert table.threshold("sigmoid_temperature") == 1.0
    assert table.threshold("prior_log_odds") == 0.0


def test_ngram_entries_not_remeasured() -> None:
    """`ngram_classifier` 與 `ngram_threshold` 由 add-ngram-classifier 量出，本 change 不動。"""
    table = load_weights()
    assert table.signals["ngram_classifier"].weight_soft.value == 4.467675
    assert table.signals["ngram_classifier"].weight_soft.basis == BASIS_MEASURED
    assert table.threshold("ngram_threshold") == 0.376648


def test_all_measured_entries_satisfy_log_ratio() -> None:
    table = load_weights()

    for signal in table.signals.values():
        for weight in (signal.weight_soft, signal.weight_hard):
            if weight is None or weight.basis != BASIS_MEASURED:
                continue
            expected = math.log(weight.p_hit_given_scam / weight.p_hit_given_ham)
            assert abs(weight.value - expected) <= MEASURED_TOLERANCE


def test_measured_signal_entries_are_quotation_and_url_shortener() -> None:
    """本 change 只把 quotation 與 url_shortener 的 weight_soft 改為 measured。"""
    table = load_weights()
    measured = {
        (signal.name, key)
        for signal in table.signals.values()
        for key, weight in (
            ("weight_soft", signal.weight_soft),
            ("weight_hard", signal.weight_hard),
        )
        if weight is not None and weight.basis == BASIS_MEASURED
    }
    assert ("quotation", "weight_soft") in measured
    assert ("url_shortener", "weight_soft") in measured
    assert ("ngram_classifier", "weight_soft") in measured


# --- 不改核心、不重寫統計、不掃門檻 -------------------------------------


def test_weights_py_unchanged() -> None:
    """`scam_guard/weights.py` 一位元組不改（見 `WEIGHTS_PY_SHA256` 的說明）。"""
    actual = hashlib.sha256(WEIGHTS_PY.read_bytes()).hexdigest()
    assert actual == WEIGHTS_PY_SHA256


def test_calibrate_does_not_reimplement_wilson_or_three_state() -> None:
    """條件機率、Wilson 與三態判定取自 signals.py / stats.py，calibrate.py 不重寫。"""
    source = CALIBRATE_PY.read_text(encoding="utf-8")
    assert "math.sqrt" not in source
    assert "scam_hits == 0" not in source
    assert "Z_95" not in source


def test_calibrate_does_not_sweep_thresholds() -> None:
    """門檻掃描屬 add-ablation，calibrate 不呼叫任何掃描函式。"""
    source = CALIBRATE_PY.read_text(encoding="utf-8")
    assert "from tools.eval.sweep" not in source
    assert "sweep(" not in source


# --- 測試輔助 ------------------------------------------------------------


def _dummy_decision() -> DecisionScore:
    return DecisionScore(
        value=1.5,
        recomputed=False,
        lower_bound=None,
        lower_bound_signal=None,
        upper_bound=None,
        upper_bound_signal=None,
        rationale="測試佔位",
    )


def _signal_named(document: dict, name: str) -> dict:
    for raw in document["signal"]:
        if raw["name"] == name:
            return raw
    raise AssertionError(f"找不到訊號 {name!r}")
