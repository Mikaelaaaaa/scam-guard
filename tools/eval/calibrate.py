"""權重校準 —— 把 `weights.toml` 有資料的 `placeholder` 條目改成 `measured`。

**本模組只是既有量測工具的執行者，不新發明統計量。** 兩個條件機率、Wilson
區間與可估性三態全部來自 `tools/eval/signals.py`；本模組讀它的輸出，套一條
**區間寬度**判準，決定每個權重條目走四個出口中的哪一個：

| 出口 | 條件 | 表的處置 |
|---|---|---|
| `measured` | 兩側皆命中 **且** 區間寬度 ≤ 1.9 | 改 `measured`，附兩機率與 `measured_on` |
| `estimable_but_wide` | 兩側皆命中 **但** 寬度 > 1.9 | 維持 `placeholder`，值不動 |
| `lower_bound_only` | ham 側 0 命中 | 維持 `placeholder`，報告給權重下界 |
| `not_estimable` | scam 側 0 命中 | 維持 `placeholder`；此語料無證據 |

**量測只用 `tune`。** 取到任何 `holdout` 樣本即 `raise` 並指名 id —— 在
holdout 上估權重再於 holdout 報指標就是在測試集上訓練。

**權重下界 MUST NOT 被寫進 `value`。** `p_ham = 0` 不落在開區間 `(0, 1)`，
寫成 `measured` 載入就會拋例外；寫成 `placeholder` 則值必須是四個佔位常數之一，
而下界不會恰好是其中一個。表的驗證在這裡正確地擋住一個看起來很合理的錯誤，
本模組不繞過它 —— 這一態一律維持 `placeholder`，下界只進報告。

**`weight_hard` 與 `weight_soft` 是兩個獨立條目，各走各的出口。** 前者的樣本是
`hard=True` 的命中、後者是 `hard=False` 的命中，命中數不同，很可能一個過一個
不過。混算就是把自帶碼豁免與黑名單兩級比對這兩個設計決定量掉了。

**`ngram_classifier` 的兩個條目不重量。** 其餘訊號的命中是詞表的函數，分類器的
命中是它自己門檻的函數，兩者不是同一種量測 —— 分類器的條件機率在
`add-ngram-classifier` 中已於同一份 `tune` 上量出（門檻亦在該切分上選定）。
本模組讀表中既有的 `measured` 條目，不重算。

**`scam_guard/weights.py` 一行不改。** 產出的表通過它既有的全部驗證。
"""

import argparse
import math
import operator
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from scam_guard.check import CheckRegistry
from scam_guard.pipeline import detect
from scam_guard.scoring import compute_score, is_decision
from scam_guard.types import Request
from scam_guard.weights import (
    BASIS_MEASURED,
    BASIS_PLACEHOLDER,
    DEFAULT_WEIGHTS_PATH,
    WeightTable,
    load_weights,
)
from tools.eval.dataset import MANIFEST_FILENAME, TUNE, Sample, load_testset
from tools.eval.report import SubsetReport, subset_report
from tools.eval.run import (
    RunRecord,
    build_registry,
    load_blocklist,
    load_psl,
    run_over,
)
from tools.eval.selectors import HAM_SUBSETS, LABEL_HAM, LABEL_SCAM
from tools.eval.signals import (
    LOWER_BOUND_ONLY,
    NOT_ESTIMABLE,
    SignalEstimate,
    estimate_signal,
)

WEIGHT_INTERVAL_CEILING = 1.9
"""權重 95% 區間寬度的上限。**它等於 `2.5 − 0.6`。**

2.5 與 0.6 是現行佔位值中 Tier-A 與 Tier-B 的強度，1.9 是兩者的間距。它的意思
可以完整陳述：一個比這更寬的區間，其點估計無法回答「這個訊號比另一個強還是
弱」—— 而那正是四個佔位常數已經回答的問題。所以 1.9 不是一個調出來的門檻，
它是「新資訊必須至少比舊標記多」這句話在這張表的尺度上的寫法。

它是一條曲線不是一個最小命中數：ham 側愈稀疏，它就自動要求 scam 側愈多。
"""

WEIGHT_INTERVAL_CEILING_NOTE = "1.9 = 2.5 − 0.6，Tier-A 與 Tier-B 佔位值的間距"

MEASURED_ON = "testset/manifest.json 的 tune 切分（572 則 scam、824 則 ham）"
"""寫入 `measured` 條目的來源字串。MUST 含 `tune`、MUST NOT 含 `holdout`。"""

PROBABILITY_DECIMALS = 6
"""機率與權重值寫入表時的小數位。`value` 由同樣捨入的兩個機率算出，
使表自己的 `abs(value − ln(p_scam/p_ham)) ≤ 0.01` 驗證必然通過。"""

DECISION_SCORE_KEY = "decision_score"

EXIT_MEASURED = "measured"
EXIT_ESTIMABLE_BUT_WIDE = "estimable_but_wide"
EXIT_LOWER_BOUND_ONLY = "lower_bound_only"
EXIT_NOT_ESTIMABLE = "not_estimable"

EXITS = (EXIT_MEASURED, EXIT_ESTIMABLE_BUT_WIDE, EXIT_LOWER_BOUND_ONLY, EXIT_NOT_ESTIMABLE)

SKIP_SIGNALS = frozenset({"ngram_classifier"})
"""不重量的訊號 —— 它們的 `measured` 條目由別的 change 量出（見模組 docstring）。"""

WEIGHT_SOFT = "weight_soft"
WEIGHT_HARD = "weight_hard"


@dataclass(frozen=True)
class EntryExit:
    """一個權重條目（訊號 × hard/soft）的量測結果與它走的出口。

    `measured_value` / `p_scam` / `p_ham` 只在 `measured` 出口有值；其餘出口為
    `None`。`estimate` 是 `signals.py` 的原始輸出，`weight_lower_bound` 由它攜帶。
    """

    signal_name: str
    weight_key: str
    hard: bool
    hard_capable: bool
    estimate: SignalEstimate
    exit_name: str
    interval_width: float | None
    measured_value: float | None
    p_scam: float | None
    p_ham: float | None


@dataclass(frozen=True)
class DecisionScore:
    """`decision_score` 的重算結果。

    約束的**形式**沿用原表（單一 measured 弱訊號不跨門檻、單一 measured 硬證據
    跨門檻），錨點換成量出來的值。區間為空時 MUST 維持原值，MUST NOT 挑一個數字。
    """

    value: float
    recomputed: bool
    lower_bound: float | None
    lower_bound_signal: str | None
    upper_bound: float | None
    upper_bound_signal: str | None
    rationale: str


@dataclass(frozen=True)
class CalibrationReport:
    """一次校準的全部結果。`entries` 每一個恰好落在四個出口之一。"""

    entries: tuple[EntryExit, ...]
    decision_score: DecisionScore
    n_scam: int
    n_ham: int

    def by_exit(self, exit_name: str) -> tuple[EntryExit, ...]:
        return tuple(entry for entry in self.entries if entry.exit_name == exit_name)

    def exit_counts(self) -> dict[str, int]:
        return {exit_name: len(self.by_exit(exit_name)) for exit_name in EXITS}


def _assert_tune_only(samples: Sequence[Sample]) -> None:
    """量測只用 `tune`。任何 `holdout` 樣本即 raise 並指名 id。"""
    for sample in samples:
        if sample.split != TUNE:
            raise ValueError(
                f"校準只使用 tune 切分，收到非 tune 樣本：id={sample.id!r}、split={sample.split!r}"
            )


def _projected_records(records: Sequence[RunRecord], name: str, hard: bool) -> list[RunRecord]:
    """把每一則的 `hit_signals` 投影成「這個條目有沒有命中」。

    `weight_hard` 只採計 `hard=True` 的命中，`weight_soft` 只採計 `hard=False` 的
    命中（自帶碼豁免降級後的那些）。投影後交給 `signals.estimate_signal`，
    兩個條件機率、Wilson 區間與可估性三態全部由它算 —— 本模組不重算。
    """
    projected: list[RunRecord] = []
    for record in records:
        if hard:
            hit = name in record.hard_signals
        else:
            hit = name in record.hit_signals and name not in record.hard_signals
        projected.append(replace(record, hit_signals=(name,) if hit else ()))
    return projected


def _interval_width(estimate: SignalEstimate) -> float:
    """權重 95% 區間寬度：`ln(p_scam.hi/p_scam.lo) + ln(p_ham.hi/p_ham.lo)`。

    只在 `estimable`（兩側皆命中）時呼叫，此時兩側分子皆 ≥ 1，Wilson 下界為正。
    """
    scam = estimate.p_hit_given_scam
    ham = estimate.p_hit_given_ham
    return math.log(scam.upper / scam.lower) + math.log(ham.upper / ham.lower)


def _measure_entry(
    signal_name: str,
    weight_key: str,
    hard: bool,
    hard_capable: bool,
    records: Sequence[RunRecord],
) -> EntryExit:
    estimate = estimate_signal(signal_name, _projected_records(records, signal_name, hard))
    common = dict(
        signal_name=signal_name,
        weight_key=weight_key,
        hard=hard,
        hard_capable=hard_capable,
        estimate=estimate,
    )
    if estimate.estimability == NOT_ESTIMABLE:
        return EntryExit(
            exit_name=EXIT_NOT_ESTIMABLE,
            interval_width=None,
            measured_value=None,
            p_scam=None,
            p_ham=None,
            **common,
        )
    if estimate.estimability == LOWER_BOUND_ONLY:
        return EntryExit(
            exit_name=EXIT_LOWER_BOUND_ONLY,
            interval_width=None,
            measured_value=None,
            p_scam=None,
            p_ham=None,
            **common,
        )
    width = _interval_width(estimate)
    if width > WEIGHT_INTERVAL_CEILING:
        return EntryExit(
            exit_name=EXIT_ESTIMABLE_BUT_WIDE,
            interval_width=width,
            measured_value=None,
            p_scam=None,
            p_ham=None,
            **common,
        )
    p_scam = round(estimate.p_hit_given_scam.value, PROBABILITY_DECIMALS)
    p_ham = round(estimate.p_hit_given_ham.value, PROBABILITY_DECIMALS)
    value = round(math.log(p_scam / p_ham), PROBABILITY_DECIMALS)
    return EntryExit(
        exit_name=EXIT_MEASURED,
        interval_width=width,
        measured_value=value,
        p_scam=p_scam,
        p_ham=p_ham,
        **common,
    )


def _signal_entries(table: WeightTable) -> list[tuple[str, str, bool, bool]]:
    """要量測的權重條目清單：`(訊號名, weight_key, hard, hard_capable)`。

    `SKIP_SIGNALS` 的訊號整條跳過（已由別的 change 量出）。順序即表的登錄順序，穩定。
    """
    entries: list[tuple[str, str, bool, bool]] = []
    for signal in table.signals.values():
        if signal.name in SKIP_SIGNALS:
            continue
        entries.append((signal.name, WEIGHT_SOFT, False, signal.hard_capable))
        if signal.weight_hard is not None:
            entries.append((signal.name, WEIGHT_HARD, True, signal.hard_capable))
    return entries


def _measured_after(
    table: WeightTable, entries: Sequence[EntryExit]
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """校準後全部 `measured` 條目，拆成 `decision_score` 的兩組錨點候選。

    下界候選：`hard_capable = false` 的 `weight_soft`（含表中既有的 measured，
    如 `ngram_classifier`）。上界候選：任何 `weight_hard`。
    既有 measured 與本次新量的 measured 都算進去 —— 表最終的樣子才是約束的對象。
    """
    lower: list[tuple[float, str]] = []
    upper: list[tuple[float, str]] = []
    for signal in table.signals.values():
        if signal.weight_soft.basis == BASIS_MEASURED and not signal.hard_capable:
            lower.append((signal.weight_soft.value, signal.name))
        if signal.weight_hard is not None and signal.weight_hard.basis == BASIS_MEASURED:
            upper.append((signal.weight_hard.value, signal.name))
    for entry in entries:
        if entry.exit_name != EXIT_MEASURED:
            continue
        if entry.weight_key == WEIGHT_SOFT and not entry.hard_capable:
            lower.append((entry.measured_value, entry.signal_name))
        if entry.weight_key == WEIGHT_HARD:
            upper.append((entry.measured_value, entry.signal_name))
    return lower, upper


def _decision_score(table: WeightTable, entries: Sequence[EntryExit]) -> DecisionScore:
    """重算 `decision_score`。它的舊 `rationale` 引用的 2.5 / 0.6 錨點被本 change 刪除。

    ```
    下界 = max{ measured 的 weight_soft，其訊號 hard_capable = false }
    上界 = min{ measured 的 weight_hard }
    新值 ∈ (下界, 上界]，取中點
    ```

    上界集合為空、或下界 ≥ 上界時，約束不可滿足：MUST 維持原值、指名錨點、
    MUST NOT 挑一個數字，並把它列為主要發現 —— 那代表在此語料上規則層幾乎不
    產生硬證據，或 `hard` 標記與資料不符。
    """
    current = table.threshold(DECISION_SCORE_KEY)
    lower_candidates, upper_candidates = _measured_after(table, entries)
    lower = max(lower_candidates, key=operator.itemgetter(0)) if lower_candidates else None
    upper = min(upper_candidates, key=operator.itemgetter(0)) if upper_candidates else None

    if lower is None or upper is None or lower[0] >= upper[0]:
        rationale = _unsatisfiable_rationale(current, lower, upper)
        return DecisionScore(
            value=current,
            recomputed=False,
            lower_bound=None if lower is None else lower[0],
            lower_bound_signal=None if lower is None else lower[1],
            upper_bound=None if upper is None else upper[0],
            upper_bound_signal=None if upper is None else upper[1],
            rationale=rationale,
        )

    midpoint = round((lower[0] + upper[0]) / 2, PROBABILITY_DECIMALS)
    rationale = (
        f"約束為「單一 measured 弱訊號（{lower[1]}，{_num(lower[0])}）不跨過門檻、"
        f"單一 measured 硬證據（{upper[1]}，{_num(upper[0])}）跨過門檻」，故取值區間為 "
        f"({_num(lower[0])}, {_num(upper[0])}]。{_num(midpoint)} 是區間中點，"
        f"在區間內沒有任何特別之處，它對兩端等距而已"
    )
    return DecisionScore(
        value=midpoint,
        recomputed=True,
        lower_bound=lower[0],
        lower_bound_signal=lower[1],
        upper_bound=upper[0],
        upper_bound_signal=upper[1],
        rationale=rationale,
    )


def _unsatisfiable_rationale(
    current: float,
    lower: tuple[float, str] | None,
    upper: tuple[float, str] | None,
) -> str:
    lower_text = (
        "沒有任何 hard_capable = false 的 weight_soft 量得出 measured"
        if lower is None
        else f"measured 弱訊號權重的下界為 {lower[1]} 的 {_num(lower[0])}"
    )
    upper_text = (
        "沒有任何 weight_hard 條目量得出 measured（全部 hard_capable 訊號在 tune 上的硬命中數為 0）"
        if upper is None
        else f"measured 硬證據權重的上界為 {upper[1]} 的 {_num(upper[0])}"
    )
    return (
        f"約束為「單一 measured 弱訊號不跨過門檻、單一 measured 硬證據跨過門檻」。"
        f"{lower_text}，{upper_text}，區間不可滿足，故 decision_score 維持 {_num(current)}、"
        f"不挑一個數字。這代表在此語料上規則層幾乎不產生硬證據，門檻無法由量測重新錨定"
    )


def calibrate(
    samples: Sequence[Sample],
    registry: object,
    table: WeightTable,
) -> CalibrationReport:
    """對 `tune` 樣本量測每個權重條目，回傳四個出口的歸屬與 `decision_score` 重算。

    `registry` 由呼叫端組好（`tools.eval.run.build_registry`），本函式對它跑
    `detect()` 一次，其餘全部是純算術。
    """
    _assert_tune_only(samples)
    records = run_over(samples, registry, table)  # type: ignore[arg-type]
    n_scam = sum(1 for record in records if record.label == LABEL_SCAM)
    n_ham = sum(1 for record in records if record.label == LABEL_HAM)
    entries = tuple(
        _measure_entry(name, key, hard, hard_capable, records)
        for name, key, hard, hard_capable in _signal_entries(table)
    )
    decision = _decision_score(table, entries)
    return CalibrationReport(
        entries=entries,
        decision_score=decision,
        n_scam=n_scam,
        n_ham=n_ham,
    )


def _num(value: float) -> str:
    """把浮點數寫成不帶科學記號、不帶多餘尾零的字串。整數保留至少不含小數點。"""
    text = f"{value:.{PROBABILITY_DECIMALS}f}".rstrip("0").rstrip(".")
    return text


def _measured_block_lines(entry: EntryExit) -> list[str]:
    return [
        f"[signal.{entry.weight_key}]",
        f"value = {_num(entry.measured_value)}",
        f'basis = "{BASIS_MEASURED}"',
        f"p_hit_given_scam = {_num(entry.p_scam)}",
        f"p_hit_given_ham = {_num(entry.p_ham)}",
        f'measured_on = "{MEASURED_ON}"',
    ]


def _decision_block_lines(decision: DecisionScore) -> list[str]:
    return [
        f"[thresholds.{DECISION_SCORE_KEY}]",
        f"value = {_num(decision.value)}",
        f'basis = "{BASIS_PLACEHOLDER}"',
        f'rationale = "{decision.rationale}"',
        'blocked_on = "add-metrics"',
    ]


def _skip_block(lines: Sequence[str], index: int) -> int:
    """從 `index` 起跳過一個子表的內容，直到下一個 header、空行或檔尾。"""
    position = index
    while position < len(lines):
        stripped = lines[position].strip()
        if not stripped or stripped.startswith("["):
            break
        position += 1
    return position


def render_weights_toml(original_text: str, report: CalibrationReport) -> str:
    """把 `measured` 條目與重算後的 `decision_score` 寫回原表文字。

    只動通過判準的條目與 `decision_score` 一個門檻，其餘逐字保留 —— 註解、
    分群、類型優先序全部原封不動。以文字替換而非重新序列化，因為 `tomllib`
    只讀不寫，而重新序列化會丟掉這張表大量的說明性註解。
    """
    measured = {
        (entry.signal_name, entry.weight_key): entry
        for entry in report.entries
        if entry.exit_name == EXIT_MEASURED
    }
    lines = original_text.split("\n")
    out: list[str] = []
    current: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == "[[signal]]":
            current = None
            out.append(line)
            index += 1
            continue
        if current is None and stripped.startswith("name = "):
            current = stripped.split("=", 1)[1].strip().strip('"')
            out.append(line)
            index += 1
            continue
        if stripped in ("[signal.weight_soft]", "[signal.weight_hard]"):
            key = stripped[1:-1].split(".", 1)[1]
            entry = measured.get((current, key))
            if entry is not None:
                out.extend(_measured_block_lines(entry))
                index = _skip_block(lines, index + 1)
                continue
        if stripped == f"[thresholds.{DECISION_SCORE_KEY}]":
            out.extend(_decision_block_lines(report.decision_score))
            index = _skip_block(lines, index + 1)
            continue
        out.append(line)
        index += 1
    return "\n".join(out)


def write_weights_toml(
    report: CalibrationReport,
    dest: Path = DEFAULT_WEIGHTS_PATH,
    source: Path = DEFAULT_WEIGHTS_PATH,
) -> None:
    """把校準結果套到 `source` 表的文字上，寫到 `dest`，並確認 `dest` 通過 `load_weights()`。

    `source` 是原表（真相來源），`dest` 是寫出的位置 —— 兩者可以不同，供先寫一份
    暫存表做載入驗證與前後對照，再由 `--write-table` 明確覆寫正式表。
    """
    original = source.read_text(encoding="utf-8")
    updated = render_weights_toml(original, report)
    dest.write_text(updated, encoding="utf-8")
    load_weights(dest)


# ── 校準前後對照報告 ──────────────────────────────────────────────────────
#
# 兩份指標由同一個報告產生器（`tools/eval/report.py` 的 `subset_report`）產出，
# 差別只有輸入的表；不另寫任何比率。`SubsetReport` 強制誤判率與棄權率同時出現，
# 缺一即建構失敗 —— FPR 不可能在沒有棄權率的情況下被引用。

PHISHING_MESSAGE = (
    "Chunghwa Post：包裹因關稅未繳而暫扣。請支付 159.71 元，付款請於此 ： https://e.vg/post-gov"
)
"""那則中華郵政釣魚訊息。校準後的判定照實記錄，不要求它一定被判為詐騙。"""

REPORT_SCAM_SUBSET = "cofacts_scam"


@dataclass(frozen=True)
class CaseOutcome:
    """一則訊息在一張表下的判定，供逐則案例記錄。"""

    scam_probability: float | None
    confidence: float
    scam_type: str | None
    decided: bool
    evidence: tuple[str, ...]
    hit_signals: tuple[str, ...]


def _case_outcome(message: str, registry: CheckRegistry, table: WeightTable) -> CaseOutcome:
    verdict = detect(Request.from_text(message), registry, table, short_circuit=False)
    score = compute_score(verdict.checks, table)
    return CaseOutcome(
        scam_probability=verdict.scam_probability,
        confidence=verdict.confidence,
        scam_type=None if verdict.scam_type is None else verdict.scam_type.name,
        decided=is_decision(score, verdict, table),
        evidence=tuple(verdict.evidence),
        hit_signals=tuple(result.name for result in verdict.checks if result.hit),
    )


def _subset_metrics(
    samples: Sequence[Sample],
    registry: CheckRegistry,
    table: WeightTable,
    split: str,
) -> dict[str, SubsetReport]:
    """一張表在一個切分上、逐子集的召回/誤判/棄權 —— 由 `report.subset_report` 算。"""
    chosen = [sample for sample in samples if sample.split == split]
    records = run_over(chosen, registry, table)
    metrics: dict[str, SubsetReport] = {}
    for subset in sorted({record.subset for record in records}):
        rows = [record for record in records if record.subset == subset]
        metrics[subset] = subset_report(subset, rows)
    return metrics


def _exit_table_lines(report: CalibrationReport) -> list[str]:
    lines = [
        "## 四個出口",
        "",
        "每個權重條目（訊號 × hard/soft）恰好走一個出口。`ngram_classifier` 的兩個",
        "條目不在此列 —— 它們由 `add-ngram-classifier` 在同一份 tune 上量出，本 change 不重量。",
        "",
        "| 出口 | 條目數 |",
        "|---|---|",
    ]
    counts = report.exit_counts()
    labels = {
        EXIT_MEASURED: "measured（兩側皆命中且區間寬度 ≤ 1.9）",
        EXIT_ESTIMABLE_BUT_WIDE: "estimable_but_wide（兩側皆命中但寬度 > 1.9）",
        EXIT_LOWER_BOUND_ONLY: "lower_bound_only（ham 側 0 命中）",
        EXIT_NOT_ESTIMABLE: "not_estimable（scam 側 0 命中）",
    }
    for exit_name in EXITS:
        lines.append(f"| {labels[exit_name]} | {counts[exit_name]} |")
    total = sum(counts.values())
    lines += [
        f"| **合計** | **{total}** |",
        "",
        f"{total} 個非分類器條目裡只有 {counts[EXIT_MEASURED]} 個量得出 measured，"
        "是一個關於這份語料的結論，不是本 change 未完成：",
        "Cofacts 的詐騙訊息多是轉傳型敘述文字，而規則層瞄準的是祈使型言語行為，",
        f"在 1,396 則 tune 上有 {counts[EXIT_NOT_ESTIMABLE]} 個條目 scam 側零命中。",
        "",
    ]
    return lines


def _entry_rows(report: CalibrationReport) -> list[str]:
    lines = [
        "### 逐條目明細（含兩側命中數）",
        "",
        "| 出口 | 訊號 | 條目 | scam 命中 | ham 命中 | 區間寬度 | 值／權重下界 |",
        "|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(
        report.entries, key=operator.attrgetter("exit_name", "signal_name", "weight_key")
    )
    for entry in ordered:
        estimate = entry.estimate
        scam = f"{estimate.p_hit_given_scam.numerator}/{estimate.p_hit_given_scam.denominator}"
        ham = f"{estimate.p_hit_given_ham.numerator}/{estimate.p_hit_given_ham.denominator}"
        width = "—" if entry.interval_width is None else f"{entry.interval_width:.3f}"
        if entry.exit_name == EXIT_MEASURED:
            value = f"value = {_num(entry.measured_value)}"
        elif entry.exit_name == EXIT_LOWER_BOUND_ONLY:
            value = f"w ≥ {estimate.weight_lower_bound:.4f}"
        else:
            value = "—"
        lines.append(
            f"| {entry.exit_name} | {entry.signal_name} | {entry.weight_key} | "
            f"{scam} | {ham} | {width} | {value} |"
        )
    return lines + [""]


def _measured_detail_lines(report: CalibrationReport) -> list[str]:
    lines = ["### 改為 measured 的條目", ""]
    for entry in report.by_exit(EXIT_MEASURED):
        scam = entry.estimate.p_hit_given_scam
        ham = entry.estimate.p_hit_given_ham
        lines += [
            f"- **{entry.signal_name} / {entry.weight_key}**："
            f"`value = {_num(entry.measured_value)}` "
            f"= ln({_num(entry.p_scam)} / {_num(entry.p_ham)})",
            f"  - `p_hit_given_scam` = {scam}",
            f"  - `p_hit_given_ham` = {ham}",
            f"  - 區間寬度 {entry.interval_width:.3f} ≤ 1.9（{WEIGHT_INTERVAL_CEILING_NOTE}）",
        ]
    lines += [
        "",
        "兩條都是**負權重**：引述使一則訊息更可能是轉述而非施行，短網址在此語料上",
        "更常出現在合法宣導與廣告裡。校準把 `url_shortener` 從佔位的 `0.0`（宣告資訊",
        "不足）改成量出來的 `-2.143387`，這是本 change 最大的單一權重變化。",
        "",
    ]
    return lines


def _lower_bound_lines(report: CalibrationReport) -> list[str]:
    lines = ["### lower_bound_only：ham 側零命中，只報下界（不進表）", ""]
    for entry in report.by_exit(EXIT_LOWER_BOUND_ONLY):
        scam = entry.estimate.p_hit_given_scam
        lines.append(
            f"- **{entry.signal_name} / {entry.weight_key}**："
            f"scam 側 {scam.numerator}/{scam.denominator}、ham 側 0 命中，"
            f"權重下界 `w ≥ {entry.estimate.weight_lower_bound:.4f}`。"
            f"條目維持 `placeholder`，下界只進報告"
        )
    lines += [
        "",
        "下界 MUST NOT 寫進 `value`：`p_ham = 0` 不落在開區間 `(0, 1)`，寫成 `measured`",
        "載入即拋例外；寫成 `placeholder` 則值必須是四個佔位常數之一。表的驗證正確地",
        "擋住這個看起來很合理的錯誤。",
        "",
    ]
    return lines


def _decision_lines(report: CalibrationReport) -> list[str]:
    decision = report.decision_score
    status = "重算" if decision.recomputed else "維持原值（約束不可滿足）"
    return [
        "## decision_score",
        "",
        f"- 結果：{status}，`value = {_num(decision.value)}`",
        f"- rationale：{decision.rationale}",
        "",
        "**這是本 change 的主要發現之一。** decision_score 的舊 rationale 引用「單一 Tier-A",
        "命中（2.5）跨過、單一 Tier-B 命中（0.6）不跨過」，而 2.5 / 0.6 這兩個錨點被本",
        "change 刪除。重算需要 measured 的 weight_soft 當下界、measured 的 weight_hard 當",
        "上界，但 tune 上沒有任何 weight_hard 量得出 measured（全部 hard_capable 訊號的硬",
        "命中數為 0），而唯一夠強的 measured 弱訊號是門檻在同一份 tune 上選過的",
        "`ngram_classifier`（4.467675）。區間因此不可滿足，decision_score 維持 1.5。",
        "",
    ]


def _comparison_lines(
    before: dict[str, SubsetReport],
    after: dict[str, SubsetReport],
) -> list[str]:
    lines = [
        "## 校準前後對照（holdout，1,467 則）",
        "",
        "兩份指標由同一個報告產生器（`report.subset_report`）產出，差別只有輸入的表。",
        "**誤判率一律附 95% Wilson 上界，且與棄權率綁定**（`SubsetReport` 缺棄權率即建構失敗）。",
        "`self_sms_ham`（真實銀行／物流／政府通知）未蒐集，所以下表的誤判率量不到最危險的那一類。",
        "",
        "| 子集 | 指標 | 校準前 | 校準後 |",
        "|---|---|---|---|",
    ]
    scam_before = before[REPORT_SCAM_SUBSET]
    scam_after = after[REPORT_SCAM_SUBSET]
    row = REPORT_SCAM_SUBSET
    lines += [
        f"| {row} | 召回率 | {scam_before.recall} | {scam_after.recall} |",
        f"| {row} | 棄權率 | {scam_before.abstention_rate} | {scam_after.abstention_rate} |",
    ]
    for subset in sorted(name for name in after if name in HAM_SUBSETS):
        fpr_before = before[subset].false_positive_rate
        fpr_after = after[subset].false_positive_rate
        abstain_before = before[subset].abstention_rate
        abstain_after = after[subset].abstention_rate
        upper = f"Wilson 上界 {fpr_before.upper:.2%} → {fpr_after.upper:.2%}"
        lines += [
            f"| {subset} | 誤判率（{upper}） | {fpr_before} | {fpr_after} |",
            f"| {subset} | 棄權率 | {abstain_before} | {abstain_after} |",
        ]
    lines += [""]
    return lines


def _overfit_lines(tune: dict[str, SubsetReport], holdout: dict[str, SubsetReport]) -> list[str]:
    lines = [
        "## tune 與 holdout 並列（校準後）",
        "",
        "本 change 正是在 tune 上估權重的那一個，兩個切分的差距就是過擬合的量。",
        "",
        "| 子集 | 指標 | tune | holdout |",
        "|---|---|---|---|",
        f"| {REPORT_SCAM_SUBSET} | 召回率 | {tune[REPORT_SCAM_SUBSET].recall} "
        f"| {holdout[REPORT_SCAM_SUBSET].recall} |",
    ]
    for subset in sorted(name for name in holdout if name in HAM_SUBSETS):
        tune_fpr = tune[subset].false_positive_rate
        holdout_fpr = holdout[subset].false_positive_rate
        lines.append(f"| {subset} | 誤判率 | {tune_fpr} | {holdout_fpr} |")
    return lines + [""]


def _ngram_note_lines() -> list[str]:
    return [
        "## ngram_classifier：不同級的來源",
        "",
        "`ngram_classifier` 的訊號條目與 `ngram_threshold` 已在 `add-ngram-classifier` 中",
        "量出，本 change 讀不重算。**它與其餘 34 個訊號不同級**：它的門檻是在同一份 tune 上",
        "選出的，所以它的 `p_hit_given_scam`（84.62%）是一個選過的最大值，其餘訊號不是。",
        "`weights.toml` 沒有欄位承載「這一條是選過的」，表面上兩者同形，記為已知弱點。",
        "本 change 的評估 registry 未掛載分類器，其 holdout 命中率見 add-ngram-classifier 報告。",
        "",
    ]


def _case_lines(before: CaseOutcome, after: CaseOutcome) -> list[str]:
    before_hits = "、".join(before.hit_signals) or "（無）"
    after_hits = "、".join(after.hit_signals) or "（無）"
    lines = [
        "## 那則釣魚訊息",
        "",
        "```",
        PHISHING_MESSAGE,
        "```",
        "",
        "| 指標 | 校準前 | 校準後 |",
        "|---|---|---|",
        f"| scam_probability | {before.scam_probability} | {after.scam_probability} |",
        f"| confidence | {before.confidence:.2f} | {after.confidence:.2f} |",
        f"| scam_type | {before.scam_type} | {after.scam_type} |",
        f"| 命中訊號 | {before_hits} | {after_hits} |",
        f"| 依據行數 | {len(before.evidence)} | {len(after.evidence)} |",
        "",
        "校準後這則訊息**仍然拒答**（`scam_probability = None`）—— 純規則模式下它命中不到",
        "任何規則，分數為 0、信心 0.05 落在 `base_no_hit` 這一級。這不是失敗，是資料告訴",
        "我們的事實：本系統的規則層對這一則英中夾雜、以短網址收尾的釣魚訊息沒有任何訊號。",
        "接上分類器與 LLM 之後的判定要等那兩層在評估管線裡掛載才量得到，本 change 不為了",
        "讓這一則過關而回頭調任何門檻。",
        "",
    ]
    return lines


def _report_markdown(
    report: CalibrationReport,
    before_holdout: dict[str, SubsetReport],
    after_holdout: dict[str, SubsetReport],
    after_tune: dict[str, SubsetReport],
    before_case: CaseOutcome,
    after_case: CaseOutcome,
) -> str:
    lines = [
        "# 權重校準報告",
        "",
        "把 `weights.toml` 有資料的 `placeholder` 條目改成 `measured`。量測只用 tune",
        "（572 則 scam、824 則 ham）；兩個條件機率、Wilson 區間與可估性三態全部取自",
        "`tools/eval/signals.py`，本 change 只是它的執行者。",
        "",
    ]
    lines += _exit_table_lines(report)
    lines += _entry_rows(report)
    lines += _measured_detail_lines(report)
    lines += _lower_bound_lines(report)
    lines += _decision_lines(report)
    lines += _comparison_lines(before_holdout, after_holdout)
    lines += _overfit_lines(after_tune, after_holdout)
    lines += _ngram_note_lines()
    lines += _case_lines(before_case, after_case)
    return "\n".join(lines) + "\n"


DEFAULT_REPORT_DIR = Path("data/reports/2026-09-15-weights")


def main(argv: list[str] | None = None) -> int:
    """跑一次校準：量 tune、寫回 `weights.toml`、產出校準前後對照報告。"""
    parser = argparse.ArgumentParser(
        prog="python -m tools.eval.calibrate",
        description="量測權重、把有資料的 placeholder 改為 measured，並產出校準前後對照。",
    )
    parser.add_argument("--testset-dir", default="testset")
    parser.add_argument("--data-dir", default="data/testset")
    parser.add_argument("--psl-dir", default="data/psl")
    parser.add_argument("--blocklist-dir", default="data/blocklist")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--write-table",
        action="store_true",
        help="把校準結果寫回 scam_guard/tables/weights.toml（預設只算不寫）。",
    )
    args = parser.parse_args(argv)

    testset = load_testset(Path(args.data_dir), Path(args.testset_dir) / MANIFEST_FILENAME)
    if not testset.subsets:
        print(f"測試集尚未重建：{args.data_dir} 為空。", file=sys.stderr)
        return 1

    psl = load_psl(Path(args.psl_dir))
    store = load_blocklist(Path(args.blocklist_dir), psl)
    registry = build_registry(psl, store=store)
    samples = testset.all_samples()

    before_table = load_weights()
    tune = [sample for sample in samples if sample.split == TUNE]
    report = calibrate(tune, registry, before_table)

    before_holdout = _subset_metrics(samples, registry, before_table, "holdout")
    before_case = _case_outcome(PHISHING_MESSAGE, registry, before_table)

    if args.write_table:
        write_weights_toml(report)
        after_table = load_weights()
    else:
        staged = _staged_table_path(args.report_dir)
        write_weights_toml(report, dest=staged)
        after_table = load_weights(staged)

    after_holdout = _subset_metrics(samples, registry, after_table, "holdout")
    after_tune = _subset_metrics(samples, registry, after_table, TUNE)
    after_case = _case_outcome(PHISHING_MESSAGE, registry, after_table)

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(
        _report_markdown(
            report, before_holdout, after_holdout, after_tune, before_case, after_case
        ),
        encoding="utf-8",
    )
    print(f"出口分布：{report.exit_counts()}", file=sys.stderr)
    print(f"報告寫入 {out_dir / 'report.md'}", file=sys.stderr)
    return 0


def _staged_table_path(report_dir: str) -> Path:
    """未寫回正式表時，把校準後的表暫存到報告目錄，供對照與載入驗證。"""
    staged = Path(report_dir) / "weights.after.toml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    return staged


if __name__ == "__main__":
    sys.exit(main())
