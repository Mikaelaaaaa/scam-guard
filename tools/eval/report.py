"""報告 —— **全系統唯一的指標輸出入口**。

## 誤判率不得在沒有棄權率的情況下被取得

`add-score-compute` 已經把規則寫成 requirement。本模組要回答的是**怎麼讓它
成立**，因為一句 `MUST NOT` 攔不住 `report.fpr`。處置：

- 模組層**沒有** `fpr()` 這類函式。
- 唯一的公開入口是 `render_report()`，它一次產出混淆矩陣、比率、區間與棄權率。
- `SubsetReport` 的三個比率欄位皆為必填，省略任一即 `TypeError`；
  它沒有 `__getitem__`，也沒有只回傳其中一者的方法。
- `SubsetReport.to_markdown()` 固定輸出全部欄位，**沒有參數可以關掉其中一欄**。

這是本專案用過的模式：`add-confidence` 的 `compute_confidence()` 刻意不接收
`Score`，理由是「拿不到的東西不會被誤用」。這裡是同一件事的另一個方向 ——
**取不到單獨的誤判率，就不可能寫出一個只有誤判率的句子。**

擋不擋得住有人自己寫一行除法？擋不住。這與 `add-cofacts-fetch` 對 banned-api
的自評相同：「這不是防禦機制，是提醒機制，對象是自己人，繞過它需要刻意」。
真正的保護是報告產出器 —— 報告裡的數字全部來自 `render_report()`，
而它不可能產出一個沒有棄權率的誤判率。

## ham 子集分開，頭條取最差

`Report.headline_false_positive_rate` 取三個 ham 子集中 Wilson 上界最大者。
**不提供合併後的欄位**（沒有 `pooled_false_positive_rate` 或等價屬性），
而不是提供它並在文件裡說不要用：真的需要合併的人得自己寫出 `n1+n2+n3`，
而那一行在 review 時看得見。

## 類型判定沒有 ground truth

`add-testset` 已確認 Cofacts 的標籤不含 165 分類，測試集不得含類型欄位。
因此本模組 **MUST NOT 輸出類型的 macro-F1、準確率、precision 或 recall**。
`規劃.md` M9 的「類型分類的 macro-F1 與逐類表現」**無法履行**。
改報三件不需要正確答案也能算的事：類型產出率、與 165 官方分項統計的**邊際
分布**比較（弱證據，只能用來發現異常）、消融下的類型穩定性。
"""

import csv
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from scam_guard.types import ScamType, case_titles
from scam_guard.weights import WeightTable
from tools.eval.calibration import Calibration, calibrate
from tools.eval.run import RunId, RunRecord
from tools.eval.selectors import (
    HAM_SUBSETS,
    HOLDOUT,
    LABEL_CASE_ONLY,
    LABEL_HAM,
    LABEL_SCAM,
    MULTI_MESSAGE_SUBSETS,
    TUNE,
)
from tools.eval.signals import SignalEstimate, estimate_signals
from tools.eval.stats import (
    RULE_OF_THREE_NOTE,
    Rate,
    sample_size_table,
)
from tools.eval.sweep import SWEEP_COLUMNS, SweepPoint, sweep

MULTI_MESSAGE_MINIMUM = 10
"""少於此數時多則情境的任何指標一律不報，只列逐則輸出。"""

DEFAULT_DECISION_SCORES: tuple[float, ...] = tuple(index * 0.1 for index in range(0, 61))
"""分數軸：0.0 到 6.0，步長 0.1。

上界取 6.0 的理由是現行表的群組貢獻上限為每群 2.5，而實際同時命中三個以上
群組的樣本極少；下界取 0.0 使「完全不設分數門檻」也在圖上。
"""

DEV_SAMPLE_MEASUREMENTS: Mapping[str, str] = {
    "url_tld_risk": "600 則正例 + 599 則難負例中 15 次命中，**全在 scam 側**，ham 側零命中",
    "url_host_shape": "1,199 則中**零命中**",
    "url_blocklist": "因 `play.google.com` 被列入 165 假投資清單而產生**一筆硬證據偽陽性**",
}
"""`add-url-check` 在**開發樣本**上的實測結論，供與 holdout 並列。

⚠️ **這三個數字是在已經被用來修門檻的那批樣本上量的。** 該 change 的三個誤判
類別（`pse.is`、`lin.ee`、`play.google.com`）全部是在這 1,199 則上實測出來並
據以修改門檻的，所以它們是**調過之後**的數字，不是獨立的量測。
並列的作用是讓讀者看得出「調過的那一側」與「沒調過的那一側」差多少。
"""

CASE_FALSE_POSITIVE = "false_positive"
CASE_CONTRADICTED_ABSTENTION = "contradicted_abstention"
CASE_BLOCKLIST_TYPE = "blocklist_type_claim"

BLOCKLIST_TYPE_CIRCULARITY = (
    "命中 160055（假投資博弈網站清單）的樣本帶一個**外部類型宣稱**"
    "（`ScamType.FAKE_INVESTMENT`），是三份黑名單裡唯一的資料集層級類型宣稱。"
    "**但它 MUST NOT 被算成任何比率**：用 `url_blocklist` 的輸出當標籤、"
    "再去量一個含 `url_blocklist` 的系統，結果恆為完美。樣本數小、有偏、且循環，"
    "三者任一都足以否決一個數字。因此這裡只有逐則清單。"
)


@dataclass(frozen=True)
class SubsetReport:
    """一個子集（或一個分組）的完整表現。

    **三個比率欄位皆為必填。** 以關鍵字省略 `abstention_rate` 是 `TypeError`，
    而那正是要的行為 —— 誤判率與棄權率封裝在同一個型別裡，缺一則建構失敗。

    `false_positive_rate` 與 `recall` 為 `None` 的唯一合法情形是該側沒有樣本
    （純 ham 的子集沒有召回率，純 scam 的子集沒有誤判率），且由
    `__post_init__` 與樣本數交叉驗證 —— 「這一側是空的」與「忘了算」因此可區分。
    """

    name: str
    ham_count: int
    scam_count: int
    abstention_rate: Rate
    false_positive_rate: Rate | None
    recall: Rate | None

    def __post_init__(self) -> None:
        if (self.false_positive_rate is None) != (self.ham_count == 0):
            raise ValueError(
                f"子集 {self.name!r} 的誤判率與 ham 樣本數不一致："
                f"ham {self.ham_count} 則、誤判率 {self.false_positive_rate!r}"
            )
        if (self.recall is None) != (self.scam_count == 0):
            raise ValueError(
                f"子集 {self.name!r} 的召回率與 scam 樣本數不一致："
                f"scam {self.scam_count} 則、召回率 {self.recall!r}"
            )

    def to_markdown(self) -> str:
        """固定四欄的一列。**沒有參數可以關掉誤判率或棄權率。**"""
        false_positive = "—" if self.false_positive_rate is None else str(self.false_positive_rate)
        recall = "—" if self.recall is None else str(self.recall)
        return (
            f"| {self.name} | {self.ham_count} | {self.scam_count} | "
            f"{false_positive} | {recall} | {self.abstention_rate} |"
        )


MARKDOWN_HEADER = (
    "| 子集 | ham n | scam n | 誤判率（95% Wilson） | 召回率（95% Wilson） | 棄權率 |\n"
    "|---|---|---|---|---|---|"
)


@dataclass(frozen=True)
class TypeShare:
    """一個 `ScamType` 的預測占比與 165 官方分項統計占比。

    **這是弱證據，不得被描述為準確率。** 分布不同有兩種解釋：系統偏向某幾類，
    或 Cofacts 的樣本與 165 的報案分布不同。後者幾乎確定成立（Cofacts 偏
    LINE 轉傳，165 是全國報案）。因此它只能用來**發現異常**
    （某一類完全沒被預測過），不能用來宣稱任何正確率。
    """

    scam_type: str
    case_titles: tuple[str, ...]
    predicted: int
    predicted_share: float
    official_cases: int
    official_share: float


@dataclass(frozen=True)
class MessageCountReport:
    """單則與多則情境分開報告。`sufficient` 為假時**不輸出任何多則比率**。"""

    single: SubsetReport
    multi_count: int
    sufficient: bool
    multi: SubsetReport | None

    def __post_init__(self) -> None:
        if self.sufficient != (self.multi_count >= MULTI_MESSAGE_MINIMUM):
            raise ValueError(
                f"多則樣本數 {self.multi_count} 與 sufficient={self.sufficient} 不一致"
            )
        if not self.sufficient and self.multi is not None:
            raise ValueError("多則樣本不足時 MUST NOT 輸出任何多則比率")


@dataclass(frozen=True)
class CaseStudy:
    """一則逐則呈現的案例。依據與建議取原文，因為誤判的傷害在文案裡看得最清楚。"""

    kind: str
    id: str
    subset: str
    split: str
    text: str
    score: float
    confidence: float
    probability: float | None
    scam_type: str | None
    hit_signals: tuple[str, ...]
    evidence: tuple[str, ...]
    actions: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    """一次評估的全部數字。

    **沒有 `pooled_false_positive_rate` 或任何等價屬性。**
    需要合併三個 ham 子集的人要自己把三個 `Rate` 拿出來相加，而那一行看得見。
    """

    run_id: RunId
    split: str
    subsets: Mapping[str, SubsetReport]
    missing_subsets: tuple[str, ...]
    type_output_rate: Rate | None
    type_marginal_comparison: tuple[TypeShare, ...]
    type_stability: tuple[tuple[str, Rate], ...]
    by_blocklist_hit: Mapping[str, SubsetReport]
    by_script: Mapping[str, SubsetReport]
    by_message_count: MessageCountReport
    quotation_veto_rate: Rate
    quotation_without_expensive_rate: Rate
    no_clause_sentence_rate: Rate
    signals: tuple[SignalEstimate, ...]
    sweep_points: tuple[SweepPoint, ...]
    calibration: Calibration
    case_studies: tuple[CaseStudy, ...]
    blocklist_type_cases: tuple[CaseStudy, ...]
    notes: tuple[str, ...] = field(default=())

    @property
    def headline_false_positive_rate(self) -> tuple[str, Rate]:
        """三個 ham 子集中 95% Wilson 上界最大的那一個，連同它的子集名稱。

        取最差而不是取平均，理由是合併會讓樣本數大的容易子集淹掉樣本數小的
        困難子集 —— Cofacts 的一千餘則負例中沒有一則是真實的銀行或物流通知。
        """
        candidates = [
            (name, report.false_positive_rate)
            for name, report in self.subsets.items()
            if name in HAM_SUBSETS and report.false_positive_rate is not None
        ]
        if not candidates:
            raise ValueError(f"沒有任何 ham 子集有誤判率可報。缺少的子集：{self.missing_subsets}")
        return max(candidates, key=_upper_of)


def _upper_of(entry: tuple[str, Rate]) -> float:
    return entry[1].upper


def _split_records(records: Sequence[RunRecord], split: str) -> tuple[RunRecord, ...]:
    return tuple(record for record in records if record.split == split)


def _rated(records: Sequence[RunRecord]) -> tuple[RunRecord, ...]:
    """把 `case_only` 的樣本排除。它們 MUST NOT 進入任何比率的分子或分母。"""
    return tuple(record for record in records if record.label != LABEL_CASE_ONLY)


def subset_report(name: str, records: Sequence[RunRecord]) -> SubsetReport:
    """一組樣本的誤判率、召回率與棄權率。`case_only` 已於呼叫端排除。"""
    if not records:
        raise ValueError(f"{name!r} 沒有樣本：比率無法在空集合上計算")
    ham = [record for record in records if record.label == LABEL_HAM]
    scam = [record for record in records if record.label == LABEL_SCAM]
    return SubsetReport(
        name=name,
        ham_count=len(ham),
        scam_count=len(scam),
        abstention_rate=Rate(
            numerator=sum(1 for record in records if record.abstained),
            denominator=len(records),
        ),
        false_positive_rate=(
            None
            if not ham
            else Rate(numerator=sum(1 for record in ham if record.decided), denominator=len(ham))
        ),
        recall=(
            None
            if not scam
            else Rate(numerator=sum(1 for record in scam if record.decided), denominator=len(scam))
        ),
    )


def _official_cases(table: WeightTable) -> dict[str, int]:
    """165 各 `CaseTitle` 的件數。

    `WeightTable` 只保留 `type_priority` 的**順序**，把件數丟掉了，
    所以這裡直接讀同一份 TOML。重複讀一次檔案比在 `scam_guard/` 加一個
    只有評估會用的欄位便宜。
    """
    with table.path.open("rb") as handle:
        document = tomllib.load(handle)
    cases: dict[str, int] = {}
    for entry in document["type_priority"]:
        cases[entry["name"]] = int(entry["cases"])
    return cases


def type_marginal_comparison(
    records: Sequence[RunRecord], table: WeightTable
) -> tuple[TypeShare, ...]:
    """預測類型的邊際分布 vs. 165 官方分項統計。

    合併成員（`INSTALLMENT_CANCEL`、`ORDER_ANOMALY`）以 `case_titles()` 回傳的
    多個 `CaseTitle` **相加**後比對 —— 這回答了 `add-scam-type` 留下的
    Open Question。理由：視為「預測不到的細分」會讓那兩類永遠計為錯，
    而系統本來就被設計成不區分買家與賣家，把一個設計決定計為錯誤沒有意義。
    """
    official = _official_cases(table)
    decided = [record for record in records if record.decided and record.scam_type is not None]
    total_official = sum(official.values())
    shares: list[TypeShare] = []
    for scam_type in ScamType:
        predicted = sum(1 for record in decided if record.scam_type == scam_type.name)
        titles = case_titles(scam_type)
        shares.append(
            TypeShare(
                scam_type=scam_type.name,
                case_titles=titles,
                predicted=predicted,
                predicted_share=predicted / len(decided) if decided else 0.0,
                official_cases=official[scam_type.name],
                official_share=official[scam_type.name] / total_official,
            )
        )
    return tuple(shares)


def _by_flag(records: Sequence[RunRecord], name: str, flag: bool) -> SubsetReport | None:
    """符合條件的那一組。**空組回 `None`，不回一個分母為零的假比率。**"""
    subset = [record for record in records if record.blocklist_hit is flag]
    if not subset:
        return None
    return subset_report(name, subset)


def _quotation_veto_rate(records: Sequence[RunRecord]) -> Rate:
    """因引述而未短路的比例：命中引述**且**存在硬證據的樣本占全體的比例。

    分母是全體而不是「有硬證據的樣本」：交辦的原文是「統計因引述而未短路的
    比例」，而讀報告的人要知道的是這件事在整批訊息裡多常發生。
    """
    return Rate(
        numerator=sum(1 for record in records if record.quotation_hit and record.hard_signals),
        denominator=len(records),
    )


def _quotation_without_expensive_rate(records: Sequence[RunRecord]) -> Rate:
    """引述命中、但**沒有昂貴檢查可跑**的比例 —— 此時短路否決沒有實際效果。

    純規則模式下 `Stage.EXPENSIVE` 是空的（`domain_age` 預設不註冊），
    所以這個比例等於引述命中率本身。這不是一個 bug，是一個要寫進報告的事實：
    引述否決短路這個機制在目前的組裝下**什麼都沒有省下來也什麼都沒有換到**，
    它真正的作用要等 LLM 層落地才會出現。
    """
    return Rate(
        numerator=sum(1 for record in records if record.quotation_hit),
        denominator=len(records),
    )


def _no_clause_sentence_rate(records: Sequence[RunRecord]) -> Rate:
    """未切出子句的言語行為命中占全部言語行為命中的比例。

    分母是**言語行為規則的命中筆數**，因為只有那些規則做子句切分；
    把 URL 或規避訊號算進分母會稀釋掉這個數字要回答的問題
    （`add-speech-act-rules` 問的是否定範疇有多常涵蓋全句）。
    """
    denominator = sum(record.speech_act_hits for record in records)
    if denominator == 0:
        raise ValueError("沒有任何言語行為規則命中，未切出子句的比例無法計算")
    return Rate(numerator=sum(record.no_clause_hits for record in records), denominator=denominator)


def _case_studies(records: Sequence[RunRecord], texts: Mapping[str, str]) -> tuple[CaseStudy, ...]:
    """holdout 上全部誤判與全部矛盾拒答的逐則輸出。

    `add-verdict-render` 交辦的是「依據的可查證性與誤判時的文案傷害以**案例**
    呈現而非數字」—— 因此這裡帶原文、`evidence` 與 `actions` 的原句。
    """
    cases: list[CaseStudy] = []
    for record in records:
        if record.label == LABEL_HAM and record.decided:
            kind = CASE_FALSE_POSITIVE
        elif record.contradicted and record.abstained:
            kind = CASE_CONTRADICTED_ABSTENTION
        else:
            continue
        cases.append(
            CaseStudy(
                kind=kind,
                id=record.id,
                subset=record.subset,
                split=record.split,
                text=texts[record.id],
                score=record.score,
                confidence=record.confidence,
                probability=record.probability,
                scam_type=record.scam_type,
                hit_signals=record.hit_signals,
                evidence=record.evidence,
                actions=record.actions,
            )
        )
    return tuple(cases)


def _blocklist_type_cases(
    records: Sequence[RunRecord], texts: Mapping[str, str]
) -> tuple[CaseStudy, ...]:
    """命中黑名單且系統給出類型的樣本，逐則呈現。見 `BLOCKLIST_TYPE_CIRCULARITY`。"""
    return tuple(
        CaseStudy(
            kind=CASE_BLOCKLIST_TYPE,
            id=record.id,
            subset=record.subset,
            split=record.split,
            text=texts[record.id],
            score=record.score,
            confidence=record.confidence,
            probability=record.probability,
            scam_type=record.scam_type,
            hit_signals=record.hit_signals,
            evidence=record.evidence,
            actions=record.actions,
        )
        for record in records
        if record.blocklist_hit and record.scam_type is not None
    )


def build_report(
    records: Sequence[RunRecord],
    texts: Mapping[str, str],
    table: WeightTable,
    run_id: RunId,
    *,
    split: str = HOLDOUT,
    missing_subsets: Sequence[str] = (),
    type_stability: Sequence[tuple[str, Rate]] = (),
    notes: Sequence[str] = (),
) -> Report:
    """算出一個切分上的全部數字。檔案的寫出由 `render_report()` 負責。"""
    selected = _rated(_split_records(records, split))
    if not selected:
        raise ValueError(f"切分 {split!r} 上沒有可計入比率的樣本")
    decided = [record for record in selected if record.decided]
    multi_names = {subset.name for subset in MULTI_MESSAGE_SUBSETS}
    multi = [record for record in selected if record.subset in multi_names]
    single = [record for record in selected if record.subset not in multi_names]
    simplified = [record for record in selected if record.simplified]
    traditional = [record for record in selected if not record.simplified]

    by_script: dict[str, SubsetReport] = {}
    if simplified:
        by_script["simplified"] = subset_report("simplified", simplified)
    if traditional:
        by_script["traditional"] = subset_report("traditional", traditional)

    by_blocklist: dict[str, SubsetReport] = {}
    for name, flag in (("blocklist_hit", True), ("blocklist_miss", False)):
        report = _by_flag(selected, name, flag)
        if report is not None:
            by_blocklist[name] = report

    return Report(
        run_id=run_id,
        split=split,
        subsets={
            name: subset_report(name, [r for r in selected if r.subset == name])
            for name in sorted({record.subset for record in selected})
        },
        missing_subsets=tuple(missing_subsets),
        type_output_rate=(
            None
            if not decided
            else Rate(
                numerator=sum(1 for record in decided if record.scam_type is not None),
                denominator=len(decided),
            )
        ),
        type_marginal_comparison=type_marginal_comparison(selected, table),
        type_stability=tuple(type_stability),
        by_blocklist_hit=by_blocklist,
        by_script=by_script,
        by_message_count=MessageCountReport(
            single=subset_report("single_message", single),
            multi_count=len(multi),
            sufficient=len(multi) >= MULTI_MESSAGE_MINIMUM,
            multi=(
                subset_report("multi_message", multi)
                if len(multi) >= MULTI_MESSAGE_MINIMUM
                else None
            ),
        ),
        quotation_veto_rate=_quotation_veto_rate(selected),
        quotation_without_expensive_rate=_quotation_without_expensive_rate(selected),
        no_clause_sentence_rate=_no_clause_sentence_rate(selected),
        signals=estimate_signals(sorted(table.signals), selected),
        sweep_points=sweep(selected, table, DEFAULT_DECISION_SCORES),
        calibration=calibrate(selected),
        case_studies=_case_studies(selected, texts),
        blocklist_type_cases=_blocklist_type_cases(selected, texts),
        notes=tuple(notes),
    )


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _rates_rows(report: Report) -> list[list[str]]:
    rows: list[list[str]] = []
    groups: list[tuple[str, Mapping[str, SubsetReport]]] = [
        ("subset", report.subsets),
        ("blocklist", report.by_blocklist_hit),
        ("script", report.by_script),
    ]
    for kind, mapping in groups:
        for name, subset in mapping.items():
            for metric, rate in (
                ("false_positive_rate", subset.false_positive_rate),
                ("recall", subset.recall),
                ("abstention_rate", subset.abstention_rate),
            ):
                if rate is None:
                    continue
                rows.append(
                    [
                        kind,
                        name,
                        metric,
                        str(rate.numerator),
                        str(rate.denominator),
                        f"{rate.value:.6f}",
                        f"{rate.lower:.6f}",
                        f"{rate.upper:.6f}",
                    ]
                )
    return rows


def _signals_rows(report: Report) -> list[list[str]]:
    return [
        [
            estimate.name,
            str(estimate.p_hit_given_scam.numerator),
            str(estimate.p_hit_given_scam.denominator),
            f"{estimate.p_hit_given_scam.value:.6f}",
            f"{estimate.p_hit_given_scam.upper:.6f}",
            str(estimate.p_hit_given_ham.numerator),
            str(estimate.p_hit_given_ham.denominator),
            f"{estimate.p_hit_given_ham.value:.6f}",
            f"{estimate.p_hit_given_ham.upper:.6f}",
            estimate.estimability,
            "" if estimate.weight_lower_bound is None else f"{estimate.weight_lower_bound:.4f}",
        ]
        for estimate in report.signals
    ]


def _headline_lines(report: Report) -> list[str]:
    name, rate = report.headline_false_positive_rate
    lines = [
        f"# 評估報告（{report.split}）",
        "",
        f"`run_id`：`{report.run_id}`",
        "",
        "## 頭條",
        "",
        f"- **誤判率（取三個 ham 子集中 Wilson 上界最大者）：{name} {rate}**",
        f"- 棄權率：{report.subsets[name].abstention_rate}",
        "",
        "誤判率與棄權率必須一起看。一個永遠回答「無法判定」的系統誤判率是 0 ——",
        "本報告的產出器不可能產出一個沒有棄權率的誤判率。",
        "",
    ]
    if report.missing_subsets:
        lines += [
            "> ⚠️ 下列子集**未蒐集**，其誤判率未被量測，而頭條數字 MUST NOT 被當成",
            "> 它們的代理值：" + "、".join(report.missing_subsets) + "。",
            "",
        ]
    return lines


def _subset_lines(report: Report) -> list[str]:
    lines = ["## 各子集", "", MARKDOWN_HEADER]
    lines += [subset.to_markdown() for subset in report.subsets.values()]
    lines += [
        "",
        "**MUST NOT 把多個 ham 子集合併成一個誤判率。** 合併會讓樣本數大的容易",
        "子集淹掉樣本數小的困難子集，而本報告的型別裡沒有那個欄位。",
        "",
        "## 分組",
        "",
        MARKDOWN_HEADER,
    ]
    lines += [subset.to_markdown() for subset in report.by_blocklist_hit.values()]
    lines += [subset.to_markdown() for subset in report.by_script.values()]
    lines += [report.by_message_count.single.to_markdown()]
    if report.by_message_count.multi is not None:
        lines.append(report.by_message_count.multi.to_markdown())
    else:
        lines += [
            "",
            f"多則情境樣本 {report.by_message_count.multi_count} 則 < "
            f"{MULTI_MESSAGE_MINIMUM} 則，**任何多則指標一律不報**，只列逐則輸出。",
        ]
    return lines + [""]


def _type_lines(report: Report) -> list[str]:
    lines = [
        "## 類型判定",
        "",
        "**本節不含 macro-F1、準確率、precision 或 recall。** 測試集不含類型欄位",
        "（Cofacts 的標籤不含 165 案類），類型判定**沒有可對照的正確答案**。",
        "`規劃.md` M9 的「類型分類的 macro-F1 與逐類表現」無法履行。",
        "",
        "- 類型產出率（被判為詐騙的樣本中 `scam_type` 非 None 的比例）："
        + (
            "沒有任何樣本被判為詐騙，此比率未定義"
            if report.type_output_rate is None
            else str(report.type_output_rate)
        ),
        "",
        "### 與 165 官方分項統計的邊際分布比較",
        "",
        "這是**弱證據**。分布不同有兩種解釋：系統偏向某幾類，或 Cofacts 的樣本與",
        "165 的報案分布不同 —— 後者幾乎確定成立。只能用來發現異常",
        "（某一類完全沒被預測過），不能用來宣稱任何正確率。",
        "",
        "| ScamType | 165 CaseTitle | 預測數 | 預測占比 | 165 件數 | 165 占比 |",
        "|---|---|---|---|---|---|",
    ]
    for share in report.type_marginal_comparison:
        lines.append(
            f"| {share.scam_type} | {'、'.join(share.case_titles)} | {share.predicted} | "
            f"{share.predicted_share:.2%} | {share.official_cases} | {share.official_share:.2%} |"
        )
    if report.type_stability:
        lines += ["", "### 類型穩定性（不同消融設定下 `scam_type` 是否改變）", ""]
        lines += [f"- {name}：{rate}" for name, rate in report.type_stability]
    return lines + [""]


def _handoff_lines(report: Report) -> list[str]:
    return [
        "## 上游交辦",
        "",
        f"- `add-quotation-check`：因引述而未短路的比例 {report.quotation_veto_rate}",
        f"- 同上（純規則模式）：引述命中但無昂貴檢查可跑 {report.quotation_without_expensive_rate}"
        "，此情形下短路否決**沒有實際效果**",
        f"- `add-speech-act-rules`：未切出子句的命中比例 {report.no_clause_sentence_rate}",
        "- `add-blocklist-store`：黑名單命中與其餘樣本已於「分組」一節分開報告",
        "- `add-normalize-text`：繁簡子集已於「分組」一節分開報告",
        "- `add-type-resolve`：單則與多則情境的**類型 F1 無法履行**，改報產出率；"
        "多則樣本不足時連產出率也不報",
        "- `add-scam-type`：兩個合併成員以 `case_titles()` 的多個 CaseTitle 相加後對齊",
        "- `add-verdict-render`：誤判與矛盾拒答的逐則文案見 `cases.md`",
        "",
    ]


def _dev_comparison_lines(report: Report) -> list[str]:
    """`add-url-check` 的 dev 樣本結論 vs. 本次 holdout 的重測。"""
    estimates = {estimate.name: estimate for estimate in report.signals}
    lines = [
        "## URL 訊號：開發樣本 vs. holdout",
        "",
        "左欄是 `add-url-check` 在**開發樣本**上的結論，而開發樣本已經被用來修過"
        "門檻（`pse.is`、`lin.ee`、`play.google.com` 三個誤判類別都是在那 1,199 則",
        "上實測出來並據以改門檻的）——**那一側的數字是調過之後的**。",
        "右欄是本次在未被逐則檢視的 holdout 上重測的結果。",
        "",
        "| 訊號 | 開發樣本（調過之後） | holdout P(命中｜詐騙) | holdout P(命中｜合法) | 可估性 |",
        "|---|---|---|---|---|",
    ]
    for name, dev in DEV_SAMPLE_MEASUREMENTS.items():
        estimate = estimates[name]
        lines.append(
            f"| {name} | {dev} | {estimate.p_hit_given_scam} | "
            f"{estimate.p_hit_given_ham} | {estimate.estimability} |"
        )
    return lines + [""]


def _calibration_lines(report: Report) -> list[str]:
    lines = ["## 校準", "", report.calibration.note, ""]
    if not report.calibration.bins:
        return lines + [
            "**校準未量測。** 這一節不會消失 —— `add-verdict-render` 的"
            "「文案不得宣稱校準」這條禁令的解除條件就是這一節，",
            "讓它消失等於讓那條 requirement 失去它的對照物。",
            "",
        ]
    lines += ["| 機率區間 | n | 預測平均 | 實際詐騙比例（95% Wilson） |", "|---|---|---|---|"]
    for calibration_bin in report.calibration.bins:
        lines.append(
            f"| {calibration_bin.lower_probability:.4f}–"
            f"{calibration_bin.upper_probability:.4f} | "
            f"{calibration_bin.observed.denominator} | "
            f"{calibration_bin.mean_predicted:.4f} | {calibration_bin.observed} |"
        )
    return lines + [""]


def _sample_size_lines() -> list[str]:
    lines = [
        "## 樣本量與可宣稱上界（零誤判）",
        "",
        "| n | Wilson 95% 上界 | rule of three（3/n） |",
        "|---|---|---|",
    ]
    for n, wilson, rule_of_three in sample_size_table():
        lines.append(f"| {n} | {wilson:.2%} | {rule_of_three:.2%} |")
    return lines + ["", RULE_OF_THREE_NOTE, ""]


def _bullets(lines: Sequence[str]) -> list[str]:
    """把一串文案渲染成項目符號。空串回「（無）」而不是一個看不出是空的空白。"""
    if not lines:
        return ["  （無）"]
    return [f"  - {line}" for line in lines]


def _cases_markdown(report: Report) -> str:
    lines = [
        f"# 逐則案例（{report.split}）",
        "",
        f"`run_id`：`{report.run_id}`",
        "",
        "兩類：**誤判**（ham 樣本被判為詐騙）與**矛盾拒答**（硬證據與引述同時命中"
        "而降級為無法判定）。兩者都以原文、依據與建議的原句呈現 ——",
        "誤判的傷害在文案裡看得最清楚，一個比率看不出來。",
        "",
        f"第三類 `{CASE_BLOCKLIST_TYPE}` 是黑名單命中樣本的類型對照。",
        BLOCKLIST_TYPE_CIRCULARITY,
        "",
    ]
    for case in report.case_studies + report.blocklist_type_cases:
        lines += [
            f"## {case.kind}｜{case.id}（{case.subset} / {case.split}）",
            "",
            f"- 分數 {case.score:.2f}、信心 {case.confidence:.2f}、機率 {case.probability}",
            f"- 類型：{case.scam_type}",
            f"- 命中訊號：{'、'.join(case.hit_signals) or '（無）'}",
            "",
            "```",
            case.text,
            "```",
            "",
            "依據：",
            *_bullets(case.evidence),
            "",
            "建議：",
            *_bullets(case.actions),
            "",
        ]
    return "\n".join(lines) + "\n"


def render_report(
    records: Sequence[RunRecord],
    texts: Mapping[str, str],
    table: WeightTable,
    run_id: RunId,
    out_dir: Path,
    *,
    split: str = HOLDOUT,
    missing_subsets: Sequence[str] = (),
    type_stability: Sequence[tuple[str, Rate]] = (),
    notes: Sequence[str] = (),
) -> Report:
    """**全系統唯一的報告入口。** 產出五個檔案並回傳 `Report`。

    不繪圖：畫圖需要 matplotlib，那會是 `tools/` 的第一個非標準庫依賴，
    而報告是人寫的，人拿 CSV 畫出來的圖會比一個預設樣式的圖好。
    **數字的正確性是本模組的責任，圖的美觀不是。**
    """
    report = build_report(
        records,
        texts,
        table,
        run_id,
        split=split,
        missing_subsets=missing_subsets,
        type_stability=type_stability,
        notes=notes,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = _headline_lines(report)
    lines += _subset_lines(report)
    lines += _type_lines(report)
    lines += _handoff_lines(report)
    lines += _dev_comparison_lines(report)
    lines += _calibration_lines(report)
    lines += _sample_size_lines()
    if report.notes:
        lines += ["## 附記", "", *[f"- {note}" for note in report.notes], ""]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "cases.md").write_text(_cases_markdown(report), encoding="utf-8")
    _write_csv(
        out_dir / "rates.csv",
        ("kind", "name", "metric", "numerator", "denominator", "value", "lower", "upper"),
        _rates_rows(report),
    )
    _write_csv(
        out_dir / "sweep.csv",
        SWEEP_COLUMNS,
        [point.as_row() for point in report.sweep_points],
    )
    _write_csv(
        out_dir / "signals.csv",
        (
            "signal",
            "scam_hits",
            "scam_n",
            "p_hit_given_scam",
            "p_scam_upper",
            "ham_hits",
            "ham_n",
            "p_hit_given_ham",
            "p_ham_upper",
            "estimability",
            "weight_lower_bound",
        ),
        _signals_rows(report),
    )
    return report


SPLITS: tuple[str, ...] = (TUNE, HOLDOUT)
"""報告 MUST 同時列出兩個切分上的每一個指標 —— 兩者的差距就是過擬合的量。

一句「holdout 未被逐則檢視」沒有任何人能驗證；兩組並列的數字可以。
"""
