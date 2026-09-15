"""門檻掃描 —— 兩個獨立的閘門，三條共用 x 軸的曲線。

系統的判定由兩個作用在**不同的量**上的閘門決定：

    拒答     ⟺ confidence <  thresholds.confidence_floor
    詐騙判定 ⟺ confidence >= confidence_floor  且  score >= decision_score

所以掃一條曲線一定會漏掉一個維度。這回答了 `add-score-compute` 留下的
Open Question（「應該掃描兩者的二維網格還是各自獨立掃描」）：**是二維，
而且因為一個軸是離散的，二維並不貴。**

`add-confidence` 的信心是 `min(base, *caps)`，`base` 取自四個登錄值、
`caps` 取自三個登錄值，所以輸出落在一個**七元素集合的去重結果**上，
實際為六個值 `{0.05, 0.30, 0.35, 0.50, 0.55, 0.90}`。把 `confidence_floor`
設在任何兩個相鄰值之間，分類結果完全相同 —— **`confidence_floor` 只有五個
有意義的切點**（六個值之間的五個間隙），不是一個連續軸。

⚠️ `add-metrics` 的 tasks 4.1 只列了六個門檻鍵（漏了 `cap_truncated`），
去重後是五個值、四個間隙。本模組依 design 與 spec 的敘述讀**七個**鍵
（四個 `base_*` 加三個 `cap_*`），去重得六個值、五個切點。

**不畫 ROC。** ROC 的兩個軸都在「系統有做出判定」的樣本上計算，棄權率不在
圖上 —— 一個棄權率 90% 的設定，在剩下的 10% 上可以有一條近乎完美的 ROC 曲線。
把它畫出來等於提供一個看起來很好、但把主要問題藏起來的圖。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from scam_guard.weights import WeightTable
from tools.eval.run import RunRecord
from tools.eval.selectors import LABEL_HAM, LABEL_SCAM
from tools.eval.stats import Rate

CONFIDENCE_THRESHOLD_KEYS: tuple[str, ...] = (
    "base_hard",
    "base_multi_group",
    "base_single_group",
    "base_no_hit",
    "cap_contradiction",
    "cap_unseen_pattern",
    "cap_truncated",
)
"""信心值的全部登錄來源。`min(base, *caps)` 的輸出只可能是其中之一。"""

SWEEP_COLUMNS: tuple[str, ...] = (
    "confidence_floor",
    "decision_score",
    "false_positive_rate",
    "recall",
    "abstention_rate",
)
"""`sweep.csv` 的固定五欄。**三個比率缺一即 raise**，沒有參數可以關掉其中一欄。"""


@dataclass(frozen=True)
class SweepPoint:
    """掃描網格上的一點。三個比率皆為必填 —— 省略任何一個是 `TypeError`。

    這是把 `add-score-compute` 的「門檻掃描 MUST 同時產出誤判率與棄權率兩條
    曲線」變成一個**會失敗的檢查**，而不是一句約定。
    """

    confidence_floor: float
    decision_score: float
    false_positive_rate: Rate
    recall: Rate
    abstention_rate: Rate

    def as_row(self) -> tuple[str, ...]:
        return (
            f"{self.confidence_floor:.4f}",
            f"{self.decision_score:.4f}",
            f"{self.false_positive_rate.value:.6f}",
            f"{self.recall.value:.6f}",
            f"{self.abstention_rate.value:.6f}",
        )


def confidence_cutpoints(table: WeightTable) -> tuple[float, ...]:
    """六個離散信心值之間的五個切點，取相鄰兩值的中點。

    中點只是這個間隙的一個代表 —— 間隙內任何取值產生完全相同的分類結果，
    這正是「不是連續軸」的意思。
    """
    values = sorted({table.threshold(key) for key in CONFIDENCE_THRESHOLD_KEYS})
    return tuple((values[index] + values[index + 1]) / 2 for index in range(len(values) - 1))


def _point(records: Sequence[RunRecord], floor: float, decision_score: float) -> SweepPoint:
    ham = [record for record in records if record.label == LABEL_HAM]
    scam = [record for record in records if record.label == LABEL_SCAM]
    if not ham or not scam:
        raise ValueError(f"掃描需要兩側皆非空：ham {len(ham)} 則、scam {len(scam)} 則")
    decided_ham = sum(
        1 for record in ham if record.confidence >= floor and record.score >= decision_score
    )
    decided_scam = sum(
        1 for record in scam if record.confidence >= floor and record.score >= decision_score
    )
    abstained = sum(1 for record in records if record.confidence < floor)
    return SweepPoint(
        confidence_floor=floor,
        decision_score=decision_score,
        false_positive_rate=Rate(numerator=decided_ham, denominator=len(ham)),
        recall=Rate(numerator=decided_scam, denominator=len(scam)),
        abstention_rate=Rate(numerator=abstained, denominator=len(records)),
    )


def sweep(
    records: Sequence[RunRecord],
    table: WeightTable,
    decision_scores: Sequence[float],
) -> tuple[SweepPoint, ...]:
    """五個信心切點 × 呼叫端給的分數軸。

    掃描是一次**純算術** —— `RunRecord` 保留了 `score` 與 `confidence` 的原始值，
    所以不需要為每個切點重跑 `detect()`。重跑會讓每個切點各自看到一份可能不同的
    `Document`，而掃描要問的是「同一份判定在不同門檻下會怎樣」。
    """
    if not decision_scores:
        raise ValueError("分數軸為空：掃描需要至少一個 decision_score")
    return tuple(
        _point(records, floor, decision_score)
        for floor in confidence_cutpoints(table)
        for decision_score in decision_scores
    )
