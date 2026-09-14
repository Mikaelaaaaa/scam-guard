"""逐訊號的兩個條件機率與它們可不可估。

權重的尺度是對數似然比 `w = ln( P(命中|詐騙) / P(命中|合法) )`，所以要把一條
`basis = "placeholder"` 的權重改成 `measured`，需要的正是這兩個機率。

**ham 側零命中時不做平滑。** Laplace 平滑會產生一個數字，而那個數字是**先驗的
函數**不是資料的函數 —— 把 `+1/+2` 加進去之後算出來的 `w` 寫進
`basis = "measured"` 的條目，就成了一份看起來像實測結果的東西，
而那正是 `add-weight-table` 的允許集合機制要防的事。

替代做法給的資訊更多：報告該訊號 `p_hit_given_ham` 的 95% Wilson 上界，
由此得到權重的一個**下界**：

    p_ham ≤ upper(0, n_ham)  ⟹  w = ln(p_scam / p_ham) ≥ ln(p_scam / upper)

例：某訊號在 600 則 scam 中命中 120 則（`p_scam = 0.20`）、在 400 則 ham 中
零命中（`upper = 0.95%`）→ `w ≥ ln(0.20 / 0.0095) = 3.05`。
這是一個**可以陳述、不編造**的結果：「這條訊號的對數似然比至少是 3.05，
而上界不可估」。
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tools.eval.run import RunRecord
from tools.eval.selectors import LABEL_HAM, LABEL_SCAM
from tools.eval.stats import Rate, wilson_upper

ESTIMABLE = "estimable"
LOWER_BOUND_ONLY = "lower_bound_only"
NOT_ESTIMABLE = "not_estimable"

ESTIMABILITY_MEANING: Mapping[str, str] = {
    ESTIMABLE: '兩側皆有命中 —— 可改為 basis = "measured"',
    LOWER_BOUND_ONLY: "scam 側有命中、ham 側零命中 —— 維持 placeholder，報告附權重下界",
    NOT_ESTIMABLE: "scam 側亦零命中 —— 維持 placeholder；該訊號在此語料上沒有證據",
}


@dataclass(frozen=True)
class SignalEstimate:
    """一個訊號在一份語料上的兩個條件機率與可估性。

    `weight_lower_bound` 只在 `lower_bound_only` 時有值；其餘兩態為 `None`，
    因為那兩態下「下界」這個概念不存在（可估時該報點估計，不可估時什麼都不該報）。
    """

    name: str
    p_hit_given_scam: Rate
    p_hit_given_ham: Rate
    estimability: str
    weight_lower_bound: float | None

    def __post_init__(self) -> None:
        if (self.weight_lower_bound is None) != (self.estimability != LOWER_BOUND_ONLY):
            raise ValueError(
                f"訊號 {self.name!r} 的可估性為 {self.estimability!r}，"
                f"但權重下界為 {self.weight_lower_bound!r} —— 只有 "
                f"{LOWER_BOUND_ONLY!r} 才有下界"
            )


def _estimability(scam_hits: int, ham_hits: int) -> str:
    if scam_hits == 0:
        return NOT_ESTIMABLE
    if ham_hits == 0:
        return LOWER_BOUND_ONLY
    return ESTIMABLE


def estimate_signal(name: str, records: Sequence[RunRecord]) -> SignalEstimate:
    """對一個訊號算兩個條件機率。`case_only` 的樣本已由呼叫端排除。"""
    scam = [record for record in records if record.label == LABEL_SCAM]
    ham = [record for record in records if record.label == LABEL_HAM]
    if not scam or not ham:
        raise ValueError(
            f"訊號 {name!r} 的條件機率需要兩側皆非空：scam {len(scam)} 則、ham {len(ham)} 則"
        )
    scam_hits = sum(1 for record in scam if name in record.hit_signals)
    ham_hits = sum(1 for record in ham if name in record.hit_signals)
    estimability = _estimability(scam_hits, ham_hits)
    lower_bound = None
    if estimability == LOWER_BOUND_ONLY:
        lower_bound = math.log((scam_hits / len(scam)) / wilson_upper(0, len(ham)))
    return SignalEstimate(
        name=name,
        p_hit_given_scam=Rate(numerator=scam_hits, denominator=len(scam)),
        p_hit_given_ham=Rate(numerator=ham_hits, denominator=len(ham)),
        estimability=estimability,
        weight_lower_bound=lower_bound,
    )


def estimate_signals(
    names: Sequence[str], records: Sequence[RunRecord]
) -> tuple[SignalEstimate, ...]:
    """對權重表登錄的每一個訊號各算一次。順序即傳入的順序，穩定。"""
    return tuple(estimate_signal(name, records) for name in names)
