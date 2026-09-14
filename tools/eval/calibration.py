"""校準 —— 只在非拒答子集上，只調溫度，且很可能做不了。

`add-score-compute` 定了「機率在 `add-metrics` 產出校準曲線之前 MUST NOT 被
宣稱為校準過的」，並留了一個 Open Question：把 `sigmoid_offset` 調成正值會讓
`p` 的值域從 `[0.5, 1)` 下移，而「系統不宣稱這不是詐騙」這條語意就被校準
悄悄改掉了。

**本模組的答案：不調 `offset`，只調 `temperature`。** 那條語意不是一個可以用
校準換掉的東西 —— `add-score-compute` 的原話是「合法訊息的特徵是**沒有**詐騙
特徵，那是證據不存在，不是反向證據存在」，而下限 0 是那句話的執行。
**校準是一個統計調整，它沒有資格改變一個語意決定。**

兩個後果要講清楚：

- **可靠度圖只有右半邊。** `p ∈ [0.5, 1)`，`[0, 0.5)` 這一段永遠沒有點。
  系統結構上無法表達「這則有 20% 機率是詐騙」。
- **這是條件校準。** 拒答機制讓「拿到數字的樣本」本來就是一個被選過的子集，
  因此 MUST 標明它是在非拒答子集上的校準，且該子集的大小 MUST 與曲線一起報告
  —— 與誤判率/棄權率是同一個形狀。

非拒答子集少於 100 則時**不畫曲線**，輸出「樣本不足，校準未量測」而不是讓
這一節消失 —— `add-verdict-render` 有一條 requirement（文案不得宣稱校準）
的解除條件就是這一節。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from tools.eval.run import RunRecord
from tools.eval.selectors import LABEL_SCAM
from tools.eval.stats import Rate

MIN_BIN_SIZE = 30
"""每個 bin 的最小樣本數。**沒有依據，是一個起點，不是調過的參數。**

報告 MUST 列出實際的 bin 數與每個 bin 的 n，讓讀者自己看得出哪些點可信。
"""

MIN_SUBSET_SIZE = 100
"""非拒答子集少於此數即不產出曲線。"""

INSUFFICIENT = "樣本不足，校準未量測"


@dataclass(frozen=True)
class CalibrationBin:
    """一個 bin：機率區間、樣本數、預測平均值與實際詐騙比例（附區間）。"""

    lower_probability: float
    upper_probability: float
    mean_predicted: float
    observed: Rate


@dataclass(frozen=True)
class Calibration:
    """校準結果。`bins` 為空即代表未量測，此時 `note` 說明為什麼。

    `decided_count` 與 `total_count` 一起呈現，使「這條曲線是在哪一個被選過的
    子集上畫的」看得見。
    """

    decided_count: int
    total_count: int
    bins: tuple[CalibrationBin, ...]
    note: str


def _bin_count(size: int) -> int:
    """bin 數由樣本量推出，**MUST NOT 固定為某個數字**（例如 10）。"""
    return max(1, size // MIN_BIN_SIZE)


def calibrate(records: Sequence[RunRecord]) -> Calibration:
    """在非拒答子集上分 bin。只讀 `probability`，完全不碰任何門檻。

    分 bin 依機率排序後等量切，而不是等寬切：等寬切在一個集中於高機率端的
    分布上會產生大量空 bin，而空 bin 的 `Rate` 建構會 raise —— 那不是一個
    要用 try 繞過的錯誤，是選錯了分 bin 方式。
    """
    decided = [record for record in records if record.probability is not None]
    if len(decided) < MIN_SUBSET_SIZE:
        return Calibration(
            decided_count=len(decided),
            total_count=len(records),
            bins=(),
            note=f"{INSUFFICIENT}（非拒答子集 {len(decided)} 則 < {MIN_SUBSET_SIZE} 則）",
        )
    ordered = sorted(decided, key=_probability_of)
    count = _bin_count(len(ordered))
    size = len(ordered) // count
    bins: list[CalibrationBin] = []
    for index in range(count):
        start = index * size
        end = len(ordered) if index == count - 1 else (index + 1) * size
        chunk = ordered[start:end]
        probabilities = [_probability_of(record) for record in chunk]
        bins.append(
            CalibrationBin(
                lower_probability=probabilities[0],
                upper_probability=probabilities[-1],
                mean_predicted=sum(probabilities) / len(probabilities),
                observed=Rate(
                    numerator=sum(1 for record in chunk if record.label == LABEL_SCAM),
                    denominator=len(chunk),
                ),
            )
        )
    return Calibration(
        decided_count=len(decided),
        total_count=len(records),
        bins=tuple(bins),
        note=(
            f"條件校準：只在非拒答子集（{len(decided)}/{len(records)} 則）上計算。"
            f"機率值域為 [0.5, 1)，可靠度圖只有右半邊 —— 系統結構上無法表達"
            f"「這則有 20% 機率是詐騙」。每 bin 至少 {MIN_BIN_SIZE} 則，"
            f"bin 數由樣本量推出而非固定。"
        ),
    )


def _probability_of(record: RunRecord) -> float:
    """排序鍵。`probability` 為 `None` 的樣本已在呼叫端被濾掉。"""
    if record.probability is None:
        raise ValueError(f"樣本 {record.id!r} 已拒答，不得進入校準")
    return record.probability
