"""比率的表示型別與 Wilson 區間 —— 評估側唯一的統計實作。

**只用標準庫。** Wilson 需要的全部東西是 `math.sqrt` 與一個常數；
混淆矩陣需要的是幾個 `sum()`。引入 numpy 或 scipy 是為了一個常數付一個依賴，
而本專案已為此拒絕過 PyYAML、`tldextract` 與拼音函式庫。

**`float` 不是合法的比率表示。** 一個沒有分母的比率無法被引用，一個沒有區間的
比率無法被判斷強度。`規劃.md` §M9 的驗收條件是「報告中不出現無區間的誤判率
宣稱」——把它寫成型別而不是寫成一條規則，理由與 `add-weight-table` 把 `basis`
寫成欄位相同：**一條靠自律維持的規則不會有任何機制報告它被違反。**

## 樣本量與可宣稱上界（零誤判）

零誤判時 Wilson 上界化簡為 `z² / (n + z²)`，是一個只依賴 n 的量：

| n | Wilson 95% 上界（x=0） | rule of three（3/n） |
|---|---|---|
| 30 | **11.35%** | 10.00% |
| 60 | 6.02% | 5.00% |
| 100 | 3.70% | 3.00% |
| 149 | **2.51%** | **2.01%** |
| 150 | 2.50% | 2.00% |
| **189** | **1.99%** | 1.59% |
| 300 | 1.26% | 1.00% |
| 400 | 0.95% | 0.75% |
| 600 | 0.64% | 0.50% |

**`規劃.md` §六、`add-speech-act-rules` 與 `add-score-compute` 寫的「149 則」
是 rule of three 的數字，不是 Wilson。** 同一段文字的前一句用 Wilson 算
`0/30 → 11.35%`（正確），後一句換了公式而沒有說。驗收門檻的原文寫的是
「95% Wilson 上界 ≤ 2%」，所以正確的樣本量是 **189**。
`RULE_OF_THREE_NOTE` 與 `tests/test_eval_stats.py` 的一條測試把這個差異釘住。

**零誤判這個前提有多關鍵：** 要宣稱 ≤ 2%，`x = 1` 需要 n = 280、`x = 2` 需要
n = 361。也就是說 189 則的自建子集只要出現**一次**誤判就不夠了。
這件事要在報告裡講，因為它決定了「≤ 2%」這個門檻的脆弱程度。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

Z_95 = 1.959963984540054
"""標準常態分布的雙側 95% 分位數（`Φ⁻¹(0.975)`）。寫成常數而不是引入 scipy。"""

TARGET_UPPER = 0.02
"""驗收門檻：hard negative 集上誤判率的 95% Wilson 上界。"""

SAMPLE_SIZES: tuple[int, ...] = (30, 60, 100, 149, 150, 189, 300, 400, 600)
"""對照表涵蓋的樣本量。149 與 189 並列是刻意的 —— 它們是同一個門檻的兩種算法。"""

RULE_OF_THREE_NOTE = (
    "rule of three（`3/n`）是零誤判時 95% 上界的一個近似，"
    "在 n 大時與 Wilson 接近，在本專題用到的 n 上**系統性地偏小**："
    "n=149 時 rule of three 給 2.01%、Wilson 給 2.51%。"
    "驗收門檻的原文寫的是「95% Wilson 上界 ≤ 2%」，因此本專題全部誤判率宣稱"
    "一律使用 Wilson，MUST NOT 使用 rule of three；要宣稱 ≤ 2% 需要 189 則"
    "而不是 149 則。"
)

ZERO_ERROR_SAMPLE_SIZE = 189
ONE_ERROR_SAMPLE_SIZE = 280
TWO_ERROR_SAMPLE_SIZE = 361
"""在 0 / 1 / 2 次誤判下，95% Wilson 上界仍 ≤ 2% 所需的最小樣本量。

三個數字並列的作用是讓讀者看得出「零誤判」這個前提有多關鍵：
189 則的子集只要出現一次誤判，可宣稱的上界就從 1.99% 跳到 2.94%。
"""


def wilson_interval(x: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 區間。

        中心 = (p̂ + z²/2n) / (1 + z²/n)
        半寬 = z·√(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n)

    選 Wilson 而不選 Wald：Wald 在 `x = 0` 時給出 `[0, 0]`，一個寬度為零的
    區間 —— 那正是本專題最常遇到的情形（多數訊號在 ham 側零命中）。
    """
    if n <= 0:
        raise ValueError(f"Wilson 區間無法在樣本量 {n} 上計算：n 必須為正整數")
    if not 0 <= x <= n:
        raise ValueError(f"命中數 {x} 不落在 [0, {n}]")
    proportion = x / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def wilson_upper(x: int, n: int, z: float = Z_95) -> float:
    """Wilson 區間的上界。對 `x = 0` 化簡為 `z² / (n + z²)`。"""
    return wilson_interval(x, n, z)[1]


@dataclass(frozen=True)
class Rate:
    """一個比率：分子、分母、數值與 95% Wilson 區間。

    `value`、`lower`、`upper` 是**推導欄位**，不由呼叫端提供 —— 提供得了就能
    提供一個與分子分母不一致的區間，而不一致的區間沒有任何地方會報告。

    **`denominator == 0` 時建構 raise**，不回傳一個數值為 0 的 `Rate`。
    分母為零代表那個子集是空的，而一個空子集上的「誤判率 0%」是本專案明文
    禁止的「把大聲的失敗換成安靜的錯答案」。
    """

    numerator: int
    denominator: int
    value: float = field(init=False)
    lower: float = field(init=False)
    upper: float = field(init=False)

    def __post_init__(self) -> None:
        if self.denominator == 0:
            raise ValueError(
                "比率無法在空樣本集上計算：denominator 為 0。"
                "一個空子集上的「誤判率 0%」是把大聲的失敗換成安靜的錯答案。"
            )
        lower, upper = wilson_interval(self.numerator, self.denominator)
        object.__setattr__(self, "value", self.numerator / self.denominator)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def __str__(self) -> str:
        return (
            f"{self.value:.2%}（{self.numerator}/{self.denominator}，"
            f"95% CI {self.lower:.2%}–{self.upper:.2%}）"
        )

    def overlaps(self, other: "Rate") -> bool:
        """兩個比率的 95% 區間是否重疊。

        `add-ablation` 以此判斷某訊號的召回貢獻能不能與同命中率的隨機注入
        區分。**這不是假設檢定** —— `add-metrics` 的 Non-Goals 已經拒絕引入
        顯著性檢定的機制（33 個訊號 × 數個指標的多重比較校正是一整套決定，
        沒有校正的檢定比沒有檢定更誤導）。區間重疊是一個保守、不需要額外
        統計機器的判斷。
        """
        return self.lower <= other.upper and other.lower <= self.upper


def sample_size_table() -> tuple[tuple[int, float, float], ...]:
    """`(n, Wilson 上界(x=0), rule of three)` 的對照表。供報告直接重印。"""
    return tuple((n, wilson_upper(0, n), 3 / n) for n in SAMPLE_SIZES)


def minimum_n_for(x: int, target_upper: float = TARGET_UPPER) -> int:
    """給定誤判次數，Wilson 上界仍不超過 `target_upper` 所需的最小樣本量。

    以遞增搜尋而不是解析解：`x > 0` 時上界是 n 的隱函數，解析解要解一個二次式，
    而它在 `x = 0` 退化 —— 兩條路徑的程式碼會比一個迴圈長。上限 100,000 是
    一個明確的失敗點，不是一個靜默的截斷。
    """
    for n in range(max(x, 1), 100_001):
        if wilson_upper(x, n) <= target_upper:
            return n
    raise ValueError(f"在 n ≤ 100000 內找不到使 {x} 次誤判的 Wilson 上界 ≤ {target_upper} 的樣本量")


def count_true(flags: Sequence[bool]) -> int:
    """`True` 的個數。存在的理由只有一個：讓分子的來源在呼叫端讀得出來。"""
    return sum(1 for flag in flags if flag)
