"""從消融結果推出一份建議 —— **建議，不是變更授權**。

四個值域：

| 值 | 條件 | 意思 |
|---|---|---|
| `keep` | 關閉後召回率下降，且贏得過同命中率的隨機對照組 | 這條訊號有超出「多一次命中」之外的貢獻 |
| `remove` | 在 holdout 上完全不命中 | 它在這份語料上沒有任何證據支撐 |
| `regroup` | 有命中，但在群組內長期被別的訊號蓋過 | 取 max 使它的貢獻恆為零，分群可能不對 |
| `insufficient_data` | 有命中但與隨機對照組無法區分，或根本沒註冊 | 樣本量或權重尺度不足以下結論 |

`remove` 與 `regroup` 容易被後續工作者當成直接執行的指令。因此
`recommendations.csv` 的第一行是 `DISCLAIMER`，而不是欄位名。
"""

from dataclasses import dataclass

from tools.eval.stats import Rate

KEEP = "keep"
REMOVE = "remove"
REGROUP = "regroup"
INSUFFICIENT = "insufficient_data"

DISCLAIMER = (
    "# 本表為建議，不構成 scam_guard/tables/weights.toml 的變更授權；"
    "採納前 MUST 檢視支撐數字與樣本量。"
)

SHADOW_RATIO = 0.8
"""被蓋過的次數佔命中次數的比例，超過即建議重新分群。

**沒有依據，是一個起點。** 取 max 使群組內非最大者的貢獻永遠是零，
而「長期」要多長沒有任何資料可以決定。
"""


@dataclass(frozen=True)
class Recommendation:
    """一條建議與它的理由。理由 MUST 含支撐數字，否則採納者無從判斷。"""

    name: str
    kind: str
    verdict: str
    reason: str


def recommend(
    name: str,
    kind: str,
    *,
    registered: bool,
    hit_rate: Rate | None,
    delta_recall: float,
    shadowed_count: int,
    distinguishable_from_random: bool | None,
) -> Recommendation:
    """依上表推出建議。判定順序由強到弱，取第一個成立者。"""
    if not registered:
        return Recommendation(
            name=name,
            kind=kind,
            verdict=INSUFFICIENT,
            reason="未註冊於評估用的 registry，這一輪沒有量到任何東西",
        )
    if hit_rate is None or hit_rate.numerator == 0:
        return Recommendation(
            name=name,
            kind=kind,
            verdict=REMOVE,
            reason=(
                f"在 holdout 上完全不命中（0/{'' if hit_rate is None else hit_rate.denominator}），"
                f"這份語料上沒有任何證據支撐它"
            ),
        )
    if shadowed_count >= SHADOW_RATIO * hit_rate.numerator:
        return Recommendation(
            name=name,
            kind=kind,
            verdict=REGROUP,
            reason=(
                f"{hit_rate.numerator} 次命中中有 {shadowed_count} 次在群組內被蓋過，"
                f"取 max 使它的貢獻恆為零"
            ),
        )
    if distinguishable_from_random is False:
        return Recommendation(
            name=name,
            kind=kind,
            verdict=INSUFFICIENT,
            reason=(
                f"命中率 {hit_rate.value:.2%}，但召回貢獻的 95% 區間與同命中率的"
                f"隨機注入重疊 —— **無法與隨機注入區分**，不得宣稱為有效貢獻"
            ),
        )
    if delta_recall >= 0.0:
        return Recommendation(
            name=name,
            kind=kind,
            verdict=INSUFFICIENT,
            reason=(
                f"關閉之後召回率變化為 {delta_recall:+.4%}（未下降），"
                f"命中率 {hit_rate.value:.2%}；在 placeholder 權重下看不出貢獻"
            ),
        )
    return Recommendation(
        name=name,
        kind=kind,
        verdict=KEEP,
        reason=(
            f"關閉之後召回率下降 {abs(delta_recall):.4%}，命中率 {hit_rate.value:.2%}，"
            f"且贏過同命中率的隨機對照組"
        ),
    )


RECOMMENDATION_COLUMNS = ("name", "kind", "verdict", "reason")
