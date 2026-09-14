"""計分 —— 把一組 `CheckResult` 變成一個對數勝算分數與一個機率。

```
score = prior_log_odds + max(0, Σ_g  max{ w(r) : r ∈ 群組 g 中命中的結果 })
p     = sigmoid( (score - offset) / temperature )
```

**求和只有在權重是對數似然比時才有機率意義。** `weights.toml` 把權重定義為
`w = ln( P(命中|詐騙) / P(命中|合法) )`；在訊號條件獨立的假設下，後驗對數勝算
等於先驗對數勝算加上各訊號的 `w` 之和 —— 這就是求和。若權重是「重要性分數」
之類的東西，相加沒有意義。這件事必須寫在這裡，因為本層是唯一做加法的地方。

**同群組取 max 在補的是那個獨立假設。** 一則假檢警訊息同時命中 `safe_account`、
`atm_operation`、`secrecy_demand`，不是三個獨立證據，是一段腳本。相加會把 `ln`
加三次。取 max 是承認不獨立之後最保守的處置：只採信群組內最強的那一個。
取 max 的範圍是**群組本身**，不細分為（群組 × 證據主體）—— 一則含五個惡意連結
的訊息若得到五倍分數，那個五倍反映的是連結數量，不是詐騙的確定性。
代價是刻意的不對稱：**依據要完整，分數要保守。**

**分數的下限是 0，系統不產生「這不是詐騙」的宣稱。** 理由不是防刷，是語意：
分數為負代表「證據指向這不是詐騙」，而本系統沒有任何訊號有資格說這句話。
合法訊息的特徵是**沒有**詐騙特徵，那是證據不存在，不是反向證據存在。
代價要明講：`p` 的值域因此是 `[0.5, 1)`，系統永遠不輸出低於 0.5 的可能性。
那個需求由信心值回答 —— 低信心加上零分是「我沒有東西可判」。

**引述是兩段式，而第二段不是扣分，是放棄下判定。** 見 `compute_score()`。

**機率 MUST NOT 被描述為校準過的**，在 `add-metrics` 產出校準曲線之前。
現在的權重全部是 `basis = "placeholder"`，由它們算出的 `p` 只保證**單調**：
證據愈強、愈多群組命中，`p` 愈大。單調性是現在可以測的，校準不是。

本模組 MUST NOT import `scam_guard.pipeline`（會成環）或 `scam_guard.rules`
（引述與黑名單的名稱由 `[roles]` 取得），且 MUST NOT 引用任何 LLM 檢查的名稱：
計分只認「`hit=True` 的結果」與「表裡的條目」，新的訊號層登錄進表就自動被算進去。
單調升級（外部模型只能升不能降）表達為「該訊號在表中的權重非負」這一條**表層**
條件，而不是一個判斷檢查名稱的分支 —— 後者要維護一份名稱清單，漏一個就是漏一道
防線；非負條件在載入表時就被檢查，與檢查叫什麼名字無關。
"""

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

from scam_guard.types import CheckResult, Verdict
from scam_guard.weights import WeightTable


@dataclass(frozen=True)
class GroupContribution:
    """一個群組採計了哪一筆結果、貢獻多少，以及被 max 蓋掉的那些。

    `shadowed` 是 `add-ablation` 判斷分群對不對的唯一材料：一個群組若長期都是
    同一條規則在提供 max，其他成員可能根本不該在裡面。取 max 使群組內非最大者
    的貢獻永遠是零，而沒有任何機制會報告它 —— 除了這個欄位。
    """

    group: str
    taken: CheckResult
    weight: float
    shadowed: tuple[CheckResult, ...] = ()


@dataclass(frozen=True)
class Score:
    """計分結果。回傳結果物件而非一個浮點數，因為三個下游都需要中間值。

    `add-confidence` 要 `contradicted`，`add-verdict-render` 要知道每個群組採計了
    哪一筆（依據要依貢獻排序），`add-ablation` 要逐群組的貢獻才能做逐層分析。
    只回傳一個浮點數，這三者都得自己重算一次，而重算一次就是第二個實作。

    `probability` 的值域驗證只擋 0 與 1 兩個端點。現行表的 13 個群組每群最多
    貢獻 2.5，分數上限 32.5，`sigmoid(32.5)` 與 1.0 仍差 7.7e-15，落在 float64
    分得出來的範圍內 —— 端點在目前的表下不可能出現，出現就是表或計分錯了。
    """

    value: float
    probability: float
    group_contributions: tuple[GroupContribution, ...]
    contradicted: bool
    quotation_applied: bool

    def __post_init__(self) -> None:
        if not 0.0 < self.probability < 1.0:
            raise ValueError(
                f"機率必須落在開區間 (0, 1)：probability={self.probability}、score={self.value}"
            )


def _sigmoid(x: float) -> float:
    """數值穩定的 logistic：`x` 很大時 `exp(-x)` 下溢、`x` 很小時 `exp(x)` 溢位。

    兩式在數學上相同，各自避開自己那一側的溢位，所以對任何有限的 `x` 都不拋例外。
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def _group_contributions(
    results: Sequence[CheckResult], table: WeightTable
) -> tuple[GroupContribution, ...]:
    """把命中結果分群，每群取權重最大的一筆，其餘記入 `shadowed`。

    權重以 `table.weight_for(name, hard)` 取得；未登錄的名稱讓 `KeyError` 傳播，
    MUST NOT 以 0 權重略過 —— 一個沒有權重的訊號悄悄不計分，等於少一個證據
    而沒有任何地方會說。
    """
    grouped: dict[str, list[tuple[float, CheckResult]]] = {}
    for result in results:
        weight = table.weight_for(result.name, result.hard)
        grouped.setdefault(table.group_of(result.name), []).append((weight, result))
    contributions = []
    for group, entries in grouped.items():
        ordered = sorted(entries, key=operator.itemgetter(0), reverse=True)
        contributions.append(
            GroupContribution(
                group=group,
                taken=ordered[0][1],
                weight=ordered[0][0],
                shadowed=tuple(result for _, result in ordered[1:]),
            )
        )
    return tuple(sorted(contributions, key=operator.attrgetter("weight", "group"), reverse=True))


def compute_score(results: Sequence[CheckResult], table: WeightTable) -> Score:
    """由命中的檢查結果算出分數與機率。

    只採計 `hit=True` 的結果 —— `hit=False` 的記錄（未命中、因短路未執行）
    不影響分數。

    **引述訊號的兩段式處置：**

    ```
    無任何 hard=True 的命中  → 引述以負權重進入加總
    有 hard=True 的命中      → 引述不進入加總，改標記 contradicted
                               （硬證據全部來自黑名單精確命中時除外）
    ```

    第一段是有效的：只命中 Tier-B 的宣導文扣掉 `-1.5` 之後落到下限 0。
    第二段承認**不可區分**並選擇拒答而不是答錯 —— 宣導文與偽裝成宣導的真釣魚，
    兩者的訊號完全相同（都是 Tier-A 加引述），任何只讀這兩個布林值的規則對兩者
    必然給出同一個答案，所以「都扣」（宣導文仍被判為詐騙）與「都不扣」（更糟）
    之間沒有可用的取值。`contradicted` 由 `add-confidence` 消費，使信心低於門檻，
    輸出走 `scam_probability = None` 這條 `types.py` 已經定義好的路徑。

    **例外條款**：硬證據全部來自 `[roles].blocklist_exact` 時不標記矛盾。
    這條證據的性質不同 —— 黑名單命中是關於**世界**的事實（這個網址已經有人受害、
    已經被查證），不是關於這則訊息的言語行為是實施還是轉述的推論，所以引述訊號
    與它在邏輯上不矛盾。安全性的另一半是經驗性的（宣導文不會把仍在運作的惡意
    網址原樣貼出來），而**這一句沒有量測支撐**，`add-testset` MUST 主動找這種樣本。

    已知損失，不粉飾：純規則模式下，一則真釣魚（Tier-A、無黑名單命中）只要加一句
    「有人傳給我這個」就從「判為詐騙」降級為「無法判定」。三件事限制它的大小 ——
    降級的終點是拒答而不是判為合法、它只發生在純規則模式（引述命中會否決短路，
    訊息必然送到有機會分辨實施與轉述的那一層）、黑名單例外把最重要的一類救回來。
    """
    hits = [result for result in results if result.hit]
    quotation_name = table.roles["quotation"]
    blocklist_name = table.roles["blocklist_exact"]

    hard_hits = [result for result in hits if result.hard]
    quotation_hits = [result for result in hits if result.name == quotation_name]
    hard_only_blocklist = all(result.name == blocklist_name for result in hard_hits)

    quotation_applied = bool(quotation_hits) and not hard_hits
    contradicted = bool(quotation_hits) and bool(hard_hits) and not hard_only_blocklist

    scored = [result for result in hits if result.name != quotation_name]
    if quotation_applied:
        scored.extend(quotation_hits)

    contributions = _group_contributions(scored, table)
    total = max(0.0, sum(contribution.weight for contribution in contributions))
    value = table.threshold("prior_log_odds") + total
    probability = _sigmoid(
        (value - table.threshold("sigmoid_offset")) / table.threshold("sigmoid_temperature")
    )
    return Score(
        value=value,
        probability=probability,
        group_contributions=contributions,
        contradicted=contradicted,
        quotation_applied=quotation_applied,
    )


def is_decision(score: Score, verdict: Verdict, table: WeightTable) -> bool:
    """系統是否對這一則**做出詐騙判定**。一次詐騙判定就是 hard negative 上的一次誤判。

    `scam_probability` 為 `None`（無法判定）**不計為誤判** —— 宣導文的處置就是
    把誤判轉成拒答，若拒答也算誤判，那個處置在帳面上等於沒做。

    但這條規則可以被鑽：一個永遠回答「無法判定」的系統誤判率是 0。
    因此**誤判率 MUST NOT 在沒有棄權率的情況下被引用**（見 `abstention_rate()`）。
    """
    return verdict.scam_probability is not None and score.value >= table.threshold("decision_score")


def abstention_rate(verdicts: Sequence[Verdict]) -> float:
    """給定樣本集上 `scam_probability` 為 `None` 的比例。

    這個數字與誤判率必須一起看才有意義，就像 precision 與 recall：
    誤判率為 0 而棄權率為 1 的設定不該被當成通過。

    **棄權率不設門檻** —— 設一個沒有依據的門檻就是編數字。`add-metrics` MUST
    報告它，`add-ablation` 掃描判定門檻時 MUST 同時畫出兩條曲線。

    空樣本集拋例外：0/0 沒有意義，回 0.0 會讓一個空實驗看起來像一個好結果。
    """
    if not verdicts:
        raise ValueError("棄權率無法在空樣本集上計算：verdicts 為空")
    abstained = sum(1 for verdict in verdicts if verdict.scam_probability is None)
    return abstained / len(verdicts)
