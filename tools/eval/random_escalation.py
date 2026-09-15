"""隨機 escalation 對照組 —— 排除「提高分數就會提升召回」這個混淆解釋。

**問題。** 某訊號被關閉後召回率下降 3 個百分點，能不能證明這條訊號有效？
不一定。目前 33 個訊號的權重全部是 `placeholder`，只有四個值
（`2.5`、`0.6`、`0.0`、`-1.5`），任何額外命中都以相同的量提高分數 ——
召回率下降可能只反映「少了一次命中」，與這條訊號的語意判別力無關。

**做法。** 建一個**不讀取訊息文本**的假檢查，以固定機率隨機決定命中，
並且**借用被比較訊號的名稱**。借名稱而不是新增一個名稱，是因為
`WeightTable.with_overrides()` 只能改既有條目的值，**加不了新訊號**
（`weight_for()` 對未登錄的名稱拋 `KeyError`）——
design 寫的「透過 `with_overrides()` 衍生一個假訊號」在現行 API 下做不到。
借名稱達成同一件事而且更乾淨：權重、群組、`hard` 全部與被比較訊號一致，
`weights.toml` 一個字都沒改，衍生表也不需要。

**判準。** 把「真訊號在場的召回率」與「同命中率的隨機訊號在場的召回率」
兩個 `Rate` 的 95% Wilson 區間比較。重疊即標記
`distinguishable_from_random = False`，該訊號的召回貢獻 MUST NOT 被宣稱為
有效貢獻。**這不是假設檢定** —— `add-metrics` 的 Non-Goals 已經拒絕引入
顯著性檢定的機制，本模組沿用同一個立場。

**已知限制。** 這個對照組排除的是「多一次命中」的混淆，排除不了「這條規則
命中的樣本剛好都是容易分類的樣本」這種更細緻的選擇偏誤。它是一個**下界檢查**
（貢獻至少要贏過純粹湊命中），不是充分證明。
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from scam_guard.check import Check, CheckRegistry, Stage
from scam_guard.normalize import Document
from scam_guard.types import CheckResult, Request, ScamType
from tools.eval.run import RunRecord
from tools.eval.stats import Rate

DEFAULT_SEED = 20260914
"""固定種子。`add-llm-client` 已記錄「隨機輸出上無法歸因」，本模組的隨機注入
遵守同一條規則：同一次消融的多次執行結果完全相同。"""

DETAIL = "隨機對照組：不讀取訊息文本，以固定機率注入的合成命中"

CONTROL_TYPE = ScamType.PHISHING_LINK
"""對照組在 `typed=True` 時回報的類型。**哪一個成員不重要，有沒有才重要。**

`add-confidence` 的「未見型態」上限（0.35，低於地板 0.40）在「有命中但全部命中
結果的 `scam_types` 皆為空」時生效。若被比較的訊號會輸出類型而對照組不會，
對照組會因為這條上限而系統性地拒答 —— 那不是在比判別力，是在比一個被綁住手的
對照組。`typed` 由該訊號在 baseline 上**是否曾經輸出類型**決定。
"""


@dataclass(frozen=True)
class RandomEscalationCheck:
    """借用某訊號名稱的隨機命中檢查。

    命中與否由 `(seed, name, 訊息全文)` 的雜湊決定，**不由呼叫順序決定** ——
    以序號決定的話，換一個子集的迭代順序就會換一組結果，而那不是可重現。
    """

    name: str
    probability: float
    hard: bool
    typed: bool
    seed: int = DEFAULT_SEED
    stage: Stage = Stage.LOCAL

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"命中機率必須落在 [0, 1]：probability={self.probability}")

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        key = f"{self.seed}:{self.name}:{''.join(message.text for message in req.messages)}"
        if random.Random(key).random() >= self.probability:
            return []
        return [
            CheckResult(
                name=self.name,
                hit=True,
                detail=DETAIL,
                hard=self.hard,
                scam_types=[CONTROL_TYPE] if self.typed else [],
            )
        ]


def replace_check(registry: CheckRegistry, replacement: Check) -> CheckRegistry:
    """複製一份 registry，把同名的檢查換成 `replacement`。

    `CheckRegistry.register()` 對重複名稱拋 `ValueError` 而不覆蓋（那是它刻意的
    設計），所以「換掉一個檢查」只能重建一份 —— 這也保證原 registry 不被改到，
    而消融流程正需要一個乾淨的基準。
    """
    rebuilt = CheckRegistry()
    for check in registry.enabled():
        rebuilt.register(replacement if check.name == replacement.name else check)
    if replacement.name not in {check.name for check in rebuilt.enabled()}:
        raise KeyError(f"registry 中沒有名為 {replacement.name!r} 的檢查可替換")
    return rebuilt


def hit_rate(records: Sequence[RunRecord], name: str) -> Rate:
    """某訊號在一批 `RunRecord` 上的命中率 —— 隨機對照組的機率就取這個值。

    **機率固定等於被比較訊號的實際命中率，這是唯一不需要額外選擇的錨點。**
    定得太低，幾乎任何真實訊號都會贏過隨機對照，判準失去區分力；定得太高則相反。
    報告 MUST 附上這個機率的實際數值，讓讀者自行判斷判準的嚴格程度。
    """
    hits = sum(1 for record in records if name in record.hit_signals)
    return Rate(numerator=hits, denominator=len(records))
