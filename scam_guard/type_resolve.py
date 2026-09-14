"""類型判定 —— 把一堆 `CheckResult` 的 `scam_types` 收斂成一個 `ScamType | None`。

三種決定：哪些候選要被移除、候選怎麼排序、兩個來源不一致時聽誰的。

**排序的三層各有依據，沒有一層是在假裝有依據：**

1. `hard=True` 優先 —— 一個基於事實的類型優先於一個基於相關性的類型
2. 權重降序 —— 權重是對數似然比，同為事實或同為推論時判別力高者優先
3. `[[type_priority]]` —— 165 各 `CaseTitle` 的件數降序，**這是唯一一個我們
   手上真的有數字的排序依據**

第三層的替代方案全部是在假裝有依據：`ScamType` 的定義順序反映的是寫程式的人
打字的順序、字典序反映的是 Unicode 碼位、`list` 出現順序反映的是檢查的註冊
順序 —— 而註冊順序由組裝層決定，也就是說同一則訊息在 CLI 與在 API 上可能得到
不同的類型。件數不是「這個類型比較重要」，它是「在台灣比較常發生」；
在沒有其他資訊時，選較常見的那一個是有依據的猜測，而不是隨機。

**類型之間沒有高低。** 「LLM 只能升不能降」只適用於 `scam_probability`，
MUST NOT 被解釋成類型上的任何順序關係 —— `ROMANCE_MARRIAGE` 不比 `FAKE_JOB`
「嚴重」，兩者之間不存在那個關係。

**類型衝突 MUST NOT 影響 `confidence`。** `confidence` 的對象是「是不是詐騙」，
不是「是哪一種」。以它承載類型的不確定性，會讓一個「確定是詐騙但不確定是哪一種」
的判定被整個拒答，而那是更差的輸出：使用者拿不到警告。完整資訊留在
`Verdict.checks`，組合它是呈現層的責任；MUST NOT 為此新增 `Verdict` 欄位。

本模組 MUST NOT import `scam_guard.pipeline` 或 `scam_guard.rules` ——
關係經營訊號的名稱由 `[roles].relationship` 取得。
"""

import operator
from collections.abc import Sequence
from dataclasses import dataclass

from scam_guard.types import CheckResult, Coord, ScamType
from scam_guard.weights import WeightTable

REFINEMENTS: frozenset[tuple[ScamType, ScamType]] = frozenset(
    {(ScamType.FAKE_INVESTMENT, ScamType.ROMANCE_INVESTMENT)}
)
"""顯式的細化關係對，形式為（較廣的成員, 細化它的成員）。目前只有一組。

這一組是 `add-scam-type` 定義的 —— 兩者在要錢那一刻的話術相同，差別在關係
怎麼建立的 —— 不是本模組發明的。**MUST NOT 由任何類型階層推導**：`ScamType`
是平的，`add-scam-type` 明確拒絕過階層化。
"""


@dataclass(frozen=True)
class TypeCandidate:
    """一個候選類型，攜帶它的來源。

    不攤平成 `set[ScamType]`：排序需要 `hard` 與 `weight`、衝突判定需要
    `from_llm`、呈現層需要 `result` 才能把類型與它的依據對起來。
    攤平的話這三件事各自要重新推導一次。
    """

    scam_type: ScamType
    result: CheckResult
    hard: bool
    weight: float
    from_llm: bool
    priority: int

    @property
    def rank(self) -> tuple[int, float, int]:
        """排序鍵，愈小愈前：硬證據優先 → 權重降序 → 件數降序。"""
        return (0 if self.hard else 1, -self.weight, self.priority)


@dataclass(frozen=True)
class TypeResolution:
    """類型判定的輸出：首選類型，與支撐它的證據座標。

    座標一起回傳而不是讓呈現層自己去找：`ROMANCE_INVESTMENT` 是兩個訊號共現的
    產物，它的依據**必須**同時指向投資話術與關係經營兩處。只指向其中一處的話，
    使用者看到「類型：假交友(投資詐財)」但依據只有投資話術，那個「交友」是從
    哪裡來的就沒有交代。
    """

    scam_type: ScamType | None
    evidence: tuple[Coord, ...]
    conflict: bool


def _priority_index(table: WeightTable) -> dict[ScamType, int]:
    return {scam_type: index for index, scam_type in enumerate(table.type_priority)}


def _candidates(
    results: Sequence[CheckResult], table: WeightTable, llm_signals: frozenset[str]
) -> list[TypeCandidate]:
    """由全部 `hit=True` 結果的 `scam_types` 聯集構成候選，同一類型只留最強者。

    同一個 `ScamType` 由多筆結果產生時只保留最強的一個候選 —— 同一個類型出現
    兩次不會讓它更可能是那個類型，理由與同群組取 max 相同。

    **一個訊號列出多個類型時不稀釋強度**：四個候選各自帶著該訊號的完整權重。
    稀釋需要一個分配模型（平均分？依件數分？），而任何一種都是編出來的 ——
    我們沒有「`prepay_to_receive` 命中時有多少比例是假中獎」這個數字。
    代價很具體：`prepay_to_receive` 單獨命中時四個候選的 `hard` 與權重完全相同，
    結果完全由件數決定，永遠輸出 `FAKE_INVESTMENT`。那是一個**有依據但沒有
    證據**的答案，`add-metrics` 的逐類 recall 會把它量出來。修正的方向是讓規則
    更具體，不是在這裡稀釋。
    """
    priority = _priority_index(table)
    strongest: dict[ScamType, TypeCandidate] = {}
    for result in results:
        if not result.hit:
            continue
        weight = table.weight_for(result.name, result.hard)
        for scam_type in result.scam_types:
            candidate = TypeCandidate(
                scam_type=scam_type,
                result=result,
                hard=result.hard,
                weight=weight,
                from_llm=result.name in llm_signals,
                priority=priority[scam_type],
            )
            if scam_type not in strongest or candidate.rank < strongest[scam_type].rank:
                strongest[scam_type] = candidate
    return sorted(strongest.values(), key=operator.attrgetter("rank"))


def _yield_phishing_link(candidates: Sequence[TypeCandidate]) -> list[TypeCandidate]:
    """`PHISHING_LINK` 在候選集合同時含它與其他成員時讓位。

    URL 層只提供**手段**（惡意連結），**託辭**的判定屬規則層 ——
    這是 `add-url-check` 禁止 URL 層輸出託辭類型那條要求的另一半。
    具體情境：一則假檢警訊息附一個 165 黑名單上的連結，兩者的 `hard` 與權重
    完全相同，落到件數排序會選 `PHISHING_LINK`（6,966 件 > 5,429 件），
    而 165 對這種案件的分類是**假檢警**。件數排序在這裡給出錯誤答案，
    需要一條在它之前的規則。

    **讓位只影響類型標籤**：那筆 `CheckResult` 仍在 `Verdict.checks` 裡、
    仍貢獻分數、仍會被列為依據。

    讓位**逐來源**套用，不套用在合併後的集合上：非 LLM 只有 `PHISHING_LINK`
    而 LLM 另有一個成員時，在合併集合上讓位會清空非 LLM 候選，
    使 LLM 的類型取代規則的類型 —— 那與「規則優先」相反。
    """
    members = {candidate.scam_type for candidate in candidates}
    if ScamType.PHISHING_LINK in members and len(members) > 1:
        return [
            candidate
            for candidate in candidates
            if candidate.scam_type is not ScamType.PHISHING_LINK
        ]
    return list(candidates)


def _top(candidates: Sequence[TypeCandidate], from_llm: bool) -> TypeCandidate | None:
    """指定來源的首選候選。候選已依 `rank` 排序，取第一個。"""
    same_source = [candidate for candidate in candidates if candidate.from_llm is from_llm]
    ranked = _yield_phishing_link(same_source)
    if not ranked:
        return None
    return ranked[0]


def is_conflict(rule_top: ScamType | None, llm_top: ScamType | None) -> bool:
    """類型衝突：兩個來源的**首選**皆存在、不同、且不構成細化關係。

    **不以「兩個來源的類型集合不相等」定義。** 一則假檢警訊息會產生
    `{PHISHING_LINK（URL 層）, FAKE_AUTHORITY（規則層）}`，而 LLM 回一個
    `FAKE_AUTHORITY` —— 兩個集合不相等，但沒有任何人在說矛盾的話。
    用集合比對，系統會在幾乎每一則多訊號的訊息上宣告衝突，
    而一個永遠成立的判定不傳達任何資訊。
    """
    if rule_top is None or llm_top is None:
        return False
    if rule_top is llm_top:
        return False
    return (rule_top, llm_top) not in REFINEMENTS and (llm_top, rule_top) not in REFINEMENTS


def _romance_investment(
    results: Sequence[CheckResult], table: WeightTable
) -> tuple[Coord, ...] | None:
    """`ROMANCE_INVESTMENT` 的合成：投資話術 + 關係經營共現時**覆寫**一般排序。

    為什麼是覆寫而不是排序：`ROMANCE_INVESTMENT` 不在任何訊號的 `scam_types` 裡，
    它根本不在候選集合中，是兩個候選**共現**的產物。要讓它參與排序就得先給它
    一個權重，而那個權重是憑空的。覆寫表達的是：這不是「哪個候選比較強」的問題，
    是「這兩個訊號合起來指向第三個類型」。

    「同一請求內」不需要額外檢查：`Document` 涵蓋請求中的全部訊息，而
    關係經營規則本身已限制在同一則訊息內，所以兩個訊號只要都命中就已經在
    同一個請求裡。

    回傳兩筆結果的座標聯集；不成立時回傳 `None`（與「成立但沒有座標」由型別區分）。

    單則轉傳時合成不成立，輸出 `FAKE_INVESTMENT` —— `add-scam-type` 已把這寫成
    requirement 並稱之為「正確的降級而非誤判」：系統說出它實際看得到的東西。
    """
    relationship_name = table.roles["relationship"]
    investment = next(
        (
            result
            for result in results
            if result.hit and ScamType.FAKE_INVESTMENT in result.scam_types
        ),
        None,
    )
    relationship = next(
        (result for result in results if result.hit and result.name == relationship_name),
        None,
    )
    if investment is None or relationship is None:
        return None
    return tuple(sorted({*investment.evidence, *relationship.evidence}))


def resolve_type(
    results: Sequence[CheckResult],
    table: WeightTable,
    llm_signals: frozenset[str] = frozenset(),
) -> TypeResolution:
    """收斂出一個類型。**不修改任何輸入。**

    `llm_signals` 是**組裝層**提供的來源標記，本模組因此不需要知道任何 LLM 檢查
    的名稱；LLM 未掛載時它是空集合，`from_llm` 恆為假。

    ```
    合成成立         ⟹ ROMANCE_INVESTMENT（覆寫一般排序）
    非 LLM 候選非空  ⟹ 採用非 LLM 首選（無論 LLM 說什麼）
    非 LLM 候選為空  ⟹ 採用 LLM 首選
    兩者皆空         ⟹ None
    ```

    「規則優先」的實際意思是 **LLM 只補位，不覆寫**。補位的價值是實質的：
    18 個成員中有好幾個的規則層生產者被標為「弱」（`GUESS_WHO`、`FAKE_CHARITY`、
    `FAKE_PARCEL`），而那些類型的話術變化大，正是 LLM 該補的地方。

    LLM **不得單獨把 `FAKE_INVESTMENT` 升級為 `ROMANCE_INVESTMENT`**：
    關係經營訊號有沒有命中，是一件規則層看得到、可以指出座標的事。讓 LLM 在
    沒有那個座標的情況下升級類型，等於把一個可驗證的判定換成一個沒有證據的判定，
    而 `Verdict.evidence` 就會缺一半。這由「非 LLM 候選非空即採用非 LLM 首選」
    與「合成只看規則層的兩個訊號」共同保證，不需要額外的分支。

    兩者皆無候選時為 `None`，MUST NOT 為任何代表「其他」的值 ——
    `ScamType` 刻意沒有那個成員。
    """
    candidates = _candidates(results, table, llm_signals)
    rule_top = _top(candidates, from_llm=False)
    llm_top = _top(candidates, from_llm=True)
    conflict = is_conflict(
        None if rule_top is None else rule_top.scam_type,
        None if llm_top is None else llm_top.scam_type,
    )

    synthesised = _romance_investment(results, table)
    if synthesised is not None:
        return TypeResolution(
            scam_type=ScamType.ROMANCE_INVESTMENT, evidence=synthesised, conflict=conflict
        )

    chosen = rule_top if rule_top is not None else llm_top
    if chosen is None:
        return TypeResolution(scam_type=None, evidence=(), conflict=conflict)
    return TypeResolution(
        scam_type=chosen.scam_type,
        evidence=tuple(chosen.result.evidence),
        conflict=conflict,
    )
