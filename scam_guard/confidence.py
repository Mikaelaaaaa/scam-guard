"""信心值 —— 系統對這一則判定「有沒有足夠依據」的自我評估。

**信心不是機率，也不是「這個分數區間裡有多少真的是詐騙」**。後者是校準，
屬 `scam_probability` 那一側。信心 MUST NOT 被校準、MUST NOT 被描述為機率。

分數 0.5 有兩種完全不同的來源：訊號互相矛盾，與完全沒有訊號。前者有證據但
證據對立，後者什麼都沒有 —— 兩者都**沒有依據下判斷**，所以兩者信心都低。
（`規劃.md` §一把矛盾寫成「信心高」而 §M6 寫成「信心低」，本模組採後者：
信心問的是能不能下判斷，不是有多少證據。採前者的話，系統會對一則宣導文輸出
「詐騙可能性 0.5，信心高」，而使用者得到的是一個看起來很有把握的空答案。）

```
confidence = min(base, *caps)
```

**取 min 而不是加權和。** 加權和會讓三個中等條件平均成一個及格分數，
但那三個條件每一個都是「這個判定不可靠」的獨立理由。取 min 表達的是
「任何一個失格條件都足以失格」，與 `_should_short_circuit()` 用 `any()`
而非計數是同一個判斷形狀。反過來說 `base` 內部是**依序判定的四級**而不是
累加：兩個硬證據不比一個硬證據更有依據。

**信心 MUST NOT 讀取分數的大小。** 「分數高所以信心高」在直覺上成立，在語意上
完全錯誤：一則命中五條 Tier-B 的宣導文分數很高，依據卻很弱。唯一的耦合是
矛盾判定，而它讀的是計分層已經算好的一個布林值。這條以**簽章**強制 ——
`compute_confidence()` 不接收 `Score`，拿不到的東西不會被誤用。

本模組 MUST NOT import `scam_guard.pipeline`（會成環）或 `scam_guard.rules`。
"""

from collections.abc import Sequence

from scam_guard.normalize import Document
from scam_guard.types import CheckResult
from scam_guard.weights import WeightTable


def _hit_groups(results: Sequence[CheckResult], table: WeightTable) -> set[str]:
    """命中結果涵蓋的群組集合。

    用**群組數**而不是命中筆數：同一段假檢警腳本命中四條規則仍然只是一個群組、
    一件事。分群與計分取 max 用的是同一個 `WeightTable.group_of()`，兩處各自
    分群會出現「分數只算一次但信心算四次」。
    """
    return {table.group_of(result.name) for result in results if result.hit}


def _base(results: Sequence[CheckResult], table: WeightTable) -> float:
    """基準值：四級，**依序**判定，取第一個成立者。

    - **存在硬證據命中** —— `hard` 的定義是「存在一個可陳述的事實，使合法機構
      或正常使用者不可能送出這個言語行為」。有這種事實就有依據。
    - **涵蓋兩個以上群組** —— 兩個獨立群組指向同一個結論，比一個群組有依據。
    - **恰好涵蓋一個群組** —— 單一弱訊號不足以下判斷。這一級**刻意低於**門檻。
    - **完全無命中** —— `project.md`：「完全無訊號時可能性為 0.5 但信心接近 0」。

    第三級低於門檻的後果是：只命中一條 Tier-B 的訊息一律拒答，棄權率因此提高。
    選這個方向的理由是誤判率是強制驗收指標，而一條 Tier-B（「高薪日結免經驗」
    「保證獲利」）在合法的打工與理財廣告裡大量出現。
    """
    if any(result.hit and result.hard for result in results):
        return table.threshold("base_hard")
    groups = _hit_groups(results, table)
    if len(groups) > 1:
        return table.threshold("base_multi_group")
    if len(groups) == 1:
        return table.threshold("base_single_group")
    return table.threshold("base_no_hit")


def truncation_cap_applies(results: Sequence[CheckResult], doc: Document) -> bool:
    """截斷上限是否生效：前文被丟棄**且**沒有硬證據命中。

    公開而非私有，因為呈現層要用同一個條件決定要不要陳述「前 N 則未納入判斷」。
    在那裡重寫一次這個條件，兩處會在條件改動時不同步，而不同步的樣子是
    「信心被壓低但依據沒有說為什麼」—— 使用者看到一個沒有理由的低信心。
    """
    return doc.truncated and not any(result.hit and result.hard for result in results)


def _caps(
    results: Sequence[CheckResult],
    doc: Document,
    contradicted: bool,
    table: WeightTable,
) -> list[float]:
    """三個上限，各有可測定義。

    **上限一：訊號矛盾。** 定義完全委給計分層 —— 判斷矛盾需要「分數有沒有達到
    判定門檻」，而分數是那一層算的；在這裡重算一次就是第二個實作。本模組因此
    不取得引述訊號的名稱、也不檢視引述結果。

    **上限二：未見型態** —— 存在命中，且全部命中結果的 `scam_types` 皆為空。
    也就是**系統看到了東西，但說不出這是哪一種詐騙**。`規劃.md` 的原定義依賴
    一個相似度索引，而本系統沒有向量索引；這是對它最忠實的可實作替代，
    但兩者**不等價**：原定義涵蓋「像某一類但不夠像」，新定義不涵蓋。
    落在這裡的訊號：五個 `evasion_*`、`relationship_building`、`url_shortener`、
    `quotation`，四者的 spec 都明文規定 `scam_types` 為空。

    **上限三：上下文截斷且無硬證據。** 不是無條件降低 —— 最後一則說「請至 ATM
    依指示解除分期」時，被丟掉的 37 則前文不影響這個判定，硬證據是單句可指認的
    事實，本來就不依賴前文。對這種訊息降低信心，是用一個與判定無關的理由拒答。
    反過來，沒有硬證據時判定靠的是弱訊號累積與跨訊息軌跡，而軌跡的前半段正是
    被丟掉的那一段 —— 具體的受害者是 `relationship_building`，二十一條裡唯一的
    跨句規則，而自我揭露通常在對話最前面。

    **未執行的檢查不是證據。** 只看 `hit=True`：未掛載（`checks` 中無記錄）、
    執行了未命中、因短路未執行，三者都不降低信心。這同時滿足 `add-url-check`
    指名交辦的「`url_shortener` 命中時 URL 層的沉默不得被當成負面證據」——
    那四個檢查回傳空陣列即 `hit=False`，不需要任何特例。只看 `hit` 也讓本層
    **不需要讀 `detail` 字串**，而 `NOT_HIT` / `SKIPPED` 是 `pipeline` 的常數，
    本層不能 import 它。
    """
    caps = []
    if contradicted:
        caps.append(table.threshold("cap_contradiction"))
    hits = [result for result in results if result.hit]
    if hits and all(not result.scam_types for result in hits):
        caps.append(table.threshold("cap_unseen_pattern"))
    if truncation_cap_applies(results, doc):
        caps.append(table.threshold("cap_truncated"))
    return caps


def compute_confidence(
    results: Sequence[CheckResult],
    doc: Document,
    contradicted: bool,
    table: WeightTable,
) -> float:
    """信心值，落在 `[0, 1]`。

    **簽章不接收 `Score`** —— 拿不到分數就不可能讀分數大小。`contradicted` 是
    計分層算好的布林值，本層不重新判斷引述。

    值域驗證放在這裡而不是 `Verdict`：`types.py` 明寫「系統的資料載體，
    不含任何判斷邏輯」，且同一個 PR 已經有 `add-weight-table` 在改 `CheckResult`，
    兩份 delta 改同一條 requirement 會在封存時互相覆蓋。
    """
    confidence = min([_base(results, table), *_caps(results, doc, contradicted, table)])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"信心值必須落在 [0, 1]：confidence={confidence}")
    return confidence
