"""呈現 —— 產出 `Verdict.evidence` 與 `Verdict.actions` 兩個陣列。

`Verdict` 的六個欄位裡，使用者真正會讀的是這兩個：`scam_probability` 是一個數字、
`scam_type` 是一個標籤，而「為什麼」與「那我該怎麼辦」在這裡。

**依據是組裝出來的，不是生成出來的。**

```
依據行 = detail  [+ 一段取自 doc.raw_at(coord) 的原文片段]
```

這條規則的價值不在文風，在於它**從結構上排除了一整類錯誤**：一個會自己寫句子的
呈現層可以寫出「此網域註冊於 6 天前，屬高風險」—— 前半來自 `detail`，後半是它
自己加的判斷，而沒有任何地方會報告它。可測形式：對每一行依據，去掉原文片段之後
MUST 等於某個 `hit=True` 結果的 `detail`。

唯二的例外是 `CONTRADICTION_NOTE` 與 `TRUNCATION_NOTE`，兩句對**系統狀態**的
陳述。它們不是對訊息的推測 —— 說的是「系統分不出來」「有幾則沒納入判斷」，
不是「這則訊息是宣導文」。後者是一個沒有證據的猜測。

**原文片段取 `raw_at()` 而不是 `text_at()`**，這不是風格選擇：正規化會改變顯示
形狀（全形數字變半形），而規避痕跡只存在於原文。`evasion_split_word` 的依據若取
正規化後的文字，使用者會看到「監管帳戶」四個正常的字，而訊息裡寫的是
「監 管 帳 戶」—— 依據指的東西與使用者看到的不一樣，而那正是這條訊號存在的理由。
越界座標讓 `KeyError` 傳播，**MUST NOT 在這裡吞掉**：吞掉的話，一個算錯座標的
檢查會變成少一行依據，而少一行不會有人發現。

**本層不遮蔽個資，而這需要解釋。** `project.md` 已經寫死遮蔽點：

> 遮蔽個資後再呼叫模型保護不了任何東西 —— 未遮蔽的原文與模型在同一塊記憶體裡。
> 因此個資遮蔽只在寫入 log 前套用；log 長期保存且事後可能被人翻閱，
> 那才是真正的外流面。

同樣的推論適用於呈現：`Verdict` 回給的是**送出這則訊息的那個使用者**，他本來就
看得到原文。對他遮蔽自己的訊息沒有保護任何人，只會讓依據指向一串佔位符。
遮蔽點在 `add-redact-apply`（寫 log 之前）。這個決定寫在這裡而不是留白 ——
留白的話，往後一定有人以為此處漏了一道遮蔽。

本模組 MUST NOT import `scam_guard.pipeline` 或 `scam_guard.rules`。
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from scam_guard.confidence import truncation_cap_applies
from scam_guard.normalize import Document
from scam_guard.scoring import Score
from scam_guard.types import CheckResult
from scam_guard.weights import WeightTable

SPECULATIVE_TERMS: frozenset[str] = frozenset(
    {"可疑", "危險", "不明", "很可能", "應該是", "一定是", "肯定", "小心", "注意"}
)
"""推測性形容詞。共同性質是**它們要求使用者相信我們**。

「這個網域可疑」沒有給使用者任何他自己能判斷的東西；「這個網域 6 天前才註冊」給了。
"""

VERDICT_CLAIMS: frozenset[str] = frozenset({"是詐騙", "為詐騙", "詐騙訊息", "確定是"})
"""宣告式片語。判定住在 `scam_probability` 裡，不住在散文裡。

依據的工作是陳述系統看到了什麼，不是重複結論：不說「這是詐騙」，
而說「此訊息要求提供驗證碼；銀行不會這樣要求」。

**「詐騙」一詞本身不禁。** 「evil.com 於 2026-08 列入 165 反詐騙諮詢專線_
遭停止解析涉詐網站」含「詐」字，而它是一個可查證的引用，是依據的**範本**
而不是反例。禁的是**宣告**，不是**引述官方名稱** —— 這個區別必須寫清楚，
否則實作會用一條粗暴的規則把最好的一條依據擋掉。
"""

CALIBRATION_CLAIMS: frozenset[str] = frozenset({"校準", "準確率", "正確率", "誤判率"})
"""在評估產出校準曲線之前，機率 MUST NOT 被描述為校準過的。

這些詞出現在報告裡是對的，出現在對使用者的輸出裡就是宣稱。
"""

CONTRADICTION_NOTE = "系統無法分辨這則訊息是在實施這個手法，還是在轉述它"
"""矛盾成立時附上的一句，**對系統狀態的陳述**。

MUST NOT 寫成「這可能是一則宣導文」—— 那是對訊息的推測，而系統正是分不出來。
"""

TRUNCATION_NOTE = "前 {dropped} 則訊息因超過上限未納入判斷"
"""截斷上限生效時附上的一句，來源為 `Document.dropped_messages`。

使用者拿到的是一個被壓低的信心，而壓低它的原因不在任何 `CheckResult` 裡
（`truncated` 是 `Document` 的欄位）。不說出來，使用者看到的是一個沒有理由的低信心。
"""

QUOTE_FORMAT = "{detail}：「{raw}」"
"""依據行附上原文片段的唯一形式。去掉片段之後必須還原成 `detail`。"""

QUOTE_SEPARATOR = "：「"

HARM_ALLOWED: frozenset[str] = frozenset({"無", "可回復的延遲"})
"""建議的准入條件。

使用者的規則是「建議的動作要是即使誤判也不傷害使用者的」，而字面的「無損失」
擋不住系統自己的範例：`project.md` 寫「不要照做，撥打 165 查證」，
而「不要照做」在訊息為真時**是有代價的** —— 一件合法的事被延後了。
正確的界線是**損失可不可回復**。
"""

ANY_GROUP = "*"
"""`applies_to_groups` 中代表「全體群組」的標記。

查證動作一律附上，而那是一條**規則**不是某個群組的性質；列出當下全部群組名稱
會讓新增一個群組時忘記回來改這裡，而忘記的後果是查證動作安靜地不再附上。
"""


@dataclass(frozen=True)
class Action:
    """一則建議。

    `harm_if_genuine` 是目錄裡的一個**欄位**而不是註解：測試逐筆斷言它落在
    `HARM_ALLOWED` 裡，所以「加一個新建議」這件事必須先回答「訊息若為真，
    做這件事的代價是什麼」。

    `applies_to_groups` 是**群組**而不是 `ScamType`，兩個理由：類型可能是 `None`
    （未見型態時系統仍看到了訊號，仍該給建議）；而建議對應的是「訊息要你做什麼」，
    那是群組的語意 —— 同一個 `FAKE_AUTHORITY` 可能來自不同群組，
    而使用者要被擋下的具體動作不同。
    """

    text: str
    applies_to_groups: frozenset[str]
    harm_if_genuine: str


ACTIONS: tuple[Action, ...] = (
    Action(
        text="撥打 165 反詐騙專線查證",
        applies_to_groups=frozenset({ANY_GROUP}),
        harm_if_genuine="無",
    ),
    Action(
        text="不要提供簡訊驗證碼、金融卡密碼或網銀帳號密碼給任何人",
        applies_to_groups=frozenset({"credential_solicit", "deliver_bank_instrument"}),
        harm_if_genuine="無",
    ),
    Action(
        text="不要點訊息裡的連結，改用你自己找到的官方 App 或網址",
        applies_to_groups=frozenset({"url_reputation", "url_shortener"}),
        harm_if_genuine="無",
    ),
    Action(
        text="用你自己查到的官方客服電話回撥，不要回撥訊息裡的號碼",
        applies_to_groups=frozenset({"authority_script", "parcel_notice", "seller_verification"}),
        harm_if_genuine="無",
    ),
    Action(
        text="查證前不要依訊息指示操作",
        applies_to_groups=frozenset({ANY_GROUP}),
        harm_if_genuine="可回復的延遲",
    ),
    Action(
        text="匯款前先與家人或行員談過",
        applies_to_groups=frozenset({"advance_fee", "too_good_offer"}),
        harm_if_genuine="可回復的延遲",
    ),
)
"""封閉的建議目錄。`Verdict.actions` MUST 為其中項目文字的子集，MUST NOT 生成。

生成式的建議沒有任何機制可以檢查它的代價，而這一層的錯誤直接落在使用者身上。

**被准入條件擋下的動作，逐條記在這裡**，否則那條規則看起來是空的：

- 「立即封鎖並刪除」—— 訊息為真時資訊永久遺失，不可回復
- 「報警」—— 訊息為真時浪費公務資源，且對使用者有實際成本
- 「不要理會」—— 與「不要照做」不同，它沒有查證出口，是不可回復的忽略
- 「小心詐騙」—— 不是一個動作，無法執行
"""

_DIGIT = re.compile(r"\d")
_DOMAIN = re.compile(r"[A-Za-z0-9-]+\.[A-Za-z]{2,}")


def _has_speculative_term(line: str) -> bool:
    return any(term in line for term in SPECULATIVE_TERMS)


def _has_verdict_claim(line: str) -> bool:
    return any(claim in line for claim in VERDICT_CLAIMS)


def _is_verifiable(line: str) -> bool:
    """三者之一：含具體數字、含可查證的名稱、含一段引自原文的片段。

    三者的共同性質是**使用者可以不依賴本系統自行查核**。

    「可查證的名稱」以「含一個網域形狀的字串」近似 —— 資料集正式名稱與機關名稱
    在本專案裡都帶數字（`165`），品牌冒用的依據帶網域，所以這個近似在現有的
    `detail` 上是足夠的。它是近似而不是定義，記在這裡而不是假裝完整。
    """
    return bool(_DIGIT.search(line)) or bool(_DOMAIN.search(line)) or QUOTE_SEPARATOR in line


def _line(result: CheckResult, doc: Document) -> str:
    """一行依據：`detail`，有座標時附上該句原文。

    座標以 `doc.raw_at()` 解析，越界時讓 `KeyError` 傳播。
    """
    if not result.evidence:
        return result.detail
    return QUOTE_FORMAT.format(detail=result.detail, raw=doc.raw_at(result.evidence[0]))


def _quotation_result(results: Sequence[CheckResult], table: WeightTable) -> CheckResult | None:
    name = table.roles["quotation"]
    return next((result for result in results if result.hit and result.name == name), None)


def render_evidence(
    results: Sequence[CheckResult],
    score: Score,
    doc: Document,
    table: WeightTable,
) -> list[str]:
    """依據行，依群組貢獻降序，截至 `[thresholds].max_evidence_lines`。

    **同群組的多筆命中只呈現被計分採計的那一筆**，與取 max 一致：四條假檢警規則
    命中時列四行，會讓使用者以為有四個獨立的理由，而分數只算了一次。
    被截掉與被去重的結果仍完整保留於 `Verdict.checks`。

    排序來源是 `Score.group_contributions`，**不重算權重** —— 計分結果攜帶逐群組
    明細的理由之一就是這個。

    **完全無命中時回傳空陣列。** 對一則「明天見」說「建議撥打 165」是製造焦慮，
    而且使用者很快會學到「它對每一則都這樣說」，於是真正需要被讀的那一次也被略過。

    矛盾與截斷兩句附在最後，且**佔用行數上限**：先算它們的行數，剩下的才給依據，
    否則一則八個群組命中的矛盾訊息會把那句「系統分不出來」擠掉。
    """
    if not any(result.hit for result in results):
        return []

    tail: list[str] = []
    if score.contradicted:
        quotation = _quotation_result(results, table)
        if quotation is not None:
            tail.append(_line(quotation, doc))
        tail.append(CONTRADICTION_NOTE)
    if truncation_cap_applies(results, doc):
        tail.append(TRUNCATION_NOTE.format(dropped=doc.dropped_messages))

    budget = max(0, int(table.threshold("max_evidence_lines")) - len(tail))
    taken = [_line(contribution.taken, doc) for contribution in score.group_contributions[:budget]]
    return taken + tail


def choose_actions(
    results: Sequence[CheckResult],
    score: Score,
    verdict_abstains: bool,
    table: WeightTable,
) -> list[str]:
    """建議，依命中的群組選出，截至 `[thresholds].max_actions`。

    **拒答時只給 `harm_if_genuine == "無"` 的動作**：「可回復的延遲」那一類需要
    一個判定作為前提，而此時沒有判定。

    **完全無命中時回傳空陣列** —— 與依據同一個理由。

    回傳的是目錄中的 `text`，不做任何字串拼接或格式化。
    """
    if not any(result.hit for result in results):
        return []

    groups = [contribution.group for contribution in score.group_contributions]
    if score.contradicted:
        quotation = _quotation_result(results, table)
        if quotation is not None:
            groups.append(table.group_of(quotation.name))

    allowed = [
        action for action in ACTIONS if not verdict_abstains or action.harm_if_genuine == "無"
    ]
    ranked = [action for action in allowed if ANY_GROUP in action.applies_to_groups]
    for group in groups:
        ranked.extend(
            action
            for action in allowed
            if group in action.applies_to_groups and action not in ranked
        )
    return [action.text for action in ranked[: int(table.threshold("max_actions"))]]
