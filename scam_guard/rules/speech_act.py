"""言語行為規則的比對引擎與規則目錄 —— 系統第一個產生硬證據的地方。

**比對的單位是言語行為，不是關鍵詞。** 索取驗證碼與提供驗證碼含有幾乎相同的詞：

    詐騙：「請把剛收到的驗證碼告訴我」          ← 索取，對方得利
    正常：「您的驗證碼是 123456，請勿告訴他人」  ← 提供，自帶碼，且含否定

兩句都含「驗證碼」與「告訴」。任何以這兩個詞共現為條件的規則，會把台灣每天發出的
每一封一次性密碼簡訊判成詐騙。判別資訊不在詞彙上，在四元組的後兩欄：

    (述語, 客體, 接收者, 要求極性)

四個分量全部要在**同一個子句**內滿足才算命中（`context` 語境詞是第五個分量，
放寬到整句 —— 見 `SpeechActRule.context`）。

**一條規則等於一個 `Check` 實例**，不是一個大檢查內含 21 個分支。
`Check` 是結構型別，帶 `name` / `stage` / `__call__` 的 dataclass 直接滿足它，
於是 `registry.disable("secrecy_demand")` 就是 `add-ablation` 的逐條開關 ——
不需要新機制，而且 `CheckRegistry.disable()` 對拼錯的名稱拋 `KeyError`，
拼錯不會讓整組實驗悄悄變成對照組。

**詞表是模組層常數，不外部化成設定檔。** 外部化的問題不在詞表，在詞表以外的
東西：極性要求、接收者集合、豁免條件、語境前提是**邏輯**不是資料，寫進 YAML
會長出一個半殘的 DSL，而本專案的 CI（`ruff check` 與 `pytest`）不看 YAML，
一個拼錯的極性欄位會安靜地變成預設值。權重是另一回事 —— 那是純量資料，
由 `add-weight-table` 的 `weights.yaml` 承接。
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from scam_guard.check import Check, CheckRegistry, Stage
from scam_guard.normalize import Document
from scam_guard.rules.clause import is_negated, split_clauses
from scam_guard.types import CheckResult, Coord, Request, ScamType

RELATIONSHIP_RULE = "relationship_building"
"""關係經營訊號的規則名稱 —— **跨 change 的名稱契約**。

`add-type-resolve` 要靠 `CheckResult.name` 認出這個訊號來合成
`ROMANCE_INVESTMENT`（命中投資話術 + 命中關係經營 → 假交友投資詐財）。
以具名常數而非字面字串跨層引用，與 `pipeline.py` 的 `QUOTATION_CHECK` 同一個模式：
改名時 import 會壞掉，而字面字串不會。
"""

HARD_WEIGHT = 2.5
"""Tier-A（硬證據）的**佔位**權重。"""

WEAK_WEIGHT = 0.6
"""Tier-B（弱訊號）與降級後的**佔位**權重。

`weight` 在此階段只有兩個值，這是刻意的。現在沒有任何資料可以說 `solicit_otp`
比 `safe_account` 重或輕 —— 測試集要到 `add-testset` 才存在。編出二十一個有差異
的數字等於憑空製造一份看起來像實測結果的東西。兩個值誠實地表達「此刻只知道分兩層」。

兩個數字取自 `tests/test_pipeline.py` 既有的 `hard_hit()` / `weak_hit()`，
沿用是為了不在 repo 裡多一組來源不明的假數字。
**承載判定資訊的欄位是 `hard`，不是 `weight`**；`add-weight-table` 落地後
規則 MUST NOT 保留硬編碼的權重。
"""

SELF_DIRECTED = frozenset(
    {
        "我",
        "本人",
        "此號碼",
        "這個號碼",
        "本簡訊",
        "回傳",
        "回覆",
        "回復",
        "私訊",
        "傳給",
        "發給",
        "提供給",
        "客服",
        "專員",
        "上面的連結",
        "下列連結",
        "以下連結",
    }
)
"""自向接收者標記 —— 要求的動作指向**發訊者控制的對象或管道**。

「回傳」「回覆」沒有明示對象，但在祈使語境中隱含「回給發訊者」，歸入自向。
這是一個判斷不是事實，寫在這裡供日後推翻。

清單同時含人稱（我、本人、客服）與**管道**（此號碼、本簡訊、上面的連結）——
接收者問的是「動作的終點由誰控制」，不是「有沒有出現人稱代名詞」。
各規則可在此基礎上補自己那個言語行為特有的終點（監管帳戶、提款機、遠端軟體），
見各規則的 `receivers`。
"""

OTHER_DIRECTED = frozenset({"任何人", "他人", "別人", "外人", "第三方", "對外"})
"""他向接收者標記 —— 受訊者被**保護**的方向，出現即不命中。

「請勿將驗證碼告知他人」是保護性語句，述語與客體都與索取驗證碼相同，
判別資訊完全落在這一組上。本模組把它套用在**全部**規則而不只 `hard=True` 的規則
（spec 只要求後者）：多擋的方向是漏報，而本專案的強制驗收指標是誤判率。
"""

HELP_CHANNELS = frozenset(
    {
        "家人",
        "家屬",
        "父母",
        "子女",
        "親友",
        "行員",
        "櫃員",
        "警察",
        "員警",
        "警方",
        "檢察官",
        "店員",
    }
)
"""被阻斷的求助管道 —— `secrecy_demand` 專用的**第三組**接收者。

這是全部規則裡最脆弱的東西：

    詐騙：「不要告訴你的家人」          ← 阻斷求助
    正常：「請勿將驗證碼告知他人」       ← 保護你

兩句的述語相同、極性相同（都是否定），判別資訊 **100%** 落在這張名詞清單上。
而 `secrecy_demand` 是 `hard=True`，命中會觸發短路、跳過 LLM，
**它的誤判沒有第二層攔截**。

因此「任何人」「他人」「別人」「第三方」MUST NOT 出現在這張清單裡 ——
收進來一個泛稱就會把每一封一次性密碼簡訊判成假檢警。
`tests/test_speech_act_rules.py` 有一條專門的迴歸測試斷言
「請勿將驗證碼告知他人」不命中 `secrecy_demand`，以及一條斷言本清單不含泛稱。

已知漏洞：「同事」「主管」「朋友」不在清單中，寫成這樣的詐騙訊息會漏報。
漏一個詞是漏報，多收一個泛稱是誤報 —— 在沒有樣本可以決定之前，選擇漏報。
"""

IMPERATIVE = frozenset({"請", "麻煩", "煩請", "務必", "速", "儘速", "盡快", "立即", "立刻", "趕快"})
"""祈使標記 —— 接收者缺席時的**降級**命中條件，不是 `hard` 的替代品。

要求子句內有自向接收者，會漏掉索取類最常見的幾種講法：

    請把驗證碼告訴我        ← 有「我」，命中
    請提供驗證碼           ← 沒有接收者，原本完全不命中
    麻煩提供一下驗證碼       ← 同上

而「請提供驗證碼」在真實詐騙訊息裡比「請把驗證碼告訴我」常見。接收者要求
壓誤判的代價，是把每一種沒有明寫接收者的索取一起排除掉了。

降級而非直接命中，是因為 `hard=True` 會觸發短路、跳過 LLM，
而合法簡訊幾乎一定滿足兩者之一：自己附上了碼，或帶否定極性的警告。
祈使標記本身不足以排除「客服打來請你唸驗證碼」這類真實誤判，
所以這條路徑一律 `hard=False`、走 Tier-B 權重，把判斷交給後面幾層。

spec 的受益者判定只約束 `hard=True` 的命中，這條路徑不違反它。
"""

CODE_EXEMPT_RULES = frozenset({"solicit_otp", "secrecy_demand"})
"""受自帶碼豁免影響的規則。

只有這兩條的理由：豁免的推論是「訊息裡已經有碼 → 發訊者知道碼 → 他在**提供**
而非**索取**」，索取者的定義性質就是他不知道那個碼。這個推論只對
**以那個碼為客體**的言語行為成立。

`secrecy_demand` 納入是因為正當的一次性密碼簡訊句尾就是一句否定的保密要求
（「請勿告知」），與本規則的極性相同，只差在接收者是泛稱還是求助管道 ——
自帶碼是它第二層的緩解。

其餘 19 條不受影響：「請匯 50000 元到監管帳戶」裡的數字不使
`safe_account` 的推論失效，匯款金額與「發訊者知不知道某個秘密」無關。
"""

SELF_CONTAINED_CODE = re.compile(r"(?<!\d)(\d{4,8})(?!\d)(?!\s*(?:元|塊|萬|NT\$|TWD))")
"""獨立的 4 至 8 位數字 —— 自帶碼豁免的判定依據。

「獨立」有兩個條件，各自對應一種攻擊或誤判：

1. **前後皆非數字**。否則「訂單802045734652」裡的任意 6 位子字串都算數。
2. **後方不緊接金額單位**。否則「請匯 50000 元到指定帳戶，順便把驗證碼給我」
   會因為 `50000` 是五位數而取得豁免 —— 那是攻擊者只要寫一個金額就能取得的豁免。
"""


def _contains_any(text: str, start: int, end: int, words: frozenset[str]) -> bool:
    """區間 `[start, end)` 內是否出現詞表中的任一詞（整個詞須落在區間內）。"""
    return any(text.find(word, start, end) != -1 for word in words)


def _predicate_positions(
    sentence: str, clause: tuple[int, int], words: frozenset[str]
) -> list[int]:
    """子句內各述語的首次出現位置，由左至右。

    取每個述語的首次出現而非全部出現：否定範疇問的是「述語之前有沒有否定詞」，
    同一個述語的第二次出現必然在第一次之後，判定結果不會更寬鬆。
    """
    start, end = clause
    positions = [sentence.find(word, start, end) for word in words]
    return sorted(position for position in positions if position != -1)


def has_self_contained_code(doc: Document, message_index: int) -> str | None:
    """整**則**訊息內是否含獨立的 4 至 8 位數字，有則回傳觸發豁免的那串數字。

    **範圍是訊息，不是句子。** 真實的一次性密碼簡訊常見兩行：
    「您的驗證碼為 482913」換行「請勿提供他人」。`split_sentences()` 把換行
    當分隔符，兩行是兩個句子 —— 以句子為範圍的豁免在這裡就失效了，
    而那正是最常見的正當簡訊形狀。
    """
    for index in doc.message_range(message_index):
        match = SELF_CONTAINED_CODE.search(doc.sentences[index])
        if match is not None:
            return match.group(1)
    return None


class Match(Enum):
    """命中的強度。`WEAK` 是接收者缺席但有祈使標記的降級路徑，見 `IMPERATIVE`。"""

    FULL = "full"
    WEAK = "weak"


class Polarity(Enum):
    """規則**要求**的極性。

    ⚠️ 極性是規則的欄位，不是命中後的取消條件。把否定範疇寫成
    「命中後再檢查有沒有否定，有就取消」，`secrecy_demand` 這條
    （要求勿告知家人）就**永遠不會命中** —— 它偵測的言語行為本身
    就是一個否定祈使句。比對時檢查的是「實際極性 == 規則要求的極性」。
    """

    REQUIRE_POSITIVE = "positive"
    REQUIRE_NEGATIVE = "negative"


def _detail(
    summary: str, fact: str, code: str | None, single_clause: bool, no_receiver: bool
) -> str:
    """組裝 `CheckResult.detail`。

    `detail` MUST 為具體的事實陳述，MUST NOT 為「話術可疑」這類形容 ——
    `Verdict.evidence` 最後要寫的是「這則訊息要求你把驗證碼傳給對方」，
    不是一個分數。硬證據另外帶出使它成為硬證據的那個可陳述事實。

    降級時 MUST 說出降級這件事與觸發降級的數字，否則 `add-ablation` 分析時
    分不出「沒命中」與「命中後被豁免」。
    """
    parts = [summary]
    if fact:
        parts.append(f"事實：{fact}")
    if code is not None:
        parts.append(f"訊息內已含 {len(code)} 位數字 {code}，自帶碼豁免成立，降級為弱訊號")
    if no_receiver:
        parts.append("子句內未出現接收者，僅憑祈使標記命中，降級為弱訊號")
    if single_clause:
        parts.append("本句未切出子句，否定範疇涵蓋全句")
    return "；".join(parts)


@dataclass(frozen=True)
class SpeechActRule:
    """一條言語行為規則，本身就是一個 `Check`。

    四元組的三個分量是詞表（`predicates`、`objects`、`receivers`），
    第四個是 `polarity`。`context` 是第五個、選用的分量。

    **分量的比對範圍不同，這是刻意的**：述語、客體、接收者、極性在**子句**內
    比對（否定範疇由標點界定）；`context` 在整**句**內比對 ——
    語境不是言語行為的一部分，它是這個言語行為周圍的環境，
    「您的獎金已核准，請先匯手續費」的「獎金」與「先匯手續費」本來就不同子句。

    空詞表代表**不設該項要求**。`hard=True` 的規則不得留空 `receivers`
    與 `fact`，由 `__post_init__` 強制。
    """

    name: str
    summary: str
    predicates: frozenset[str]
    polarity: Polarity
    objects: frozenset[str] = frozenset()
    receivers: frozenset[str] = frozenset()
    context: frozenset[str] = frozenset()
    scam_types: tuple[ScamType, ...] = ()
    hard: bool = False
    fact: str = ""
    stage: Stage = Stage.LOCAL

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"規則名稱必須為 snake_case：name={self.name!r}")
        if not self.predicates:
            raise ValueError(f"規則必須有述語詞表：name={self.name!r}")
        if self.hard and not self.receivers:
            raise ValueError(
                f"hard=True 的規則必須宣告接收者標記，否則受益者無法判定：name={self.name!r}"
            )
        if self.hard and not self.fact:
            raise ValueError(f"hard=True 的規則必須陳述使它成為硬證據的事實：name={self.name!r}")
        if ScamType.ROMANCE_INVESTMENT in self.scam_types:
            raise ValueError(
                f"ROMANCE_INVESTMENT 不得由單一規則直接輸出，須由 add-type-resolve "
                f"以投資話術 + 關係經營合成：name={self.name!r}"
            )

    def _matches_clause(self, sentence: str, clause: tuple[int, int]) -> Match | None:
        """四元組在此子句內的滿足程度。未命中回傳 `None`。

        接收者缺席但子句帶祈使標記時回傳 `Match.WEAK` —— 見 `IMPERATIVE`。
        """
        start, end = clause
        if self.objects and not _contains_any(sentence, start, end, self.objects):
            return None
        if _contains_any(sentence, start, end, OTHER_DIRECTED):
            return None
        want_negated = self.polarity is Polarity.REQUIRE_NEGATIVE
        if not any(
            is_negated(sentence, clause, position) == want_negated
            for position in _predicate_positions(sentence, clause, self.predicates)
        ):
            return None
        if not self.receivers or _contains_any(sentence, start, end, self.receivers):
            return Match.FULL
        if want_negated or not _contains_any(sentence, start, end, IMPERATIVE):
            return None
        return Match.WEAK

    def _matches(self, sentence: str) -> tuple[Match, bool] | None:
        """句子是否命中。命中時回傳（強度，本句是否只有一個子句），未命中回傳 `None`。"""
        if self.context and not _contains_any(sentence, 0, len(sentence), self.context):
            return None
        clauses = split_clauses(sentence)
        strengths = [self._matches_clause(sentence, clause) for clause in clauses]
        hit = [item for item in strengths if item is not None]
        if not hit:
            return None
        strength = Match.FULL if Match.FULL in hit else Match.WEAK
        return strength, len(clauses) <= 1

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        """逐句比對，命中的句子**依訊息分組**，每則訊息產出一筆 `CheckResult`。

        依訊息分組而非全句合併，是因為自帶碼豁免的範圍就是一則訊息 ——
        第 0 則自帶碼、第 3 則沒有時，兩則的 `hard` 本來就不同，
        合併成一筆就只能二選一。

        未命中時回傳空陣列，不回傳 `hit=False` 的佔位結果（`add-check-protocol`）。
        """
        matches: list[tuple[Coord, Match, bool]] = []
        for coord, sentence in zip(doc.coords, doc.sentences):
            matched = self._matches(sentence)
            if matched is not None:
                matches.append((coord, matched[0], matched[1]))
        if not matches:
            return []

        results: list[CheckResult] = []
        for message_index in sorted({coord[0] for coord, _, _ in matches}):
            in_message = [item for item in matches if item[0][0] == message_index]
            code = None
            if self.name in CODE_EXEMPT_RULES:
                code = has_self_contained_code(doc, message_index)
            full = any(strength is Match.FULL for _, strength, _ in in_message)
            hard = self.hard and code is None and full
            results.append(
                CheckResult(
                    name=self.name,
                    hit=True,
                    weight=HARD_WEIGHT if hard else WEAK_WEIGHT,
                    detail=_detail(
                        self.summary,
                        self.fact if self.hard else "",
                        code,
                        any(single for _, _, single in in_message),
                        not full,
                    ),
                    evidence=[coord for coord, _, _ in in_message],
                    scam_types=list(self.scam_types),
                    hard=hard,
                )
            )
        return results


SELF_DISCLOSURE = frozenset(
    {
        "我是工程師",
        "我在國外",
        "我在杜拜",
        "我在新加坡",
        "我是軍人",
        "我是醫生",
        "駐外",
        "外派",
        "我離婚",
        "我喪偶",
        "我一個人住",
        "我父母過世",
        "我自己創業",
        "我做期貨",
        "我做外匯",
        "我姑姑",
        "我舅舅",
    }
)
"""自我揭露 —— 職業、海外、婚姻狀態、家庭變故。"""

EMOTIONAL_COMMITMENT = frozenset(
    {
        "想照顧你",
        "照顧你一輩子",
        "當我女朋友",
        "當我男朋友",
        "我們結婚",
        "娶你",
        "嫁給我",
        "視訊給你看",
        "只想跟你",
        "最信任你",
        "把你當家人",
        "我愛你",
        "寶貝",
        "親愛的",
    }
)
"""情感承諾。"""

CHANNEL_SHIFT = frozenset(
    {
        "加我line",
        "加我LINE",
        "加我賴",
        "加我的line",
        "換到這裡聊",
        "私下聊",
        "不要在這個平台",
        "這個平台不方便",
        "留個電話",
        "加телеgram",
        "加我telegram",
        "加我WhatsApp",
        "我們私聊",
    }
)
"""通道轉移 —— 把對話帶離可被平台稽核的地方。"""

RELATIONSHIP_CATEGORIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("自我揭露", SELF_DISCLOSURE),
    ("情感承諾", EMOTIONAL_COMMITMENT),
    ("通道轉移", CHANNEL_SHIFT),
)


@dataclass(frozen=True)
class RelationshipRule:
    """關係經營訊號 —— 與其餘二十條**性質不同**的一條規則。

    三個差異，每一個都必須寫進來，否則會被當成寫壞的 `SpeechActRule`：

    1. **它不是一個言語行為，是三類話語的共現**（自我揭露、情感承諾、通道轉移），
       因此沒有述語／客體／接收者／極性可言。
    2. **它是跨句規則。** 其餘二十條全部在子句內完成判定；這條要求三類中
       至少兩類命中且落在**不同句子**，`evidence` 因此會有多個座標。
       跨句判定 MUST 以 `doc.message_range()` 限制在同一則訊息內 ——
       `Document` 是扁平的，不限制就會把第 0 則的自我揭露與第 3 則的情感承諾
       算成共現，而那兩則可能相隔數日。
    3. **它不輸出 `ScamType`。**「這段對話在經營關係」本身不是一種詐騙類型。
       `ROMANCE_INVESTMENT` 由 `add-type-resolve` 合成：
       命中投資話術**且**命中本規則。

    已知代價：單則轉傳時本規則幾乎不可能命中（需要跨句的兩類共現），
    所以 `ROMANCE_INVESTMENT` 在單則情境下仍然產不出來。
    那是 `add-scam-type` 已經接受的降級（單則輸出 `FAKE_INVESTMENT`），
    不是本規則的失敗。
    """

    name: str = RELATIONSHIP_RULE
    stage: Stage = Stage.LOCAL
    hard: bool = field(default=False, init=False)

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        results: list[CheckResult] = []
        for message_index in sorted({coord[0] for coord in doc.coords}):
            hit_coords = self._coords_in_message(doc, message_index)
            if hit_coords is not None:
                results.append(
                    CheckResult(
                        name=self.name,
                        hit=True,
                        weight=WEAK_WEIGHT,
                        detail="關係經營：自我揭露、情感承諾與通道轉移中的兩類出現在不同句子",
                        evidence=hit_coords,
                        scam_types=[],
                        hard=False,
                    )
                )
        return results

    def _coords_in_message(self, doc: Document, message_index: int) -> list[Coord] | None:
        """該則訊息是否有兩類落在不同句子，有則回傳全部命中句子的座標。"""
        matched: list[tuple[str, Coord]] = []
        for index in doc.message_range(message_index):
            sentence = doc.sentences[index]
            for label, words in RELATIONSHIP_CATEGORIES:
                if _contains_any(sentence, 0, len(sentence), words):
                    matched.append((label, doc.coords[index]))
        if not any(
            first[0] != second[0] and first[1] != second[1]
            for first in matched
            for second in matched
        ):
            return None
        return sorted({coord for _, coord in matched})


TIER_A_RULES: tuple[SpeechActRule, ...] = (
    SpeechActRule(
        name="solicit_otp",
        summary="索取簡訊驗證碼",
        predicates=frozenset(
            {"告訴", "告知", "提供", "傳", "回覆", "回傳", "輸入", "給", "念", "唸", "報"}
        ),
        objects=frozenset(
            {
                "驗證碼",
                "認證碼",
                "簡訊碼",
                "動態密碼",
                "一次性密碼",
                "OTP",
                "otp",
                "驗證簡訊",
                "授權碼",
            }
        ),
        receivers=SELF_DIRECTED,
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.ACCOUNT_TAKEOVER,),
        hard=True,
        fact="一次性密碼的定義就是不得轉交，任何機構都不會向本人索取它",
    ),
    SpeechActRule(
        name="solicit_card_secret",
        summary="索取信用卡號、有效期限或背面末三碼",
        predicates=frozenset({"告訴", "告知", "提供", "傳", "輸入", "填", "回覆", "給", "報"}),
        objects=frozenset(
            {"卡號", "信用卡號", "有效期限", "末三碼", "背面三碼", "安全碼", "CVV", "cvv", "CVC"}
        ),
        receivers=SELF_DIRECTED,
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.PHISHING_LINK, ScamType.FAKE_AUTHORITY),
        hard=True,
        fact="金融機構不會以訊息或來電索取卡片背面末三碼",
    ),
    SpeechActRule(
        name="solicit_bank_credentials",
        summary="索取網路銀行帳號密碼或約定轉帳設定",
        predicates=frozenset(
            {"告訴", "告知", "提供", "傳", "輸入", "填", "回覆", "給", "設定", "開通"}
        ),
        objects=frozenset(
            {
                "網銀密碼",
                "網路銀行密碼",
                "網銀帳號",
                "帳號密碼",
                "使用者代號",
                "用戶代號",
                "約定轉帳",
                "約定帳號",
                "提款卡密碼",
                "金融卡密碼",
            }
        ),
        receivers=SELF_DIRECTED,
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.BANK_ACCOUNT_HARVEST,),
        hard=True,
        fact="金融機構不會索取網路銀行密碼，也不會請客戶代為設定約定轉帳",
    ),
    SpeechActRule(
        name="deliver_bank_instrument",
        summary="要求寄送存摺、印章或提款卡",
        predicates=frozenset(
            {"寄", "寄送", "郵寄", "宅配", "交付", "交給", "送到", "面交", "提供"}
        ),
        objects=frozenset({"存摺", "印章", "提款卡", "金融卡", "晶片卡", "存摺封面", "銀行卡"}),
        receivers=SELF_DIRECTED,
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.BANK_ACCOUNT_HARVEST,),
        hard=True,
        fact="金融機構不收寄存摺與提款卡，交付它們等同交出帳戶控制權",
    ),
    SpeechActRule(
        name="atm_operation",
        summary="要求至提款機依指示操作或解除設定",
        predicates=frozenset(
            {"至", "到", "去", "前往", "操作", "按", "輸入", "依指示", "依照指示", "使用"}
        ),
        objects=frozenset({"解除", "取消", "更正", "校正", "認證", "升級", "分期", "設定", "扣款"}),
        receivers=frozenset({"ATM", "atm", "提款機", "自動櫃員機", "櫃員機"}),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.INSTALLMENT_CANCEL, ScamType.ORDER_ANOMALY),
        hard=True,
        fact="提款機沒有解除分期付款或更正訂單設定的功能，它只能匯出款項",
    ),
    SpeechActRule(
        name="safe_account",
        summary="要求把款項匯入監管、安全或公證帳戶",
        predicates=frozenset(
            {"匯", "匯入", "匯到", "匯款", "轉帳", "轉入", "存入", "存到", "繳交"}
        ),
        receivers=frozenset(
            {
                "監管帳戶",
                "安全帳戶",
                "公證帳戶",
                "保管帳戶",
                "專案帳戶",
                "監管賬戶",
                "安全賬戶",
                "檢警帳戶",
            }
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_AUTHORITY,),
        hard=True,
        fact="我國法制不存在監管帳戶或安全帳戶，檢警不會要求民眾把錢匯到任何帳戶",
    ),
    SpeechActRule(
        name="secrecy_demand",
        summary="要求不得告知家人、行員或警察",
        predicates=frozenset({"告訴", "告知", "說", "講", "透露", "提起", "通知", "報警", "求證"}),
        receivers=HELP_CHANNELS,
        polarity=Polarity.REQUIRE_NEGATIVE,
        scam_types=(ScamType.FAKE_AUTHORITY,),
        hard=True,
        fact="司法機關不會要求當事人對家人、行員或警察保密，偵查不公開拘束的是承辦人員",
    ),
    SpeechActRule(
        name="remote_control_tool",
        summary="要求安裝遠端控制軟體",
        predicates=frozenset({"安裝", "下載", "開啟", "執行", "點選", "加入", "設定"}),
        receivers=frozenset(
            {
                "AnyDesk",
                "anydesk",
                "ANYDESK",
                "TeamViewer",
                "teamviewer",
                "RustDesk",
                "AirDroid",
                "快速支援",
                "遠端協助",
                "遠端控制",
                "向日葵",
            }
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_AUTHORITY, ScamType.FAKE_LOAN),
        hard=True,
        fact="金融機構與司法機關不以遠端控制軟體處理業務，安裝後對方可直接操作你的手機",
    ),
    SpeechActRule(
        name="prepay_to_receive",
        summary="為了領取一筆錢而要求先付出一筆錢",
        predicates=frozenset({"繳", "匯", "付", "支付", "轉帳", "儲值", "先繳", "先匯", "先付"}),
        objects=frozenset(
            {
                "手續費",
                "保證金",
                "稅金",
                "稅款",
                "工本費",
                "解凍金",
                "認證金",
                "開通費",
                "服務費",
                "履約金",
                "保險金",
            }
        ),
        receivers=SELF_DIRECTED
        | frozenset({"指定帳戶", "以下帳戶", "下列帳戶", "此帳戶", "指定帳號", "以下帳號"}),
        context=frozenset(
            {"領取", "領獎", "出金", "撥款", "退款", "解凍", "提領", "放款", "中獎", "獎金"}
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(
            ScamType.FAKE_PRIZE,
            ScamType.FAKE_LOAN,
            ScamType.FAKE_INVESTMENT,
            ScamType.FAKE_JOB,
        ),
        hard=True,
        fact="領取一筆錢不需要先匯出一筆錢，合法的費用一律自應付金額中扣除",
    ),
    SpeechActRule(
        name="seller_verification",
        summary="要求賣家至連結完成認證或開通商店",
        predicates=frozenset({"完成", "進行", "辦理", "點選", "申請", "認證", "開通", "設定"}),
        objects=frozenset(
            {"賣家認證", "賣場認證", "商店認證", "開通商店", "賣家資格", "面交認證", "分期開通"}
        ),
        receivers=frozenset({"連結", "網址", "客服", "專員", "下列", "以下", "我"}),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_BUYER,),
        hard=True,
        fact="拍賣平台不存在需由買家提供連結才能完成的賣家認證或開通商店流程",
    ),
)
"""Tier-A —— `hard=True`，命中可觸發短路。

`hard=True` 的判準：**存在一個可陳述的事實，使合法機構或正常使用者不可能
送出這個言語行為**，而該事實 MUST 寫在 `fact` 裡並出現在 `detail` 中。
「違規但可能發生」不算 —— 那是 Tier-B。
"""

TIER_B_RULES: tuple[SpeechActRule, ...] = (
    SpeechActRule(
        name="guaranteed_return",
        summary="宣稱保證獲利或零風險",
        predicates=frozenset(
            {
                "保證獲利",
                "保證收益",
                "保證賺",
                "穩賺不賠",
                "穩賺",
                "零風險",
                "無風險",
                "包賺",
                "必賺",
                "穩定獲利",
                "保本保息",
                "日結獲利",
                "獲利翻倍",
                "穩定配息",
            }
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_INVESTMENT,),
    ),
    SpeechActRule(
        name="romance_pretext",
        summary="以清關費、機票或醫療費為由要錢，並承諾來台或結婚",
        predicates=frozenset({"需要", "支付", "匯", "繳", "寄", "幫我", "借我", "先幫"}),
        objects=frozenset(
            {"清關費", "關稅", "機票", "機票錢", "醫療費", "簽證費", "包裹卡關", "手續費"}
        ),
        context=frozenset(
            {"結婚", "來台", "見面", "一起生活", "未婚妻", "老婆", "娶", "嫁", "團聚", "退休"}
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.ROMANCE_MARRIAGE,),
    ),
    SpeechActRule(
        name="identity_docs",
        summary="索取身分證、健保卡或存摺封面照片",
        predicates=frozenset({"提供", "傳", "上傳", "拍", "寄", "給", "傳送", "附上"}),
        objects=frozenset(
            {
                "身分證正反面",
                "身分證照片",
                "身分證",
                "健保卡",
                "駕照",
                "存摺封面",
                "證件照",
                "雙證件",
            }
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_JOB, ScamType.FAKE_LOAN, ScamType.BANK_ACCOUNT_HARVEST),
    ),
    SpeechActRule(
        name="escort_deposit",
        summary="要求先儲值或先付訂金才派人",
        predicates=frozenset({"儲值", "加值", "先付", "先匯", "匯", "付", "刷卡", "轉帳"}),
        objects=frozenset({"誠意金", "訂金", "定金", "保證金", "車馬費", "點數", "儲值金"}),
        context=frozenset(
            {"外約", "全套", "半套", "出鐘", "應召", "茶莊", "指壓", "上門服務", "妹妹"}
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.SEXUAL_SERVICE,),
    ),
    SpeechActRule(
        name="game_code",
        summary="索取點數卡序號或指定代儲、代管站",
        predicates=frozenset({"提供", "傳", "給", "輸入", "購買", "代儲", "拍", "儲值"}),
        objects=frozenset(
            {
                "點數卡",
                "點數序號",
                "遊戲點數",
                "序號",
                "虛寶",
                "寶物",
                "代儲",
                "代管站",
                "MyCard",
                "遊戲幣",
            }
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.GAME_ITEM,),
    ),
    SpeechActRule(
        name="identity_reset",
        summary="宣稱換了號碼或帳號並要對方猜身分",
        predicates=frozenset(
            {
                "猜猜我是誰",
                "猜我是誰",
                "你猜我是誰",
                "換號碼了",
                "換了新號碼",
                "換新手機了",
                "我的新帳號",
                "我的新號碼",
                "這是我的新",
            }
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.GUESS_WHO,),
    ),
    SpeechActRule(
        name="charity_personal_account",
        summary="具名募款並指定帳戶與期限",
        predicates=frozenset({"捐款", "捐助", "贊助", "匯款", "轉帳", "匯", "小額捐"}),
        objects=frozenset(
            {"急難救助", "醫療費", "罹癌", "救助金", "善款", "重病", "孤兒院", "流浪動物"}
        ),
        receivers=frozenset({"帳戶", "帳號", "戶名", "指定帳戶", "郵局帳號"}),
        context=frozenset(
            {"今天", "今日", "最後", "截止", "期限", "只剩", "急需", "明天前", "限時"}
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_CHARITY,),
    ),
    SpeechActRule(
        name="parcel_notice",
        summary="冒稱物流通知並要求更新地址或補繳費用",
        predicates=frozenset({"更新", "填寫", "補繳", "繳納", "確認", "點選", "重新", "補填"}),
        objects=frozenset({"地址", "關稅", "運費", "收件資訊", "派送", "清關", "收件地址", "郵資"}),
        context=frozenset({"包裹", "郵局", "快遞", "貨件", "物流", "宅配", "郵政", "配送"}),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_PARCEL,),
    ),
    SpeechActRule(
        name="loan_no_check",
        summary="宣稱免聯徵、免對保或線上快速核貸",
        predicates=frozenset(
            {
                "免聯徵",
                "免對保",
                "免抵押",
                "免留車",
                "不看信用",
                "信用瑕疵可辦",
                "黑戶可辦",
                "快速核貸",
                "當日撥款",
                "秒過件",
                "免照會",
            }
        ),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_LOAN,),
    ),
    SpeechActRule(
        name="high_pay_no_skill",
        summary="宣稱高薪日結、免經驗、在家可做",
        predicates=frozenset(
            {
                "日領",
                "日結",
                "高薪",
                "免經驗",
                "無經驗",
                "輕鬆賺",
                "在家工作",
                "動動手指",
                "時間自由",
            }
        ),
        context=frozenset({"徵", "招募", "兼職", "打工", "工作", "職缺", "應徵", "求職", "人手"}),
        polarity=Polarity.REQUIRE_POSITIVE,
        scam_types=(ScamType.FAKE_JOB,),
    ),
)
"""Tier-B —— `hard=False`，命中不觸發短路。

`guaranteed_return` 落在這裡是一個會被質疑的判斷，先講清楚：保證獲利在台灣是
金管會禁止的廣告用語，真實金融機構不會這樣寫 —— 但「違規」與「不可能」是兩件事，
補習班寫「保證學會」、健身房寫「保證有效」。更關鍵的是它會被防詐宣導文大量觸發
（「有人說保證獲利，那是詐騙」）。Tier-B 讓它不觸發短路，訊息仍會送到 LLM
去判斷那是宣稱還是引述。
"""

SPEECH_ACT_RULES: tuple[Check, ...] = TIER_A_RULES + TIER_B_RULES + (RelationshipRule(),)
"""全部 21 條規則。型別為 `tuple[Check, ...]` 而非 `tuple[SpeechActRule, ...]` ——
`relationship_building` 不是言語行為規則（見 `RelationshipRule`），
它只滿足 `Check` 這個結構型別。
"""


def register_speech_act_rules(registry: CheckRegistry) -> None:
    """把 21 條規則逐條註冊進 registry。

    名稱重複時 `CheckRegistry.register()` 會拋 `ValueError`，不覆蓋 ——
    同名的兩條規則通常是誤註冊，覆蓋會靜默丟失其中一條。
    """
    for rule in SPEECH_ACT_RULES:
        registry.register(rule)
