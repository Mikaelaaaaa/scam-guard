"""子集的定義 —— 標籤從哪裡來，以及每一條映射假設了什麼。

**本模組不產生任何標籤。** 標籤完全由「這一則屬於哪個子集」決定，而子集由下方
一組寫死的選取條件定義。本套件中**沒有任何接受人工標籤輸入的介面**，這一點由
`tests/test_testset_build.py` 的一條測試掃描原始碼斷言。

## Cofacts 的軸與本系統的軸不是同一個

    Cofacts 的軸：這則內容是真的還是假的
    本系統的軸：這則訊息是不是在對收件人施行詐騙

兩個軸交叉出四格，而 Cofacts 的標籤只切得開上下，切不開左右：

| | 內容為假（RUMOR） | 內容為真（NOT_RUMOR / OPINIONATED） |
|---|---|---|
| **是詐騙** | 詐騙分類 + RUMOR → 正例，對得上 | 存在但 Cofacts 標不出來 |
| **不是詐騙** | 政策謠言、健康謠言 → 不可用 | 廣告/宣導 + NOT_RUMOR → 負例，靠一個假設成立 |

`NOT_RUMOR` 與 `OPINIONATED` MUST NOT 被描述為「不是詐騙」的標籤，
只得被描述為「內容經查核為真或為個人意見」——**「內容為真」不等於「不是詐騙」**。

## 結構性缺口：Cofacts 上沒有合法的銀行、物流、電信與政府通知

沒有人會把真的「您的包裹已配送至門市」送去查核 —— 那則訊息沒有真偽可查。
而規則層最可能誤判的正是這一類：`solicit_otp` 對真實一次性密碼簡訊、
`parcel_notice` 對真實物流通知、`atm_operation` 與 `order_anomaly` 對真實銀行通知、
`secrecy_demand` 對寫著「請勿告知他人」的正當簡訊。

**用 Cofacts 量出來的誤判率，量不到最危險的那一類。** 這是 `self_sms_ham`
這個自建子集存在的唯一理由，也是誤判率 MUST 按 ham 子集分開報告的理由。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

CATEGORY_SCAM = "nD2n7nEBrIRcahlYwQoW"
"""Cofacts 分類「詐騙」。"""

CATEGORY_POLICY = "mj2n7nEBrIRcahlYdArf"
"""Cofacts 分類「優惠措施、新法規、政策宣導」。"""

CATEGORY_COMMERCIAL = "nz2o7nEBrIRcahlYBgqQ"
"""Cofacts 分類「商業廣告」。"""

LABEL_SCAM = "scam"
LABEL_HAM = "ham"
LABEL_CASE_ONLY = "case_only"
"""只作案例呈現、MUST NOT 進入任何比率的分子或分母。"""

TUNE = "tune"
HOLDOUT = "holdout"


@dataclass(frozen=True)
class Subset:
    """一個子集的完整定義。

    `cofacts_filter` 為 `None` 代表這個子集不是從 Cofacts 取得的
    （自建簡訊子集），此時 `target` 仍有意義（蒐集目標），但沒有可重現的來源。
    """

    name: str
    label: str
    target: int | None
    cofacts_filter: Mapping[str, object] | None
    provenance: str
    assumption: str


COFACTS_SCAM = Subset(
    name="cofacts_scam",
    label=LABEL_SCAM,
    target=1200,
    cofacts_filter=MappingProxyType(
        {
            "categoryIds": [CATEGORY_SCAM],
            "replyTypes": ["RUMOR"],
            "articleTypes": ["TEXT"],
        }
    ),
    provenance="Cofacts 社群查核",
    assumption=(
        "被查核為假的詐騙話術就是詐騙話術。**這條假設最強** —— 分類與查核結果兩個獨立來源同向。"
    ),
)

COFACTS_HAM_AD = Subset(
    name="cofacts_ham_ad",
    label=LABEL_HAM,
    target=800,
    cofacts_filter=MappingProxyType(
        {
            "categoryIds": [CATEGORY_POLICY, CATEGORY_COMMERCIAL],
            "replyTypes": ["NOT_RUMOR", "OPINIONATED"],
            "articleTypes": ["TEXT"],
        }
    ),
    provenance="Cofacts 社群查核",
    assumption=(
        "內容屬實的宣導與廣告不是詐騙。**這條是推論不是標籤** —— 查核說的是"
        "「內容為真」，不是「不是詐騙」，而一則內容屬實的廣告仍可能帶釣魚連結。"
        "因此 holdout 上每一則被判為詐騙的 ham 樣本 MUST 被人工檢視一次，"
        "區分系統誤判與標籤錯誤，兩者分開計數；標籤錯誤的留在分母裡。"
    ),
)

COFACTS_HAM_SUSPECTED = Subset(
    name="cofacts_ham_suspected",
    label=LABEL_HAM,
    target=None,
    cofacts_filter=MappingProxyType(
        {
            "categoryIds": [CATEGORY_SCAM],
            "replyTypes": ["NOT_RUMOR", "OPINIONATED"],
            "articleTypes": ["TEXT"],
        }
    ),
    provenance="Cofacts 社群查核",
    assumption=(
        "民眾覺得像詐騙、主動送查、查核為真 → 它看起來像但不是。"
        "**這條最接近真正的誤判邊界**，也最容易錯。`add-cofacts-fetch` 已實測"
        "此池為 874 則並指名由本 change 納入；`target` 為 None 表示**全取不取樣**，"
        "取樣的唯一效果是把區間變寬。"
    ),
)

SELF_SMS_HAM = Subset(
    name="self_sms_ham",
    label=LABEL_HAM,
    target=189,
    cofacts_filter=None,
    provenance="自建：真實收到的銀行/物流/電信/政府通知",
    assumption=(
        "標籤不來自任何查核流程，而來自**可陳述的收件事實** —— 本人在某個時間"
        "從某個管道收到它，且它指涉的交易確實存在。這是全部子集中唯一沒有"
        "第三方標籤的一個，也是唯一覆蓋「合法機構的制式通知」這一類的一個。"
    ),
)

SELF_SMS_SCAM = Subset(
    name="self_sms_scam",
    label=LABEL_CASE_ONLY,
    target=30,
    cofacts_filter=None,
    provenance="自建：真實收到的詐騙簡訊",
    assumption=(
        "**標籤有循環，因此不得進入任何比率。** 唯一可用的外部標籤依據是"
        "「訊息中的網址命中 165 黑名單」，而用一個檢查的輸出當標籤、再用那個標籤"
        "量同一個檢查，`url_blocklist` 的召回率恆為 1。此子集 MAY 用於案例呈現、"
        "文案檢視與冒煙測試，MUST NOT 進入召回率、誤判率、精確率或 F1。"
    ),
)

MULTI_MESSAGE_SCAM = Subset(
    name="multi_message_scam",
    label=LABEL_SCAM,
    target=None,
    cofacts_filter=None,
    provenance="Cofacts 社群查核（自詐騙池中由 line_export 旗標標出）",
    assumption=(
        "承接 `add-message-filter` 的 `line_export` 旗標。標籤與 `cofacts_scam`"
        "同一條映射，只是這些樣本是 LINE 匯出檔、可還原成多則訊息。"
    ),
)

MULTI_MESSAGE_HAM = Subset(
    name="multi_message_ham",
    label=LABEL_HAM,
    target=None,
    cofacts_filter=None,
    provenance="Cofacts 社群查核（自兩個 ham 池中由 line_export 旗標標出）",
    assumption="標籤與來源池的 ham 映射相同，見 `cofacts_ham_ad` 與 `cofacts_ham_suspected`。",
)

COFACTS_SUBSETS: tuple[Subset, ...] = (COFACTS_SCAM, COFACTS_HAM_AD, COFACTS_HAM_SUSPECTED)
"""三個由 Cofacts selector 定義、可被第三人重建的子集。"""

SELF_SUBSETS: tuple[Subset, ...] = (SELF_SMS_HAM, SELF_SMS_SCAM)
"""兩個自建子集。**可驗證但不可重現** —— 別人沒有我的手機。

可驗證：`self_sms_*.jsonl` 的 sha256 進 manifest，保證檔案在跑完實驗後未被改過。
可重現：做不到。兩者是不同的性質，報告 MUST 分開陳述。
"""

MULTI_MESSAGE_SUBSETS: tuple[Subset, ...] = (MULTI_MESSAGE_SCAM, MULTI_MESSAGE_HAM)

COFACTS_DERIVED_NAMES: tuple[str, ...] = tuple(
    subset.name for subset in COFACTS_SUBSETS + MULTI_MESSAGE_SUBSETS
)
"""由 Cofacts 取得、因此可被第三人重建的子集名稱。這些才進版控的識別字清單。"""

SUBSETS: Mapping[str, Subset] = MappingProxyType(
    {subset.name: subset for subset in COFACTS_SUBSETS + SELF_SUBSETS + MULTI_MESSAGE_SUBSETS}
)

HAM_SUBSETS: tuple[str, ...] = (
    COFACTS_HAM_AD.name,
    COFACTS_HAM_SUSPECTED.name,
    SELF_SMS_HAM.name,
)
"""三個 ham 子集。

**誤判率 MUST 分別計算並呈現，MUST NOT 合併。** 合併之後 n ≈ 1,863，
零誤判的 Wilson 上界是 0.21% —— 一個看起來非常好的數字，但它幾乎完全由
Cofacts 的一千餘則決定，而 Cofacts 裡沒有一則真實的銀行或物流通知。
合併的效果是讓容易的樣本把困難的樣本淹掉。頭條的「系統誤判率」MUST 取三者中
95% Wilson 上界最大的那一個。
"""

EXCLUDED_POOLS: Mapping[str, str] = MappingProxyType(
    {
        "policy_or_commercial_with_RUMOR": (
            "分類為政策宣導或商業廣告、但查核為 RUMOR 的訊息（`add-cofacts-fetch` "
            "實測為該池的 27%，2,506 則）。「長照 3.0 明年上路，住院看護政府出 50%」"
            "這種內容不實的政策謠言不是詐騙，但也不是正常訊息，拿它當 ham 會汙染負例。"
        ),
        "no_normal_reply": (
            "沒有任何 status 為 NORMAL 的查核回覆的訊息（`add-cofacts-fetch` 實測 "
            "1,937 則）。沒有標籤依據。"
        ),
    }
)
"""明確排除的兩個池子，各附排除理由。此處記錄是為了讓「為什麼不是更大的資料集」
有一個可查的答案，而不是一個沒有人記得的取捨。"""

SYNTH_LOGISTICS_REFUSAL = (
    "不建立由 165 案件敘述合成的 `synth_logistics` 子集，三個理由：\n"
    "1. 授權未取得。`規劃.md` §五把「165 案件敘述授權」列為使用該資料前必須"
    "解決的事項，它還沒解決；一個未解決的前置條件不該用「先做再說」繞過。\n"
    "2. 來源本身已被 LLM 汙染。該資料約 85% 疑似經大型語言模型改寫，僅 9.5% "
    "為原始警方筆錄；拿它合成訊息等於把模型的語言分布帶進測試集，而下一個 PR "
    "就要接 LLM，那會是一個看起來很好的循環結果。\n"
    "3. 有限的蒐集工時應投入 ham 方向。誤判率是本專題的強制驗收指標，漏報不是。\n"
    "代價（不粉飾）：簡訊型詐騙短通知（假物流、假通行費、假監理）在測試集中缺席，"
    "該類的召回率完全未被量測，`parcel_notice` 這條規則的 FAKE_PARCEL 因此"
    "沒有任何正例支撐。此為必須揭露的限制的第九項。"
)

NO_TYPE_STRATIFICATION = (
    "不依詐騙類型分層：Cofacts 的標籤不含 165 案類，沒有可供分層的欄位。"
    "`規劃.md` M8 的「依類型分層，覆蓋 L2 規則可判與 L3 需判讀兩組」因此"
    "**無法履行**，列為已知缺口。同理，測試集的任何檔案 MUST NOT 含類型欄位，"
    "也不得留一個恆為空的佔位欄位 —— 留了遲早有人拿 `.get()` 去讀它，"
    "然後得到一個看起來像答案的東西。"
)

NO_BALANCED_SAMPLING = (
    "不做平衡取樣：本系統的兩個主要指標是誤判率與召回率，兩者各自在 ham 與 scam "
    "上計算，都不依賴類別比例，平衡取樣的唯一效果是丟掉樣本。"
    "`add-score-compute` 把先驗設為 0 也是同一個理由。"
)

EXTERNAL_TYPE_EVIDENCE: Mapping[str, str] = MappingProxyType(
    {
        "160055": (
            "假投資(博弈)網站黑名單。`add-blocklist-store` 已確認該資料集的定義就是"
            "「假投資(博弈)網站」，對應 ScamType.FAKE_INVESTMENT，是三份黑名單裡"
            "唯一的資料集層級類型宣稱。規模小且有偏（只涵蓋有連結且已被通報的假投資），"
            "且用 url_blocklist 的輸出當標籤再量含 url_blocklist 的系統構成循環。"
        ),
        "self_sms_scam": ("自建詐騙簡訊子集的黑名單命中。同樣的循環，且該子集本來就是 case_only。"),
    }
)
"""兩個具有外部類型依據的來源與各自的規模與偏差，交由 `add-metrics` 決定用法。

記錄它們的另一個作用是擋住「Cofacts 有分類所以有類型」這個誤解：
`cofacts_scam` 的 Cofacts 分類只有一個值（「詐騙」），不細分，沒有類型資訊。
"""

NO_LOCAL_BLOCKLIST_SUPPLEMENT = (
    "回答 `add-blocklist-store` 的 Open Question：**不建立本機補充黑名單清單。**"
    "`add-score-compute` 的黑名單例外條款建立在「這是全系統唯一的第一方事實」"
    "之上，而手動加進去的網域不是第一方事實；加了之後那條例外的依據就被稀釋，"
    "而稀釋不會有任何機制報告。"
)
