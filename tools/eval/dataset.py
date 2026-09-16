"""測試集的資料載體、切分規則與讀寫。

**切分是確定性的，不存檔、不用隨機種子：**

    tune    ⟺ sha256(article_id.encode()).digest()[0] <  128
    holdout ⟺ sha256(article_id.encode()).digest()[0] >= 128

四個性質，每一個都是選它而不選別的方式的理由：

1. **確定性** —— 不需要存一份 split 檔，也就不會有「split 檔與資料檔不同步」
   這個失敗模式。
2. **加樣本不改變既有樣本的歸屬** —— `random.shuffle` 加 seed 做不到：池子從
   1,200 變成 1,600 時結果整個變，上一次報告的 holdout 就不再是這一次的 holdout。
3. **與標籤、長度、時間皆獨立** —— id 是 Cofacts 產生的識別字，與內容無關。
4. **50/50 而不是 80/20** —— 報告引用的是 holdout 的數字，而 holdout 的 n
   決定可宣稱的上界。

`self_sms_ham` **全部為 holdout，不切分**：它太小，切一半之後零誤判的 Wilson
上界從 1.99% 變成 2.94%；而且一個用真實通知調過門檻的系統，在真實通知上的
誤判率不再是一個獨立的量測。

**測試集的任何檔案 MUST NOT 含詐騙類型欄位**，也不得留一個恆為空的佔位欄位 ——
來源平台的標籤不含 165 案類，類型判定沒有可對照的正確答案（見
`selectors.NO_TYPE_STRATIFICATION`）。
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

from tools.eval.selectors import (
    CONVERSATION_HAM,
    HOLDOUT,
    LABEL_CASE_ONLY,
    SELF_SMS_HAM,
    SUBSETS,
    TUNE,
)

SPLIT_BOUNDARY = 128
"""`sha256(id)` 首位元組的切點。256 個值對半分，期望 50/50。"""

SELF_NOTICE_TARGET = 189
"""`self_sms_ham` 的蒐集目標。**這是一個算出來的數字，不是挑的。**

對零誤判（`x = 0`），Wilson 上界化簡為 `z² / (n + z²)`。令它不超過 2%：

    z² / (n + z²) ≤ 0.02
    n ≥ z² · (1 - 0.02) / 0.02 = 1.959963984540054² × 49 = 188.2…

取整得 **189**。

⚠️ **既有 spec 三處寫的「149 則」是 rule of three（`3/149 = 2.01%`），不是
Wilson。** 同一份文件的前一句用 Wilson 算 `0/30 → 11.35%`（正確），後一句
換成 rule of three 而沒有說。驗收門檻的原文寫的是「95% Wilson 上界 ≤ 2%」，
而 Wilson 在 `n = 149` 給的是 2.51%。本套件一律用 Wilson，
差異由 `tools/eval/stats.py` 的一條測試釘住。
"""

SELF_NOTICE_MINIMUM = 60
"""`self_sms_ham` 的最低可接受樣本數。零誤判時 95% Wilson 上界為 6.02%。

低於此數時該子集 MUST 只作案例呈現，不產出任何比率。
樣本不足 MUST 照實報告樣本數與對應的上界，
MUST NOT 以合成、重複採樣或合併其他子集補足。
"""

PROVENANCE_VALUES = ("self", "granted", "public")
"""`self_sms_ham` 的三個來源，性質不同，分開記錄。

| 值 | 內容 | 性質 |
|---|---|---|
| `self` | 本人手機的簡訊、LINE 官方帳號通知、email 通知 | 真實收到，分布真實 |
| `granted` | 明確同意提供者的手機，當面逐則檢視後抄錄 | 真實收到 |
| `public` | 業者官網防詐頁面公開張貼的通知範本 | **不是真實收到**，是範本 |

`public` MUST 以 `verbatim = false` 標記並**單獨報告** —— 範本是業者寫給人看的
標準句，比真實通知更整齊，在它上面不誤判不代表在真實通知上不誤判。
`public` 占比超過三分之一時，此子集的代表性要打折，MUST 在報告中說明。
"""

PUBLIC_SHARE_CEILING = 1 / 3
"""`public` 占比的上限。**沒有依據，是一個起點。** 超過即在報告中揭露代表性打折。"""

SELF_NOTICE_FIELDS = (
    "id",
    "text",
    "provenance",
    "verbatim",
    "consent",
    "institution",
    "collected_on",
)
"""自建通知子集每一則的欄位。

**提供者的身分 MUST NOT 寫入資料檔** —— `consent` 只記三個列舉值之一，
對應關係保存於 repo 之外的離線同意清單，供提供者要求刪除時查閱。
寫進資料檔就是把一個識別資訊放進一個會被反覆讀取的檔案。
"""

DEIDENTIFICATION_RULE = (
    "MUST 遮蔽：收件人姓名、地址、手機號碼、身分證字號、電子郵件。\n"
    "MUST NOT 遮蔽任何數字串 —— 驗證碼、訂單編號、寄件碼、金額、統一編號一律保留原樣。\n"
    "理由可驗證：規則層的自帶碼豁免以「同一則訊息內存在獨立的 4 至 8 位數字」為條件"
    "（`scam_guard.rules.speech_act.SELF_CONTAINED_CODE`），數字被遮之後豁免失效，"
    "每一封正當一次性密碼簡訊都會命中 solicit_otp 且 hard=True，"
    "使該子集的誤判率趨近於 1 —— 而那個數字量到的是去識別的錯，不是系統的錯。\n"
    "去識別 MUST 以人工執行，MUST NOT 呼叫 scam_guard.pii 或 redact_document()："
    "那些元件的遮蔽點是寫入紀錄之前，資料集不在它的範圍內，且它們會遮數字。"
)

CODE_BEARING_FLOOR = 0.5
"""`self_sms_ham` 中含獨立 4 至 8 位數字的樣本比例下界。

真實一次性密碼與物流通知幾乎都帶碼，低於此比例即代表去識別遮到了數字串。
**這是一個會失敗的檢查，不是一句叮嚀。**
"""

IDS_FILENAME = "cofacts_ids.jsonl"
MANIFEST_FILENAME = "manifest.json"
IDS_FIELDS = ("id", "subset", "text_sha256")
"""版控的識別字清單每一行的欄位，**刻意不含 `label`**。

label 由 `subset` 決定，而 `subset` 的定義是本專案自己的 selector 常數。
這樣版控的內容只有一串公開識別字與一串雜湊，沒有任何一個位元組是 Cofacts 的
文字，授權問題因此退化成一個幾乎不存在的問題。
"""


@dataclass(frozen=True)
class Sample:
    """測試集中的一則。`label` 與 `split` 皆為推導值，不是存下來的欄位。"""

    id: str
    text: str
    subset: str

    @property
    def label(self) -> str:
        """標籤完全由子集決定。未知子集拋 `KeyError`，不回傳任何預設值。"""
        return SUBSETS[self.subset].label

    @property
    def split(self) -> str:
        """歸屬由識別字現算。自建通知子集全部為 holdout，不經雜湊切分。

        `conversation_ham` 的 `id` 是**正規化文字的 sha256 十六進位**（見
        `tools/fetch_conversation_corpus.py`），因此其首位元組即
        `sha256(正規化文字).digest()[0]`（十六進位前兩碼 = 首位元組的值）。
        切點沿用 `SPLIT_BOUNDARY`：`< 128 → tune`、`>= 128 → holdout`。
        不重新雜湊 `id` —— 那會變成 `sha256(sha256(文字))`，切的不是設計要的內容雜湊。
        """
        if self.subset == SELF_SMS_HAM.name:
            return HOLDOUT
        if self.subset == CONVERSATION_HAM.name:
            return TUNE if int(self.id[:2], 16) < SPLIT_BOUNDARY else HOLDOUT
        return split_of(self.id)

    @property
    def counts_in_rates(self) -> bool:
        """是否得進入任何比率的分子或分母。`case_only` 的子集一律為否。"""
        return self.label != LABEL_CASE_ONLY


@dataclass(frozen=True)
class Testset:
    """已載入的測試集。`missing` 記錄哪些子集的檔案不存在。

    **缺子集是一個要被陳述的事實，不是一個要被補齊的空洞。** 自建子集蒐不到時
    檔案就不存在，此時報告 MUST 寫「該子集未蒐集，最危險的一類誤判未被量測」，
    MUST NOT 以其餘子集的合併數字取代它。
    """

    subsets: Mapping[str, tuple[Sample, ...]]
    missing: tuple[str, ...]
    manifest: Mapping[str, object]

    def samples(self, subset: str) -> tuple[Sample, ...]:
        """取一個子集。未載入的子集拋 `KeyError`，不回傳空 tuple —— 「沒有樣本」
        與「沒有這個檔案」是兩件事，回空的會讓後者看起來像前者。"""
        if subset not in self.subsets:
            raise KeyError(f"子集未載入：{subset!r}（缺少的子集：{self.missing}）")
        return self.subsets[subset]

    def all_samples(self) -> tuple[Sample, ...]:
        return tuple(sample for samples in self.subsets.values() for sample in samples)


def split_of(article_id: str) -> str:
    """依識別字的 SHA-256 摘要首位元組決定歸屬。見模組 docstring。"""
    return TUNE if sha256(article_id.encode("utf-8")).digest()[0] < SPLIT_BOUNDARY else HOLDOUT


def text_sha256(text: str) -> str:
    """內容的雜湊，供重建時逐則比對。編碼固定 UTF-8。"""
    return sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """整個檔案的雜湊，供 manifest 記錄自建子集「可驗證但不可重現」。"""
    return sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    """寫出 JSONL，回傳筆數。呼叫端負責先確認路徑被 git 忽略。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    """讀入 JSONL。空行略過，格式錯誤讓 `json.JSONDecodeError` 傳播。"""
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def subset_path(directory: Path, subset: str) -> Path:
    return directory / f"{subset}.jsonl"


def load_testset(directory: Path, manifest_path: Path) -> Testset:
    """自產出目錄載入全部子集。缺檔的子集記入 `missing`，不以空集合冒充。"""
    loaded: dict[str, tuple[Sample, ...]] = {}
    missing: list[str] = []
    for name in SUBSETS:
        path = subset_path(directory, name)
        if not path.is_file():
            missing.append(name)
            continue
        loaded[name] = tuple(
            Sample(id=record["id"], text=record["text"], subset=name) for record in read_jsonl(path)
        )
    manifest: Mapping[str, object] = MappingProxyType({})
    if manifest_path.is_file():
        manifest = MappingProxyType(json.loads(manifest_path.read_text(encoding="utf-8")))
    return Testset(subsets=MappingProxyType(loaded), missing=tuple(missing), manifest=manifest)


def split_counts(samples: Sequence[Sample]) -> dict[str, int]:
    """兩個切分各有幾則。供 manifest 與「切分與標籤無關」的檢查使用。"""
    counts = {TUNE: 0, HOLDOUT: 0}
    for sample in samples:
        counts[sample.split] += 1
    return counts
