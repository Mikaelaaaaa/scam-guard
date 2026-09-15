"""字元 n-gram 分類器的推論端 —— 一個 `Check`，不是一個平行的子系統。

規則層在 holdout 上召回 5.73%，同一批 holdout 上的 TF-IDF 對照組召回 68.95%。
差的不是邊緣案例是主體：那則中華郵政釣魚訊息在 27 個檢查下命中 0 筆，
因為「有付款要求、有陌生短網域、有冒名機構」這三件事都不是「某個詞出現了」。
沒有一條詞表寫得出「這則訊息的用字與已知詐騙訊息相近」。

**本模組只做推論，權重表由 `tools/train_ngram.py` 離線產生。**
計分是純標準庫的算術：char_wb 切詞 → dict 查表 → `1 + log(tf)` → idf →
L2 正規化 → 與係數點積 → 加截距。`scam_guard/` 的執行期依賴因此維持 `[]`，
瀏覽器端（Pyodide）不需要安裝任何額外套件。

**載入 MUST 由呼叫端顯式執行。** `DEFAULT_MODEL_PATH` 是一個路徑常數，
不是一份已載入的模型 —— 形狀與 `scam_guard.weights` 相同，理由也相同：
import 階段讀檔會讓一個沒跑過安裝的環境在收集測試時就炸。

**`detail` 不含任何 n-gram、任何權重值、任何分數。** 高權重片段確實會到得了
使用者眼前，但走的是 `evidence` 座標這條既有的路 —— 座標由呈現層以
`Document.raw_at()` 取原文，而 log 那一側只拿得到 `RedactedText`。
兩個需求（使用者要看到字、log 不能有字）因此同時滿足。理由見
`add-ngram-classifier` 的 design：top-k n-gram 的集合由資料決定、不封閉，
可以是 `0912`、可以是姓名的兩個字，而一個控制不了自己會吐出訊息哪一塊的
檢查，就不該把那一塊寫進 log。

**`hard` 恆為 `False`。** 硬證據會觸發 `add-score-compute` 的矛盾判定，
而防詐宣導文的字元 n-gram 分布與真詐騙幾乎相同 —— 那正是 bag-of-ngrams
最會犯錯的一類。表中 `hard_capable = false` 是同一件事的交叉檢查。

**門檻不在本模組。** `hit` 由 `weights.toml` 的 `[thresholds.ngram_threshold]`
決定，模組內沒有任何門檻常數 —— 門檻與權重是同一次掃描的兩個輸出，
分開放會讓其中一個被改而另一個不知道。
"""

import json
import math
import operator
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document
from scam_guard.types import CheckResult, Coord, Request
from scam_guard.weights import WeightTable

DEFAULT_MODEL_PATH = Path(__file__).parent / "tables" / "ngram_model.json"
"""權重表的預設位置。**這是一個路徑，不是一份已載入的模型。**"""

NGRAM_SIGNAL = "ngram_classifier"
"""本檢查的名稱，亦為它在 `weights.toml` 中的訊號名稱與群組名稱。"""

NGRAM_THRESHOLD = "ngram_threshold"
"""命中門檻在 `weights.toml` 的 `[thresholds]` 中的 key。"""

MANIFEST_FIELDS = (
    "trained_on",
    "testset_manifest_sha256",
    "n_scam",
    "n_ham",
    "sklearn_version",
    "analyzer",
    "ngram_range",
    "max_features",
    "sublinear_tf",
    "norm",
    "smooth_idf",
    "lowercase",
    "value_decimals",
    "threshold",
)
"""manifest 的必要欄位。缺任何一個即拋例外並指出欄位名，MUST NOT 使用預設值。"""

SUPPORTED_FORMULA: Mapping[str, object] = MappingProxyType(
    {
        "analyzer": "char_wb",
        "sublinear_tf": True,
        "norm": "l2",
        "smooth_idf": True,
        "lowercase": True,
    }
)
"""本模組實作的那一條公式。表中的值與它不符時載入拋例外。

`sklearn` 的預設值與本系統想要的不完全一致（`sublinear_tf` 預設為 `False`），
而不一致的地方**安靜** —— 分數會系統性偏低而沒有任何地方報告。
這不是 fallback（沒有第二條計算路徑），是一道邊界檢查。
"""

EVIDENCE_MASS_SHARE = 0.5
"""取正貢獻由大到小累積到總和的這個比例為止，其餘不進 `evidence`。

**這個 0.5 沒有實測依據**，與 `max_evidence_lines = 5` 同一個性質。
**它不影響分數與判定** —— 分數是全部命中 n-gram 的點積，與挑哪幾個無關；
它只決定使用者看到哪幾句原文。
"""

NGRAM_DETAIL = (
    "訊息用字與 Cofacts 公開查核資料中的詐騙案例相近："
    "該語料 {n_scam} 則詐騙案例中有 {p_scam:.1%} 觸發此比對，"
    "{n_ham} 則非詐騙案例中有 {p_ham:.1%} 觸發；"
    "本則的觸發字串集中於 {n_sentences} 個句子"
)
"""依據的唯一形式。四個代入值全部有來源，模組內不另寫任何一份數字。

兩個比率取自 `weights.toml` 中本訊號的 `measured` 條目，兩個則數取自模型
manifest 的 `n_scam` / `n_ham`（表的 schema 沒有承載則數的欄位，而在這裡
寫死它們就是第二個真相來源）。句子數來自本次判定。

**不寫「詐騙訊息」四個字**：`scam_guard.render.VERDICT_CLAIMS` 禁的就是它。
改寫成「詐騙案例」不是修辭，是通過那張禁用表 —— design 給的示範字串
含「詐騙訊息」，那是 design 自己的四條逐項對照漏掉的一項。
"""

_WHITESPACE_RUN = re.compile(r"\s\s+")
"""`sklearn` 的 `CountVectorizer._white_spaces`：連續兩個以上的空白收成一個。

逐字重現而不是「差不多的正規化」—— 差一個字元，切出來的 n-gram 就不同，
而那個差異會系統性地改變分數且沒有任何地方會報告。`tests/test_ngram_golden.py`
以 `tune` 全量比對兩份實作，容差 `1e-12`。
"""


@dataclass(frozen=True)
class NgramModel:
    """一份已載入的模型。不可變 —— `scam_guard/` 不產生也不修改權重表。

    `terms` 的值是 `(idf, coef)`，一次查表取到兩個數。
    """

    path: Path
    manifest: Mapping[str, object]
    intercept: float
    terms: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "terms", MappingProxyType(dict(self.terms)))

    def __repr__(self) -> str:
        return (
            f"NgramModel(path={self.path!r}, terms={len(self.terms)}, intercept={self.intercept!r})"
        )

    @property
    def ngram_range(self) -> tuple[int, int]:
        bounds = self.manifest["ngram_range"]
        return int(bounds[0]), int(bounds[1])  # type: ignore[index]

    @property
    def n_scam(self) -> int:
        return int(self.manifest["n_scam"])  # type: ignore[arg-type]

    @property
    def n_ham(self) -> int:
        return int(self.manifest["n_ham"])  # type: ignore[arg-type]


def char_wb_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    """重現 `sklearn` 的 `CountVectorizer._char_wb_ngrams`，含 `lowercase` 前處理。

    逐詞左右補一個空格再取 n-gram；詞比 n 短時只取一次
    （`sklearn` 的 `if offset == 0: break`）。中文常常整則訊息是一個「詞」，
    所以那個 break 幾乎不會觸發，但漏掉它會在短訊息上產生重複計數。
    """
    minimum, maximum = ngram_range
    collapsed = _WHITESPACE_RUN.sub(" ", text.lower())
    ngrams: list[str] = []
    for word in collapsed.split():
        padded = " " + word + " "
        length = len(padded)
        for size in range(minimum, maximum + 1):
            offset = 0
            ngrams.append(padded[offset : offset + size])
            while offset + size < length:
                offset += 1
                ngrams.append(padded[offset : offset + size])
            if offset == 0:
                break
    return ngrams


def document_text(doc: Document) -> str:
    """分類器看到的那一份文字：全部正規化後的句子，以換行接起來。

    **訓練端 MUST 呼叫本函式取得輸入文字。** 訓練看正規化前的原文、推論看
    正規化後的文字，是 train/serve skew 最典型的形狀，而它不會有任何地方報錯
    —— 只會讓線上的分數系統性偏低。接起來用什麼字元不影響切詞
    （`char_wb` 先把連續空白收成一個再 `split()`），用換行只是因為它就是句界。
    """
    return "\n".join(doc.sentences)


def _require_manifest(raw: Mapping[str, object], path: Path) -> Mapping[str, object]:
    missing = [field for field in MANIFEST_FIELDS if field not in raw]
    if missing:
        raise ValueError(
            f"ngram 模型的 manifest 缺少必要欄位：{'、'.join(missing)}"
            f"（模型：{path}）—— MUST NOT 使用任何預設值"
        )
    for field, supported in SUPPORTED_FORMULA.items():
        if raw[field] != supported:
            raise ValueError(
                f"ngram 模型的 manifest.{field} 為 {raw[field]!r}，"
                f"而本推論端實作的公式要求 {supported!r}（模型：{path}）—— "
                f"不提供任何替代計算路徑"
            )
    return raw


def load_model(path: Path = DEFAULT_MODEL_PATH) -> NgramModel:
    """讀取並驗證權重表。**不在 import 時發生，也不回退為任何內建預設模型。**

    檔案缺失讓 `FileNotFoundError` 傳播（訊息已含路徑），JSON 損毀讓
    `json.JSONDecodeError` 傳播，欄位缺漏或公式不符拋 `ValueError` 並指出欄位。
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    for key in ("manifest", "intercept", "terms"):
        if key not in document:
            raise ValueError(f"ngram 模型缺少必要的頂層鍵 {key!r}（模型：{path}）")
    manifest = _require_manifest(document["manifest"], path)
    terms = {gram: (float(pair[0]), float(pair[1])) for gram, pair in document["terms"].items()}
    if not terms:
        raise ValueError(f"ngram 模型的 terms 為空（模型：{path}）")
    return NgramModel(
        path=path,
        manifest=manifest,
        intercept=float(document["intercept"]),
        terms=terms,
    )


@dataclass(frozen=True)
class NgramScore:
    """一次計分的結果與它的逐 n-gram 貢獻。

    `contributions` 是 `(n-gram, 貢獻)`，貢獻為正規化後的 tf-idf 乘上係數。
    它只供 `evidence` 使用 —— **分數是全部貢獻的和加上截距，與挑哪幾個無關。**
    """

    value: float
    contributions: tuple[tuple[str, float], ...]


def score(model: NgramModel, text: str) -> NgramScore:
    """`1 + log(tf)` × idf → L2 正規化 → 與係數點積 → 加截距。

    詞表外的 n-gram 不進向量，因此 L2 正規化是在**表內項**上做的 ——
    與 `sklearn` 的 `CountVectorizer.transform()` 一致（它只數詞表內的項）。
    一則訊息完全沒有表內項時向量為零向量，分數即截距，貢獻為空。
    """
    counts = Counter(
        gram for gram in char_wb_ngrams(text, model.ngram_range) if gram in model.terms
    )
    weighted: list[tuple[str, float]] = []
    for gram, count in counts.items():
        idf, _ = model.terms[gram]
        weighted.append((gram, (1.0 + math.log(count)) * idf))
    norm = math.sqrt(sum(value * value for _, value in weighted))
    if norm == 0.0:
        return NgramScore(value=model.intercept, contributions=())
    contributions = tuple((gram, value / norm * model.terms[gram][1]) for gram, value in weighted)
    return NgramScore(
        value=model.intercept + sum(value for _, value in contributions),
        contributions=contributions,
    )


def evidence_coords(contributions: Sequence[tuple[str, float]], doc: Document) -> list[Coord]:
    """正貢獻累積至半數質量的那些 n-gram 落在哪些句子，去重後即為座標。

    找不到句子（n-gram 跨句子邊界 —— `char_wb` 是在整份文字上切的）時回空陣列。
    **空 `evidence` 是合法值**，`hit` 仍為 `True`，呈現層就不附原文片段。
    MUST NOT 為了湊一個座標而回報第 0 句。
    """
    positive = [entry for entry in contributions if entry[1] > 0.0]
    if not positive:
        return []
    ordered = sorted(positive, key=operator.itemgetter(1), reverse=True)
    target = EVIDENCE_MASS_SHARE * sum(value for _, value in ordered)
    taken: list[str] = []
    accumulated = 0.0
    for gram, value in ordered:
        taken.append(gram)
        accumulated += value
        if accumulated >= target:
            break
    lowered = [sentence.lower() for sentence in doc.sentences]
    coords: list[Coord] = []
    for gram in taken:
        needle = gram.strip()
        if not needle:
            continue
        for index, sentence in enumerate(lowered):
            if needle in sentence and doc.coords[index] not in coords:
                coords.append(doc.coords[index])
    return coords


def _measured_probabilities(table: WeightTable) -> tuple[float, float]:
    """本訊號的兩個條件機率。表中不是 `measured` 時拋例外，不編任何數字。"""
    if NGRAM_SIGNAL not in table.signals:
        raise KeyError(f"訊號未登錄於權重表：name={NGRAM_SIGNAL!r}（表：{table.path}）")
    weight = table.signals[NGRAM_SIGNAL].weight_soft
    if weight.p_hit_given_scam is None or weight.p_hit_given_ham is None:
        raise ValueError(
            f"訊號 {NGRAM_SIGNAL!r} 的 basis 為 {weight.basis!r}，沒有條件機率可寫進依據；"
            f"本檢查的依據只由實測比率構成（表：{table.path}）"
        )
    return weight.p_hit_given_scam, weight.p_hit_given_ham


@dataclass(frozen=True)
class NgramClassifierCheck:
    """字元 n-gram 分類器。`Stage.LOCAL`，不宣告 `wants_prior`，`hard` 恆為 False。

    `Stage.LOCAL` 有兩個理由，其中一個是量出來的：純 Python 推論在 500 字的
    訊息上實測 0.22 ms，與 165 黑名單的 dict 查表同一個數量級。另一個是結構的
    —— `EXPENSIVE` 的語意是「可被短路」，而短路的觸發條件是已有硬證據；
    把本訊號放進可被短路的階段，效果是規則已命中時把它關掉，
    而那正好是需要量它的條件機率的場合。一個只在不需要它的時候執行的訊號，量不到。

    `scam_types` 恆為空陣列：它的輸出是一個分數，不含任何可對應到 165 案類的
    資訊。連帶後果是單獨命中時 `cap_unseen_pattern` 生效。
    """

    model: NgramModel
    table: WeightTable
    name: str = NGRAM_SIGNAL
    stage: Stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        result = score(self.model, document_text(doc))
        if result.value < self.table.threshold(NGRAM_THRESHOLD):
            return []
        p_scam, p_ham = _measured_probabilities(self.table)
        coords = evidence_coords(result.contributions, doc)
        return [
            CheckResult(
                name=self.name,
                hit=True,
                detail=NGRAM_DETAIL.format(
                    n_scam=self.model.n_scam,
                    p_scam=p_scam,
                    n_ham=self.model.n_ham,
                    p_ham=p_ham,
                    n_sentences=len(coords),
                ),
                evidence=coords,
                scam_types=[],
                hard=False,
            )
        ]


def register_ngram_check(
    registry: CheckRegistry,
    table: WeightTable,
    *,
    model: NgramModel | None = None,
) -> None:
    """把分類器註冊進 registry。**未提供 `model` 時不註冊。**

    沿用 `register_url_checks()` 對 `store=None` 的處置：不註冊一個永遠不命中的
    空檢查，否則「沒有訊號」與「沒有模型」在 `Verdict.checks` 裡看起來一模一樣。
    """
    if model is None:
        return
    registry.register(NgramClassifierCheck(model=model, table=table))
