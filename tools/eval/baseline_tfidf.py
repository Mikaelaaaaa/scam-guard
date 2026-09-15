"""TF-IDF 字元 n-gram + 邏輯迴歸 —— 規則系統值不值得做的判準。

`實驗結果.md` 已經量過一個**不需要規則、不需要依據、1 秒訓練完**的 baseline：
970 則平衡取樣上準確率 0.904。`add-speech-act-rules` 拒絕它的理由不是準確率而是
「產不出依據」，`add-verdict-render` 引用同一段話作為本專題存在的理由。
**那個理由需要一個數字撐住**：規則系統若連判別力都輸給它，
「產不出依據」這個優勢就要單獨拿出來論證值不值得規則系統的複雜度成本。

**在同一個框架下評估**：`tune` 訓練、`holdout` 評估、三個 ham 子集分開報告
誤判率、scam 子集報告召回率，全部用 `add-metrics` 的 `Rate`（含 Wilson 區間），
不重新實作任何統計量。

**訓練時允許合併三個 ham 子集，而且僅限此處。** `add-testset` 那條「誤判率
報告不得合併 ham 子集」管的是**報告怎麼呈現**，不管訓練資料怎麼餵 ——
一個分類器的訓練集本來就是一堆帶標籤的樣本。

**`實驗結果.md` 的 970 則是平衡取樣，`holdout` 不是。** TF-IDF 在不平衡資料上
的表現不必然等於平衡取樣下的 0.904，因此本模組**在 holdout 上重新訓練與評估**，
不直接引用那個數字。

`scikit-learn` 只進 `[project.optional-dependencies].eval`，`scam_guard/` 不
import 它、不知道它存在 —— 由 `tests/test_baseline_tfidf.py` 的一條掃描 import
的測試守著。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tools.eval.dataset import Sample
from tools.eval.selectors import HAM_SUBSETS, HOLDOUT, LABEL_HAM, LABEL_SCAM, TUNE
from tools.eval.stats import Rate

ANALYZER = "char"
NGRAM_RANGE = (2, 4)
MAX_ITERATIONS = 1000
RANDOM_STATE = 20260914
"""固定隨機狀態。同一份資料跑兩次要得到同一組數字，否則比較沒有意義。"""

DECISION_THRESHOLD = 0.5
"""判為詐騙的機率門檻。

**這個 baseline 沒有拒答，所以它的棄權率恆為 0。** 這正是要並列呈現的差異：
規則系統以高棄權率換低誤判率，TF-IDF 對每一則都給一個答案。
比較兩者的誤判率而不提這件事，是在比兩個不同的東西。
"""


@dataclass(frozen=True)
class BaselineResult:
    """TF-IDF baseline 在 holdout 上的表現，形狀與規則系統的報告一致。"""

    train_size: int
    false_positive_rates: Mapping[str, Rate]
    recall: Rate
    abstention_rate: Rate


def _label_of(sample: Sample) -> int:
    return 1 if sample.label == LABEL_SCAM else 0


def train_and_evaluate(samples: Sequence[Sample]) -> BaselineResult:
    """在 `tune` 上訓練、在 `holdout` 上評估。兩個切分的 id 集合不相交。"""
    train = [
        sample
        for sample in samples
        if sample.split == TUNE and sample.label in (LABEL_SCAM, LABEL_HAM)
    ]
    test = [
        sample
        for sample in samples
        if sample.split == HOLDOUT and sample.label in (LABEL_SCAM, LABEL_HAM)
    ]
    if not train or not test:
        raise ValueError(f"訓練 {len(train)} 則、評估 {len(test)} 則，兩者皆須非空")

    vectorizer = TfidfVectorizer(analyzer=ANALYZER, ngram_range=NGRAM_RANGE)
    features = vectorizer.fit_transform([sample.text for sample in train])
    model = LogisticRegression(max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE)
    model.fit(features, [_label_of(sample) for sample in train])

    scores = model.predict_proba(vectorizer.transform([sample.text for sample in test]))[:, 1]
    predicted = {
        sample.id: float(score) >= DECISION_THRESHOLD
        for sample, score in zip(test, scores, strict=True)
    }

    false_positive_rates: dict[str, Rate] = {}
    for name in HAM_SUBSETS:
        subset = [sample for sample in test if sample.subset == name]
        if not subset:
            continue
        false_positive_rates[name] = Rate(
            numerator=sum(1 for sample in subset if predicted[sample.id]),
            denominator=len(subset),
        )
    scam = [sample for sample in test if sample.label == LABEL_SCAM]
    return BaselineResult(
        train_size=len(train),
        false_positive_rates=false_positive_rates,
        recall=Rate(
            numerator=sum(1 for sample in scam if predicted[sample.id]), denominator=len(scam)
        ),
        abstention_rate=Rate(numerator=0, denominator=len(test)),
    )


EVIDENCE_NOTE = (
    "**準確率與「能不能產生依據」是兩個獨立的維度，本報告分開呈現。** "
    "TF-IDF 的輸出是一個機率，它產不出「這則訊息要求你把驗證碼傳給對方」"
    "這樣指向具體句子的證據座標 —— 這件事在任何準確率數字下都成立。"
    "反過來說，若規則系統的判別力大幅落後，「依據」這個優勢是否值得規則系統"
    "的維護成本，是一個要留給讀報告的人自己判斷的問題。"
)
