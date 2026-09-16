"""離線訓練字元 n-gram 分類器，產出 `scam_guard/tables/ngram_model.json`。

    python -m tools.train_ngram

**只用 `tune`。** `holdout` 不出現在本程式的任何一條資料路徑上 —— 超參數選擇、
係數擬合與門檻掃描全部在 `tune` 上完成。切分沿用 `tools/eval/dataset.py` 的
`split_of()`，不自行實作。

**文字取得路徑與 `detect()` 相同。** 樣本先過 `scam_guard.normalize.build_document`
再過 `scam_guard.ngram.document_text`，兩端因此看到同一份字。訓練看正規化前的
原文、推論看正規化後的文字，是 train/serve skew 最典型的形狀，而它不會有任何
地方報錯，只會讓線上的分數系統性偏低。

**門檻與權重是同一次掃描的兩個輸出。** 候選門檻是分類器在 `tune` 上實際產生的
分數值（不是等距網格，那樣就會有「間距怎麼選」這個問題）；選擇準則是取使
`p_hit_given_ham` 的 95% Wilson 上界 ≤ 2% 的**最小**門檻。2% 取自
`add-speech-act-rules` 已定的誤判率驗收門檻，**它不保證系統層的誤判率 ≤ 2%**
—— 一個訊號的 ham 命中率不是系統的誤判率（系統的誤判還要過 `confidence_floor`
與 `decision_score` 兩道閘）。可行集合為空時以非零結束碼結束並印出曲線，
MUST NOT 放寬。

**`sklearn` 只出現在本檔。** `scam_guard/` 的 import 圖裡沒有這個節點，
由 `tests/test_baseline_tfidf.py` 的掃描測試守著。
"""

import argparse
import csv
import gzip
import json
import math
import operator
import resource
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from scam_guard.ngram import (
    DEFAULT_MODEL_PATH,
    MANIFEST_FIELDS,
    NGRAM_SIGNAL,
    NGRAM_THRESHOLD,
    NgramModel,
    document_text,
    load_model,
    score,
)
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.types import Message
from scam_guard.weights import load_weights
from tools.eval.dataset import MANIFEST_FILENAME, Sample, Testset, file_sha256, load_testset
from tools.eval.run import build_registry, load_blocklist, load_psl, run_over
from tools.eval.selectors import (
    COFACTS_SCAM,
    CONVERSATION_HAM,
    HAM_SUBSETS,
    HOLDOUT,
    LABEL_HAM,
    LABEL_SCAM,
    TUNE,
)
from tools.eval.stats import Rate

ANALYZER = "char_wb"
"""**固定，不納入掃描。** `char` 會跨過詞邊界產生橫跨標點與空白的 n-gram，
而更重要的是 `tools/eval/baseline_tfidf.py` 用的是 `analyzer="char"` ——
兩者若都掃，68.95% 那個對照數字就不再是同一件事的對照。"""

NGRAM_RANGES: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4))
MAX_FEATURES: tuple[int | None, ...] = (10_000, 20_000, 50_000, 100_000, None)
"""25 格。`None` 那一格 MUST 也跑 —— 它是「砍維度反而更準」這個結論在本語料上
成不成立的唯一檢查。"""

EXTERNAL_CORPUS_NOTE = (
    "另一份 800k 簡體行銷簡訊語料上量到 char_wb 2–4 無上限 F1 0.9808、"
    "char_wb 2–4 max_features=50,000 F1 0.9860、char_wb 1–3 max_features=50,000 "
    "F1 0.9911。**那三個數字不轉移到本語料，一個都不轉移** —— 語料不同、"
    "長度分布不同、類別比例不同、規模差 570 倍（800k vs 1,396）。"
    "可轉移的只有「體積」那一欄，因為體積只是 max_features 乘上每條記錄的"
    "位元組數，與語料無關。"
)

METHOD_COMPARISON_NOTE = (
    "方法選型：TF-IDF char n-gram + 邏輯迴歸是**唯一在 Cofacts holdout 上有數字"
    "的那個**（68.95% 對規則層的 5.73%），不是量過之後最好的那個 —— "
    "Naive Bayes 與餘弦相似度在這份語料上沒有任何數字，三方對照實測從未執行。"
)

HAM_UPPER_CEILING = 0.02
"""門檻選擇準則：`p_hit_given_ham` 的 95% Wilson 上界不得超過此值。

取自 `add-speech-act-rules` 已定的誤判率驗收門檻（「95% Wilson 上界 ≤ 2%」），
是全專案唯一一個有來歷的比率。**它是單一訊號的 ham 命中率上界，
不是系統的誤判率。**
"""

RECALL_FLOOR = 0.75
"""停線：重訓後 scam 召回（Cofacts scam tune，`p_hit_given_scam`）低於此值即停。

75% 不是隨手挑的：`add-ngram-classifier` 已量到
`P(命中 | scam, 至少一條規則命中) = 76.24%`，低於此值意味分類器的單獨補充覆蓋
已被侵蝕到規則已能覆蓋的水準以下 —— 那正是這個訊號存在理由變弱的訊號。
觸發時印出完整曲線、以非零結束碼結束、不寫入表（沿用 `choose_threshold` 的精神）。
"""

CONVERSATION_UPPER_CEILING = 0.02
"""停線：對話誤判目標。`conversation_ham` 的 holdout 部分在選定門檻上的誤判率，
其 95% Wilson 上界須 ≤ 此值。掃遍門檻都下不到即停 —— MUST NOT 放寬 2%、
MUST NOT 挑一個看起來可以的門檻。"""

MAX_ITERATIONS = 1000
RANDOM_STATE = 20260915
CV_FOLDS = 5
VALUE_DECIMALS = 8
"""表中數值四捨五入到的小數位數。翻轉判定的樣本數不為零時 MUST 調高它，
MUST NOT 調鬆 golden test 的容差。"""

DEFAULT_TESTSET_DIR = Path("testset")
DEFAULT_DATA_DIR = Path("data/testset")
DEFAULT_PSL_DIR = Path("data/psl")
DEFAULT_BLOCKLIST_DIR = Path("data/blocklist")
DEFAULT_MODEL_OUT = Path("scam_guard/tables/ngram_model.json")
DEFAULT_REPORT_DIR = Path("data/reports/ngram")

MODEL_NOTE = "訓練產物。人工編輯此檔案沒有意義 —— 值由 tools/train_ngram.py 產生。"

TRAINED_ON = (
    "testset/manifest.json 的 tune 切分：Cofacts 公開查核資料（CC BY-SA 4.0；"
    "姓名標示與授權條款見 testset/LICENSE-DATA）+ conversation_ham 的 tune 部分"
    "（Gossiping-Chinese-Corpus，PTT 八卦版使用者貼文，作者 zake7749 以 Apache-2.0 "
    "釋出；原生台灣繁中，未經 OpenCC 轉繁）。**trained_on 記錄完整訓練組成，"
    "與 weights.toml 的 measured_on（量測池，只有 Cofacts tune）不同名是刻意的**"
)

GRID_COLUMNS = (
    "ngram_range",
    "max_features",
    "n_features",
    "cv_f1",
    "cv_precision",
    "cv_recall",
    "table_bytes",
    "table_gzip_bytes",
)

SWEEP_COLUMNS = (
    "threshold",
    "scam_hits",
    "scam_n",
    "p_hit_given_scam",
    "p_scam_lower",
    "p_scam_upper",
    "ham_hits",
    "ham_n",
    "p_hit_given_ham",
    "p_ham_lower",
    "p_ham_upper",
    "log_likelihood_ratio",
    "feasible",
)


@dataclass(frozen=True)
class GridCell:
    """一格超參數與它在 `tune` 上的交叉驗證表現與表的體積。"""

    ngram_range: tuple[int, int]
    max_features: int | None
    n_features: int
    cv_f1: float
    cv_precision: float
    cv_recall: float
    table_bytes: int
    table_gzip_bytes: int


@dataclass(frozen=True)
class SweepPoint:
    """一個候選門檻與它的兩個條件機率。`feasible` 即 ham 側上界是否 ≤ 2%。"""

    threshold: float
    p_hit_given_scam: Rate
    p_hit_given_ham: Rate
    feasible: bool

    @property
    def log_likelihood_ratio(self) -> float | None:
        """`ln(p_scam / p_ham)`。ham 側零命中時不存在，回 `None` 而不編一個數字。"""
        if self.p_hit_given_ham.numerator == 0 or self.p_hit_given_scam.numerator == 0:
            return None
        return math.log(self.p_hit_given_scam.value / self.p_hit_given_ham.value)


def tune_samples(data_dir: Path, manifest_path: Path) -> tuple[Sample, ...]:
    """`tune` 切分的全部樣本，依 id 排序使兩次執行看到同一個順序。

    `holdout` 在此被濾掉，之後的每一條路徑上都不再有它。
    """
    testset = load_testset(data_dir, manifest_path)
    if not testset.subsets:
        raise FileNotFoundError(
            f"測試集尚未重建：{data_dir} 為空。"
            f"請先執行 `python -m tools.eval.build_testset rebuild`。"
        )
    selected = [
        sample
        for sample in testset.all_samples()
        if sample.split == TUNE and sample.label in (LABEL_SCAM, LABEL_HAM)
    ]
    return tuple(sorted(selected, key=operator.attrgetter("id")))


def normalized_text(sample: Sample) -> str:
    """與 `detect()` 完全相同的一條正規化路徑。"""
    return document_text(build_document([Message(text=sample.text)], DEFAULT_LIMITS))


def _labels(samples: Sequence[Sample]) -> list[int]:
    return [1 if sample.label == LABEL_SCAM else 0 for sample in samples]


def build_pipeline(ngram_range: tuple[int, int], max_features: int | None) -> Pipeline:
    """固定五個公式相關設定；它們同時被寫進 manifest 並由推論端驗證。"""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=ANALYZER,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    sublinear_tf=True,
                    norm="l2",
                    smooth_idf=True,
                    lowercase=True,
                ),
            ),
            (
                "logistic",
                LogisticRegression(max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE),
            ),
        ]
    )


def _f1(truth: Sequence[int], predicted: Sequence[int]) -> tuple[float, float, float]:
    """正類的 precision / recall / F1。三個數字都要報，F1 單獨看不出偏在哪一側。"""
    true_positive = sum(1 for a, b in zip(truth, predicted, strict=True) if a == 1 and b == 1)
    predicted_positive = sum(1 for value in predicted if value == 1)
    actual_positive = sum(1 for value in truth if value == 1)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    harmonic = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return precision, recall, harmonic


def model_document(
    pipeline: Pipeline,
    *,
    manifest: Mapping[str, object],
    value_decimals: int,
) -> dict:
    """把擬合好的 pipeline 序列化成表的那四個頂層鍵。

    `terms` 的鍵依字典序排序，使同一份輸入兩次執行產生位元組相同的檔案。
    """
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    logistic: LogisticRegression = pipeline.named_steps["logistic"]
    idf = vectorizer.idf_
    coefficients = logistic.coef_[0]
    terms = {
        gram: [
            round(float(idf[index]), value_decimals),
            round(float(coefficients[index]), value_decimals),
        ]
        for gram, index in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(0))
    }
    return {
        "_note": MODEL_NOTE,
        "manifest": dict(manifest),
        "intercept": round(float(logistic.intercept_[0]), value_decimals),
        "terms": terms,
    }


def serialize(document: Mapping[str, object]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def scan_grid(texts: Sequence[str], labels: Sequence[int]) -> tuple[GridCell, ...]:
    """25 格，逐格以 `tune` 上的交叉驗證 F1 評分，並量表的明文與 gzip 位元組數。

    交叉驗證在 `Pipeline` 上做，向量化器在每一折內重新擬合 —— 先在全量上
    `fit_transform` 再切折就是把驗證折的詞彙洩進訓練，而那會讓每一格都好看。
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cells: list[GridCell] = []
    for ngram_range in NGRAM_RANGES:
        for max_features in MAX_FEATURES:
            pipeline = build_pipeline(ngram_range, max_features)
            predicted = cross_val_predict(pipeline, list(texts), list(labels), cv=folds)
            precision, recall, harmonic = _f1(labels, list(predicted))
            fitted = build_pipeline(ngram_range, max_features).fit(list(texts), list(labels))
            payload = serialize(
                model_document(
                    fitted,
                    manifest={"probe": True},
                    value_decimals=VALUE_DECIMALS,
                )
            ).encode("utf-8")
            cells.append(
                GridCell(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    n_features=len(fitted.named_steps["tfidf"].vocabulary_),
                    cv_f1=harmonic,
                    cv_precision=precision,
                    cv_recall=recall,
                    table_bytes=len(payload),
                    table_gzip_bytes=len(gzip.compress(payload)),
                )
            )
            print(
                f"[grid] ngram_range={ngram_range} max_features={max_features} "
                f"n_features={cells[-1].n_features} F1={harmonic:.4f} "
                f"gzip={cells[-1].table_gzip_bytes}",
                file=sys.stderr,
            )
    return tuple(cells)


def _cell_rank(cell: GridCell) -> tuple[float, int]:
    """排序鍵：F1 降序、維度升序。平手取維度小者 —— 體積直接決定 Pages 能不能用。"""
    return (-cell.cv_f1, cell.n_features)


def choose_cell(cells: Sequence[GridCell]) -> GridCell:
    return sorted(cells, key=_cell_rank)[0]


def sweep_thresholds(scores: Sequence[float], labels: Sequence[int]) -> tuple[SweepPoint, ...]:
    """候選門檻 = 分類器在 `tune` 上實際產生的分數值（去重升序）。

    候選集合不發明，所以沒有「網格間距是怎麼選的」這個問題。
    """
    scam_scores = [value for value, label in zip(scores, labels, strict=True) if label == 1]
    ham_scores = [value for value, label in zip(scores, labels, strict=True) if label == 0]
    points: list[SweepPoint] = []
    for threshold in sorted(set(scores)):
        ham_hits = sum(1 for value in ham_scores if value >= threshold)
        ham_rate = Rate(numerator=ham_hits, denominator=len(ham_scores))
        points.append(
            SweepPoint(
                threshold=threshold,
                p_hit_given_scam=Rate(
                    numerator=sum(1 for value in scam_scores if value >= threshold),
                    denominator=len(scam_scores),
                ),
                p_hit_given_ham=ham_rate,
                feasible=ham_rate.upper <= HAM_UPPER_CEILING,
            )
        )
    return tuple(points)


def choose_threshold(points: Sequence[SweepPoint]) -> SweepPoint:
    """取可行集合中的最小門檻。取端點使選擇沒有任何餘地。

    可行集合為空時 raise —— MUST NOT 放寬上界、MUST NOT 改用點估計、
    MUST NOT 挑一個看起來可以的值。
    """
    feasible = [point for point in points if point.feasible]
    if not feasible:
        raise ValueError(
            f"掃遍 {len(points)} 個候選門檻，沒有任何一個使 p_hit_given_ham 的 95% "
            f"Wilson 上界 ≤ {HAM_UPPER_CEILING:.0%}。這代表本模型在這份語料上沒有一個"
            f"可用的操作點 —— 不放寬約束，不寫入 weights.toml，不註冊檢查。"
        )
    chosen = sorted(feasible, key=operator.attrgetter("threshold"))[0]
    if chosen.p_hit_given_ham.numerator == 0:
        raise ValueError(
            f"選出的門檻 {chosen.threshold} 在 ham 側零命中，"
            f"ln(p_scam / 0) 不存在且 weights.toml 要求兩個機率落在開區間 (0, 1)。"
            f"此時權重只有下界（見 tools/eval/signals.py 的 lower_bound_only），"
            f"MUST NOT 被寫進 value。"
        )
    if chosen.p_hit_given_scam.numerator == 0:
        raise ValueError(f"選出的門檻 {chosen.threshold} 在 scam 側零命中，本訊號沒有證據")
    return chosen


@dataclass(frozen=True)
class Correlation:
    """分類器與規則的條件相關性 —— 三個量，各附 Wilson 區間。"""

    given_scam: Rate
    given_scam_with_rule: Rate
    given_scam_without_rule: Rate


def rule_hit_flags(
    samples: Sequence[Sample], psl_dir: Path, blocklist_dir: Path
) -> tuple[bool, ...]:
    """每一則有沒有任何一條規則命中。用的是評估側既有的完整 registry。"""
    psl = load_psl(psl_dir)
    store = load_blocklist(blocklist_dir, psl)
    table = load_weights()
    registry = build_registry(psl, store=store)
    records = run_over(samples, registry, table)
    return tuple(bool(record.hit_signals) for record in records)


def measure_correlation(
    scores: Sequence[float],
    labels: Sequence[int],
    rule_hits: Sequence[bool],
    threshold: float,
) -> Correlation:
    """`P(分類器命中 | scam)`、同上再條件於「至少一條規則命中」與「無任何規則命中」。

    第三列接近 0 代表本訊號沒有補到規則以外的覆蓋，**而那是一個該讓本 change
    停下來的結果**，不是一個可以繞過的數字。
    """
    scam = [
        (value >= threshold, hit)
        for value, label, hit in zip(scores, labels, rule_hits, strict=True)
        if label == 1
    ]
    with_rule = [entry for entry in scam if entry[1]]
    without_rule = [entry for entry in scam if not entry[1]]
    return Correlation(
        given_scam=Rate(numerator=sum(1 for entry in scam if entry[0]), denominator=len(scam)),
        given_scam_with_rule=Rate(
            numerator=sum(1 for entry in with_rule if entry[0]), denominator=len(with_rule)
        ),
        given_scam_without_rule=Rate(
            numerator=sum(1 for entry in without_rule if entry[0]),
            denominator=len(without_rule),
        ),
    )


def score_from_table(
    document: Mapping[str, object], analyzer: Callable[[str], list[str]], text: str
) -> float:
    """訓練端的第二份實作：讀**同一份 JSON**，以 `sklearn` 的切詞器與 numpy 重算。

    這是 golden test 的 `C`。它**不得**改用記憶體中的 `sklearn` 模型 ——
    表裡的值是四捨五入過的，拿它與 float64 的模型比，差異一定存在且與實作正確
    與否無關，測試就會被調鬆直到沒有意義。
    """
    terms: Mapping[str, Sequence[float]] = document["terms"]  # type: ignore[assignment]
    counts: dict[str, int] = {}
    for gram in analyzer(text):
        if gram in terms:
            counts[gram] = counts.get(gram, 0) + 1
    if not counts:
        return float(document["intercept"])  # type: ignore[arg-type]
    grams = sorted(counts)
    tf = np.array([1.0 + np.log(counts[gram]) for gram in grams], dtype=np.float64)
    idf = np.array([terms[gram][0] for gram in grams], dtype=np.float64)
    coefficients = np.array([terms[gram][1] for gram in grams], dtype=np.float64)
    weighted = tf * idf
    norm = np.sqrt(np.sum(weighted * weighted))
    return float(np.dot(weighted / norm, coefficients) + float(document["intercept"]))  # type: ignore[arg-type]


def build_analyzer(ngram_range: tuple[int, int]) -> Callable[[str], list[str]]:
    """`sklearn` 自己的 `char_wb` 切詞器，供 `score_from_table` 使用。

    golden test 的 `C` 用它、`B` 用 `scam_guard.ngram.char_wb_ngrams`，
    所以 `|B − C|` 同時是切詞實作的一致性檢查。
    """
    return TfidfVectorizer(
        analyzer=ANALYZER, ngram_range=ngram_range, lowercase=True
    ).build_analyzer()


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _grid_rows(cells: Sequence[GridCell]) -> list[list[str]]:
    return [
        [
            f"{cell.ngram_range[0]}-{cell.ngram_range[1]}",
            "none" if cell.max_features is None else str(cell.max_features),
            str(cell.n_features),
            f"{cell.cv_f1:.6f}",
            f"{cell.cv_precision:.6f}",
            f"{cell.cv_recall:.6f}",
            str(cell.table_bytes),
            str(cell.table_gzip_bytes),
        ]
        for cell in cells
    ]


def _sweep_rows(points: Sequence[SweepPoint]) -> list[list[str]]:
    rows: list[list[str]] = []
    for point in points:
        ratio = point.log_likelihood_ratio
        rows.append(
            [
                f"{point.threshold:.6f}",
                str(point.p_hit_given_scam.numerator),
                str(point.p_hit_given_scam.denominator),
                f"{point.p_hit_given_scam.value:.6f}",
                f"{point.p_hit_given_scam.lower:.6f}",
                f"{point.p_hit_given_scam.upper:.6f}",
                str(point.p_hit_given_ham.numerator),
                str(point.p_hit_given_ham.denominator),
                f"{point.p_hit_given_ham.value:.6f}",
                f"{point.p_hit_given_ham.lower:.6f}",
                f"{point.p_hit_given_ham.upper:.6f}",
                "" if ratio is None else f"{ratio:.6f}",
                "true" if point.feasible else "false",
            ]
        )
    return rows


def _report_lines(
    cells: Sequence[GridCell],
    chosen_cell: GridCell,
    points: Sequence[SweepPoint],
    chosen: SweepPoint,
    correlation: Correlation,
    baseline: SweepPoint | None,
    quantisation: tuple[float, int],
    peak_rss_bytes: int,
) -> list[str]:
    weight = math.log(chosen.p_hit_given_scam.value / chosen.p_hit_given_ham.value)
    lines = [
        "# 字元 n-gram 分類器：訓練報告",
        "",
        METHOD_COMPARISON_NOTE,
        "",
        "## 超參數掃描（25 格，`tune` 上的 5 折交叉驗證）",
        "",
        f"`analyzer` 固定為 `{ANALYZER}`，不納入掃描：`baseline_tfidf.py` 用的是 "
        "`analyzer=char`，兩者若都掃，68.95% 那個對照數字就不再是同一件事的對照。",
        "",
        EXTERNAL_CORPUS_NOTE,
        "",
        "| ngram_range | max_features | 維度 | CV F1 | CV precision | CV recall | "
        "明文 bytes | gzip bytes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell.ngram_range[0]}–{cell.ngram_range[1]} | "
            f"{'無上限' if cell.max_features is None else cell.max_features} | "
            f"{cell.n_features} | {cell.cv_f1:.4f} | {cell.cv_precision:.4f} | "
            f"{cell.cv_recall:.4f} | {cell.table_bytes} | {cell.table_gzip_bytes} |"
        )
    lines += [
        "",
        f"選定：`ngram_range={chosen_cell.ngram_range}`、"
        f"`max_features={chosen_cell.max_features}`，"
        f"維度 {chosen_cell.n_features}，CV F1 {chosen_cell.cv_f1:.4f}。"
        "選擇依據為 `tune` 上的交叉驗證 F1，平手取維度小者。",
        f"掃描程序峰值 RSS：{peak_rss_bytes:,} B（`resource.getrusage`）。",
        "",
        "## 門檻",
        "",
        f"候選門檻 {len(points)} 個，全部是 `tune` 樣本的實際分數。"
        f"準則：取使 `p_hit_given_ham` 的 95% Wilson 上界 ≤ "
        f"{HAM_UPPER_CEILING:.0%} 的**最小**門檻。"
        "2% 取自 `add-speech-act-rules` 已定的誤判率驗收門檻。",
        "",
        "⚠️ **這條約束不保證系統層的誤判率 ≤ 2%。** 一個訊號的 ham 命中率不是系統的"
        "誤判率 —— 系統的誤判還要過 `confidence_floor` 與 `decision_score` 兩道閘。",
        "",
        f"- 選定門檻：`{chosen.threshold:.6f}`",
        f"- `p_hit_given_scam` = {chosen.p_hit_given_scam}",
        f"- `p_hit_given_ham` = {chosen.p_hit_given_ham}",
        f"- 權重 `ln(p_scam / p_ham)` = **{weight:.4f}**",
        "",
    ]
    if baseline is None:
        lines += [
            "`baseline_tfidf.py` 的 `DECISION_THRESHOLD = 0.5` 是一個**機率**門檻，"
            "而本掃描的座標是**分數**（對數勝算），0.5 的機率對應分數 0.0。"
            "該操作點不在候選集合中，因此以最接近的候選並列，見下。",
            "",
        ]
    else:
        loss = chosen.p_hit_given_scam.value - baseline.p_hit_given_scam.value
        lines += [
            "### 並列：無 ham 側約束的操作點",
            "",
            "`tools/eval/baseline_tfidf.py` 的 `DECISION_THRESHOLD = 0.5`（機率）"
            "等價於分數 0.0，該 baseline **沒有 ham 側約束**。"
            "**兩者不可直接比較**，並列的目的是讓「這條約束花了多少召回」成為一個"
            "看得到的數字。",
            "",
            f"- 分數 0.0 附近的候選：`{baseline.threshold:.6f}`，"
            f"`p_hit_given_scam` = {baseline.p_hit_given_scam}、"
            f"`p_hit_given_ham` = {baseline.p_hit_given_ham}",
            f"- **ham 側約束的代價：召回由 {baseline.p_hit_given_scam.value:.2%} 降到 "
            f"{chosen.p_hit_given_scam.value:.2%}，差 {loss:.2%}**",
            "",
        ]
    lines += [
        "## 分類器與規則的條件相關性（`tune` 的 scam 側）",
        "",
        f"- `P(分類器命中 | scam)` = {correlation.given_scam}",
        f"- `P(分類器命中 | scam, 至少一條規則命中)` = {correlation.given_scam_with_rule}",
        f"- `P(分類器命中 | scam, 無任何規則命中)` = {correlation.given_scam_without_rule}",
        "",
    ]
    if correlation.given_scam_without_rule.value < 0.05:
        lines += [
            "⚠️ **無任何規則命中的 scam 樣本中，分類器命中率接近 0："
            "本訊號未提供規則以外的覆蓋。** 這是一個該讓本 change 停下來的結果。",
            "",
        ]
    maximum, flips = quantisation
    lines += [
        "## 量化誤差",
        "",
        f"- `value_decimals` = {VALUE_DECIMALS}",
        f"- `max |A − B|`（sklearn 模型 vs 推論端讀表）= {maximum:.3e}",
        f"- 因量化而翻轉判定的樣本數 = **{flips}**",
        "",
        "翻轉數不為零時 MUST 調高 `value_decimals`，MUST NOT 調鬆 golden test 的容差。",
        "",
    ]
    return lines


def _nearest_point(points: Sequence[SweepPoint], target: float) -> SweepPoint | None:
    """分數軸上最接近 `target` 的候選。無候選時回 `None`。

    以顯式迴圈取最小距離而不是排序加 lambda —— 本專案禁 closure，
    而為一個一次性的距離另外定一個模組層函式會比迴圈難讀。
    """
    nearest: SweepPoint | None = None
    best = math.inf
    for point in points:
        distance = abs(point.threshold - target)
        if distance < best:
            best = distance
            nearest = point
    return nearest


@dataclass(frozen=True)
class SubsetHoldout:
    """一個 ham 子集在其 holdout 上的分類器誤判率。`None` 代表該子集未載入。"""

    name: str
    rate: Rate | None


def _holdout_texts(samples: Sequence[Sample]) -> list[str]:
    """一批樣本中 holdout 部分的正規化文字，與 `detect()` 同一條路徑。"""
    return [normalized_text(sample) for sample in samples if sample.split == HOLDOUT]


def pipeline_hit_rate(pipeline: Pipeline, texts: Sequence[str], threshold: float) -> Rate | None:
    """以擬合好的 pipeline（`decision_function`，float64）量一批文字的命中率。

    量測與門檻選擇同用 `decision_function`，兩者座標一致；golden test 保證它與
    推論端讀表的分數在 `1e-12` 內，因此命中判定與線上一致。空集合回 `None`。
    """
    if not texts:
        return None
    scores = [float(value) for value in pipeline.decision_function(list(texts))]
    return Rate(numerator=sum(1 for value in scores if value >= threshold), denominator=len(scores))


def model_hit_rate(model: NgramModel, texts: Sequence[str], threshold: float) -> Rate | None:
    """以序列化模型的推論端（`scam_guard.ngram.score`）量命中率。

    供「現行模型」的 before 量測 —— 它讀的是版控中的那份表，是真正上線的推論路徑。
    空集合回 `None`。
    """
    if not texts:
        return None
    return Rate(
        numerator=sum(1 for text in texts if score(model, text).value >= threshold),
        denominator=len(texts),
    )


def measure_ham_subsets(
    testset: Testset, pipeline: Pipeline, threshold: float
) -> tuple[SubsetHoldout, ...]:
    """四個 ham 子集各自在 holdout 上的分類器誤判率。未載入的子集 `rate` 為 `None`。

    分開量、分開報告，MUST NOT 合併 —— 頭條取上界最大者（沿用 `HAM_SUBSETS` 規則）。
    """
    measured: list[SubsetHoldout] = []
    for name in HAM_SUBSETS:
        if name not in testset.subsets:
            measured.append(SubsetHoldout(name=name, rate=None))
            continue
        texts = _holdout_texts(testset.samples(name))
        rate = pipeline_hit_rate(pipeline, texts, threshold)
        measured.append(SubsetHoldout(name=name, rate=rate))
    return tuple(measured)


def _rate_cell(rate: Rate | None) -> str:
    return "（未載入）" if rate is None else str(rate)


def _holdout_report_lines(
    ham_holdouts: Sequence[SubsetHoldout],
    scam_recall_tune: Rate,
    scam_recall_holdout: Rate | None,
    before_conv: Rate | None,
    before_threshold: float,
    after_conv: Rate | None,
    chosen_threshold: float,
    stop_reasons: Sequence[str],
) -> list[str]:
    """驗收看的三個量：scam 召回、四個 ham 子集 holdout 誤判、對話 before/after。

    **不看聚合 CV-F1** —— 閒聊在非問候特徵上與 scam、宣導文皆可分，加入它可能抬高
    聚合 F1，而真正的改變局部在問候特徵往中性移動（見 design Decision 五）。
    """
    lines = [
        "## 驗收：召回、四個 ham 子集 holdout 誤判、對話 before/after",
        "",
        "**驗收不看聚合 CV-F1。** 見 design Decision 五：閒聊是可分的第三類，"
        "會抬高聚合 F1，而真正的改變局部在問候特徵。",
        "",
        f"- **scam 召回（Cofacts scam tune，與 `add-ngram-classifier` 的 84.62% 同準則同池）"
        f"= {scam_recall_tune}**",
        f"- scam 召回（Cofacts scam holdout）= {_rate_cell(scam_recall_holdout)}",
        "",
        "### 四個 ham 子集各自的 holdout 誤判率（分類器命中率，取上界最大者為頭條）",
        "",
        "| ham 子集 | holdout 誤判率（95% Wilson） |",
        "|---|---|",
    ]
    for entry in ham_holdouts:
        lines.append(f"| {entry.name} | {_rate_cell(entry.rate)} |")
    headline = [entry for entry in ham_holdouts if entry.rate is not None]
    if headline:
        worst = max(headline, key=operator.attrgetter("rate.upper"))
        lines += [
            "",
            f"頭條（四子集中 95% Wilson 上界最大者）：**{worst.name} {worst.rate}**",
        ]
    lines += [
        "",
        "### 對話誤判 before/after（`conversation_ham` holdout，各模型於各自操作點）",
        "",
        f"- **before（現行 `ngram_model.json`，門檻 {before_threshold:.6f}）"
        f"= {_rate_cell(before_conv)}**",
        f"- **after（重訓後，門檻 {chosen_threshold:.6f}）= {_rate_cell(after_conv)}**",
        "",
    ]
    if stop_reasons:
        lines += ["### ⚠️ 停線觸發", ""]
        lines += [f"- {reason}" for reason in stop_reasons]
        lines += [
            "",
            "依 design Decision 六：不放寬約束、不寫入表；印出完整曲線，由人決定"
            '調數量、啟用 `class_weight="balanced"` 或重審語料。',
            "",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.train_ngram",
        description="在 tune 上訓練字元 n-gram 分類器並產出 ngram_model.json。",
    )
    parser.add_argument("--testset-dir", default=str(DEFAULT_TESTSET_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--psl-dir", default=str(DEFAULT_PSL_DIR))
    parser.add_argument("--blocklist-dir", default=str(DEFAULT_BLOCKLIST_DIR))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--value-decimals", type=int, default=VALUE_DECIMALS)
    args = parser.parse_args(argv)

    manifest_path = Path(args.testset_dir) / MANIFEST_FILENAME
    testset = load_testset(Path(args.data_dir), manifest_path)
    samples = tune_samples(Path(args.data_dir), manifest_path)
    texts = [normalized_text(sample) for sample in samples]
    labels = _labels(samples)
    # 三個池分開（design Decision 三）：LR 擬合於全部 tune（含對話 ham 的 tune 部分）；
    # 門檻選擇與兩個 measured 條目的量測池維持只有 Cofacts tune。
    cofacts = [
        index for index, sample in enumerate(samples) if sample.subset != CONVERSATION_HAM.name
    ]
    cofacts_labels = [labels[index] for index in cofacts]
    n_scam = sum(cofacts_labels)
    n_ham = len(cofacts_labels) - n_scam
    n_conversation = len(samples) - len(cofacts)
    print(
        f"[data] 訓練池 tune {len(samples)} 則（含對話 ham tune {n_conversation}）；"
        f"量測池 Cofacts scam {n_scam}、ham {n_ham}",
        file=sys.stderr,
    )

    cells = scan_grid(texts, labels)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    chosen_cell = choose_cell(cells)
    print(
        f"[grid] 選定 ngram_range={chosen_cell.ngram_range} "
        f"max_features={chosen_cell.max_features} F1={chosen_cell.cv_f1:.4f}",
        file=sys.stderr,
    )

    pipeline = build_pipeline(chosen_cell.ngram_range, chosen_cell.max_features)
    pipeline.fit(texts, labels)
    scores = [float(value) for value in pipeline.decision_function(texts)]
    cofacts_scores = [scores[index] for index in cofacts]

    report_dir = Path(args.report_dir)
    points = sweep_thresholds(cofacts_scores, cofacts_labels)
    try:
        chosen = choose_threshold(points)
    except ValueError as error:
        write_csv(
            report_dir / "ngram_threshold_sweep.csv",
            SWEEP_COLUMNS,
            _sweep_rows(points),
        )
        print(f"[threshold] {error}", file=sys.stderr)
        return 1

    # 驗收量測（新模型的 in-memory pipeline）：四個 ham 子集 holdout + scam 召回。
    ham_holdouts = measure_ham_subsets(testset, pipeline, chosen.threshold)
    scam_recall_holdout: Rate | None = None
    if COFACTS_SCAM.name in testset.subsets:
        scam_holdout_texts = _holdout_texts(testset.samples(COFACTS_SCAM.name))
        scam_recall_holdout = pipeline_hit_rate(pipeline, scam_holdout_texts, chosen.threshold)

    # 對話誤判 before/after：現行模型於其自身門檻、新模型於新門檻。
    conversation_holdout_texts: list[str] = []
    if CONVERSATION_HAM.name in testset.subsets:
        conversation_holdout_texts = _holdout_texts(testset.samples(CONVERSATION_HAM.name))
    after_conversation = pipeline_hit_rate(pipeline, conversation_holdout_texts, chosen.threshold)
    before_model = load_model(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.is_file() else None
    before_threshold = (
        float(before_model.manifest["threshold"]) if before_model is not None else float("nan")
    )
    before_conversation = (
        model_hit_rate(before_model, conversation_holdout_texts, before_threshold)
        if before_model is not None
        else None
    )

    # 停線（design Decision 六）：召回掉破下界，或對話誤判壓不到 ≤2%。
    stop_reasons: list[str] = []
    if chosen.p_hit_given_scam.value < RECALL_FLOOR:
        stop_reasons.append(
            f"scam 召回 {chosen.p_hit_given_scam.value:.2%} 低於下界 {RECALL_FLOOR:.0%}"
        )
    if after_conversation is not None and after_conversation.upper > CONVERSATION_UPPER_CEILING:
        stop_reasons.append(
            f"conversation_ham holdout 誤判 95% Wilson 上界 {after_conversation.upper:.2%} "
            f"超過 {CONVERSATION_UPPER_CEILING:.0%}"
        )

    holdout_lines = _holdout_report_lines(
        ham_holdouts,
        chosen.p_hit_given_scam,
        scam_recall_holdout,
        before_conversation,
        before_threshold,
        after_conversation,
        chosen.threshold,
        stop_reasons,
    )

    rule_hits = rule_hit_flags(samples, Path(args.psl_dir), Path(args.blocklist_dir))
    correlation = measure_correlation(scores, labels, rule_hits, chosen.threshold)

    write_csv(report_dir / "ngram_grid.csv", GRID_COLUMNS, _grid_rows(cells))
    write_csv(report_dir / "ngram_threshold_sweep.csv", SWEEP_COLUMNS, _sweep_rows(points))

    if stop_reasons:
        base_lines = _report_lines(
            cells,
            chosen_cell,
            points,
            chosen,
            correlation,
            _nearest_point(points, 0.0),
            (0.0, 0),
            peak_rss_bytes,
        )
        (report_dir / "report.md").write_text(
            "\n".join(base_lines + [""] + holdout_lines).rstrip() + "\n", encoding="utf-8"
        )
        for reason in stop_reasons:
            print(f"[stop] {reason}", file=sys.stderr)
        print(
            f"[stop] 停線觸發，不寫入 {args.model_out}、不更新 weights.toml；"
            f"完整曲線見 {report_dir}",
            file=sys.stderr,
        )
        return 1

    manifest = {
        "trained_on": TRAINED_ON,
        "testset_manifest_sha256": file_sha256(manifest_path),
        "n_scam": n_scam,
        "n_ham": n_ham,
        "sklearn_version": sklearn.__version__,
        "analyzer": ANALYZER,
        "ngram_range": list(chosen_cell.ngram_range),
        "max_features": chosen_cell.max_features,
        "sublinear_tf": True,
        "norm": "l2",
        "smooth_idf": True,
        "lowercase": True,
        "value_decimals": args.value_decimals,
        "threshold": round(chosen.threshold, args.value_decimals),
    }
    missing = [field for field in MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"manifest 缺少欄位：{missing}")
    document = model_document(pipeline, manifest=manifest, value_decimals=args.value_decimals)
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(serialize(document), encoding="utf-8")

    analyzer = build_analyzer(chosen_cell.ngram_range)
    table_scores = [score_from_table(document, analyzer, text) for text in texts]
    deltas = [abs(a - b) for a, b in zip(scores, table_scores, strict=True)]
    flips = sum(
        1
        for a, b in zip(scores, table_scores, strict=True)
        if (a >= chosen.threshold) != (b >= chosen.threshold)
    )

    lines = _report_lines(
        cells,
        chosen_cell,
        points,
        chosen,
        correlation,
        _nearest_point(points, 0.0),
        (max(deltas), flips),
        peak_rss_bytes,
    )
    (report_dir / "report.md").write_text(
        "\n".join(lines + [""] + holdout_lines).rstrip() + "\n", encoding="utf-8"
    )

    weight = math.log(chosen.p_hit_given_scam.value / chosen.p_hit_given_ham.value)
    print(
        f"[table] {model_path}（{model_path.stat().st_size} B，"
        f"gzip {len(gzip.compress(serialize(document).encode('utf-8')))} B）",
        file=sys.stderr,
    )
    print(
        f"[weights.toml] {NGRAM_SIGNAL}.weight_soft.value = {weight:.6f}、"
        f"p_hit_given_scam = {chosen.p_hit_given_scam.value:.6f}、"
        f"p_hit_given_ham = {chosen.p_hit_given_ham.value:.6f}；"
        f"{NGRAM_THRESHOLD}.value = {chosen.threshold:.6f}",
        file=sys.stderr,
    )
    print(f"[golden] max|A − B| = {max(deltas):.3e}、翻轉 {flips} 則", file=sys.stderr)
    print(f"[report] {report_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
