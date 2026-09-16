"""訓練端與零依賴推論端讀同一份量化表時必須逐則一致。"""

import json
from pathlib import Path

import pytest

from scam_guard.ngram import DEFAULT_MODEL_PATH, load_model, score
from tools import train_ngram

# `data/testset` 是 operator-local 且 gitignored（載有受控文件文字），CI 與新 clone
# 上不存在。缺席時這條黃金測試沒有可比對的 tune 樣本，skip 而非 raise —— 與 repo
# 其餘依賴該資料的測試一致（CLAUDE.md：缺資料集是正常的，測試 skip）。
_TESTSET = Path("data/testset")
_HAS_TESTSET = _TESTSET.is_dir() and any(_TESTSET.iterdir())


@pytest.mark.skipif(not _HAS_TESTSET, reason="data/testset 未重建（operator-local，CI 上不存在）")
def test_all_tune_scores_match_the_serialized_model() -> None:
    samples = train_ngram.tune_samples(Path("data/testset"), Path("testset/manifest.json"))
    # add-conversation-ham：新 tune = 572 Cofacts scam + 824 Cofacts ham + 434 對話 ham 的
    # tune 部分（見 testset/manifest.json 的 conversation_ham.splits）。
    assert len(samples) == 1830
    texts = [train_ngram.normalized_text(sample) for sample in samples]
    labels = train_ngram._labels(samples)
    document = json.loads(DEFAULT_MODEL_PATH.read_text(encoding="utf-8"))
    model = load_model()
    analyzer = train_ngram.build_analyzer(model.ngram_range)

    runtime = [score(model, text).value for text in texts]
    training = [train_ngram.score_from_table(document, analyzer, text) for text in texts]
    maximum_runtime_delta = max(abs(a - b) for a, b in zip(runtime, training, strict=True))
    assert maximum_runtime_delta <= 1e-12

    fitted = train_ngram.build_pipeline(model.ngram_range, model.manifest["max_features"])
    fitted.fit(texts, labels)
    sklearn_scores = [float(value) for value in fitted.decision_function(texts)]
    maximum_quantisation_delta = max(
        abs(a - b) for a, b in zip(sklearn_scores, runtime, strict=True)
    )
    threshold = float(model.manifest["threshold"])
    flips = sum(
        1
        for a, b in zip(sklearn_scores, runtime, strict=True)
        if (a >= threshold) != (b >= threshold)
    )
    assert flips == 0, (
        f"量化使 {flips} 則翻轉（max |A-B|={maximum_quantisation_delta:.3e}）；"
        "請調高 value_decimals，不得調鬆 B/C 容差"
    )
