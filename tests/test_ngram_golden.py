"""訓練端與零依賴推論端讀同一份量化表時必須逐則一致。"""

import json
from pathlib import Path

from scam_guard.ngram import DEFAULT_MODEL_PATH, load_model, score
from tools import train_ngram


def test_all_tune_scores_match_the_serialized_model() -> None:
    samples = train_ngram.tune_samples(Path("data/testset"), Path("testset/manifest.json"))
    assert len(samples) == 1396
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
