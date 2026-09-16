"""TF-IDF baseline：訓練與評估分屬不同切分，且依賴不外流至偵測核心。"""

import ast
import tomllib
from pathlib import Path

import pytest

from tools.eval.baseline_tfidf import EVIDENCE_NOTE, train_and_evaluate
from tools.eval.dataset import Sample, split_of
from tools.eval.selectors import (
    COFACTS_HAM_AD,
    COFACTS_HAM_SUSPECTED,
    COFACTS_SCAM,
    HOLDOUT,
    TUNE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SCAM_TEXT = "您的帳戶涉及洗錢，請立即至 ATM 依指示操作解除管制，並將款項匯入監管帳戶"
HAM_TEXT = "本行提醒您，近期詐騙手法層出不窮，本行絕不會以電話要求您操作 ATM"


def balanced_samples(count: int = 120) -> list[Sample]:
    """兩個切分各湊出足夠的樣本。id 決定切分，因此逐一生成直到兩側都夠。"""
    samples: list[Sample] = []
    index = 0
    while len({sample.split for sample in samples}) < 2 or len(samples) < count:
        for subset, text in (
            (COFACTS_SCAM.name, f"{SCAM_TEXT}{index}"),
            (COFACTS_HAM_AD.name, f"{HAM_TEXT}{index}"),
            (COFACTS_HAM_SUSPECTED.name, f"{HAM_TEXT}的補充說明{index}"),
        ):
            samples.append(Sample(id=f"{subset}-{index}", text=text, subset=subset))
        index += 1
        if index > 500:
            raise AssertionError("生不出兩個切分皆非空的樣本集")
    return samples


def test_training_and_evaluation_ids_are_disjoint() -> None:
    samples = balanced_samples()
    train = {sample.id for sample in samples if sample.split == TUNE}
    test = {sample.id for sample in samples if sample.split == HOLDOUT}
    assert train and test
    assert not train & test
    assert all(split_of(identifier) == TUNE for identifier in train)


def test_baseline_reports_per_ham_subset_rates_with_intervals() -> None:
    result = train_and_evaluate(balanced_samples())
    assert result.train_size > 0
    assert set(result.false_positive_rates) <= {
        COFACTS_HAM_AD.name,
        COFACTS_HAM_SUSPECTED.name,
    }
    for rate in result.false_positive_rates.values():
        assert rate.lower <= rate.value <= rate.upper
    assert result.recall.lower <= result.recall.value <= result.recall.upper


def test_baseline_has_no_abstention_and_says_so() -> None:
    """這個 baseline 對每一則都給一個答案；比較誤判率時不提這件事是在比兩個不同的東西。"""
    result = train_and_evaluate(balanced_samples())
    assert result.abstention_rate.numerator == 0
    assert result.abstention_rate.value == 0.0


def test_an_empty_split_raises_instead_of_returning_a_meaningless_model() -> None:
    only_tune = [sample for sample in balanced_samples() if sample.split == TUNE]
    with pytest.raises(ValueError, match="兩者皆須非空"):
        train_and_evaluate(only_tune)


_BANNED_CORE_IMPORTS = frozenset({"sklearn", "numpy", "scipy", "opencc"})
"""偵測核心 `scam_guard/` 不得 import 的訓練/評估側依賴根。

`opencc` 由 add-conversation-ham 加入 —— 它只在 `tools/fetch_conversation_corpus.py`
的簡體備援路徑，列於 `[eval]` extra，與 `sklearn` 同性質。
"""


def test_the_detection_core_never_imports_sklearn() -> None:
    """`scikit-learn`（與 `opencc`）只進 `[project.optional-dependencies].eval`。"""
    for path in sorted((REPO_ROOT / "scam_guard").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not [
                    alias
                    for alias in node.names
                    if alias.name.split(".")[0] in _BANNED_CORE_IMPORTS
                ], path
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module.split(".")[0] not in _BANNED_CORE_IMPORTS, path


def test_sklearn_is_declared_only_in_the_eval_extra() -> None:
    document = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["dependencies"] == []
    extras = document["project"]["optional-dependencies"]
    assert any("scikit-learn" in item for item in extras["eval"])
    for name, items in extras.items():
        if name == "eval":
            continue
        assert not [item for item in items if "scikit-learn" in item]


def test_the_evidence_dimension_is_stated_separately_from_accuracy() -> None:
    """準確率與「能不能產生依據」是兩個獨立的維度，報告分開呈現。"""
    assert "證據座標" in EVIDENCE_NOTE
    assert "準確率" in EVIDENCE_NOTE
