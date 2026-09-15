"""模型表格式、唯讀性與副檔名 guardrail。"""

import importlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import scam_guard.ngram as ngram
from scam_guard.ngram import DEFAULT_MODEL_PATH, load_model


def _document() -> dict:
    return json.loads(DEFAULT_MODEL_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_manifest_missing_field_fails_loudly(tmp_path: Path) -> None:
    document = _document()
    del document["manifest"]["value_decimals"]
    with pytest.raises(ValueError, match="value_decimals"):
        load_model(_write(tmp_path, document))


def test_unsupported_formula_reports_field_and_values(tmp_path: Path) -> None:
    document = _document()
    document["manifest"]["sublinear_tf"] = False
    with pytest.raises(ValueError, match=r"sublinear_tf.*False.*True"):
        load_model(_write(tmp_path, document))


def test_import_does_not_read_the_model() -> None:
    with patch.object(Path, "read_text", side_effect=AssertionError("import 不得讀檔")):
        importlib.reload(ngram)


def test_loaded_model_is_immutable() -> None:
    model = load_model()
    gram = next(iter(model.terms))
    with pytest.raises(TypeError):
        model.terms[gram] = (0.0, 0.0)  # type: ignore[index]
    with pytest.raises(TypeError):
        model.manifest["threshold"] = 0.0  # type: ignore[index]


def test_jsonl_extension_remains_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "scam_guard/tables/ngram_model.jsonl"],
        check=False,
    )
    assert result.returncode == 0
    assert (
        subprocess.run(
            ["git", "check-ignore", "-q", str(DEFAULT_MODEL_PATH)], check=False
        ).returncode
        == 1
    )
