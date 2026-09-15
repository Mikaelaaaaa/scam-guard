"""四個欄位的常數、序關係，與 GBNF grammar 的產生。

本檔**不需要** `llama_cpp`。用真正的 llama.cpp grammar 引擎檢驗接受 / 拒絕的
測試在 `tests/test_llm_grammar.py`。

兩層引號的端到端檢驗在本檔（`test_every_category_branch_survives_json_loads_and_scam_type`）
而不是只在那一檔 —— 它是唯一擋得住「少一層 / 多一層引號」的斷言，
而那一檔需要一個 806 MB 的模型檔才能跑。
"""

import ast
import json
import re
from pathlib import Path

import pytest

from scam_guard.llm import schema
from scam_guard.llm.schema import (
    FIELD_NAMES,
    LABELS,
    MAX_EVIDENCE_IDS,
    MAX_NOTES_CHARS,
    build_grammar,
    label_rank,
)
from scam_guard.render import SPECULATIVE_TERMS, VERDICT_CLAIMS
from scam_guard.types import ScamType

# 兩張禁用詞表，**複製自 `scam_guard/render.py`**（`SPECULATIVE_TERMS` 與
# `VERDICT_CLAIMS`）。複製而非直接引用，是為了讓「有人往表裡加一個詞」這件事
# 在這裡變成一個失敗而不是一個安靜通過的測試 —— 下方另有一條斷言複本與來源相等。
SPECULATIVE_TERMS_COPY = frozenset(
    {"可疑", "危險", "不明", "很可能", "應該是", "一定是", "肯定", "小心", "注意"}
)
VERDICT_CLAIMS_COPY = frozenset({"是詐騙", "為詐騙", "詐騙訊息", "確定是"})

SCHEMA_SOURCE = Path(schema.__file__).read_text(encoding="utf-8")

BANNED_IMPORTS = ("llama_cpp", "torch", "transformers", "llm_runtime", "scam_guard.pipeline")


def branches(rule: str) -> list[str]:
    """從產生的 grammar 中取出某條 rule 的替代式，未去逸出。"""
    for line in build_grammar().splitlines():
        name, _, body = line.partition("::=")
        if name.strip() == rule:
            return [alternative.strip() for alternative in body.split("|")]
    raise AssertionError(f"grammar 中沒有 rule {rule!r}")


def unescape_terminal(terminal: str) -> str:
    """把一個 GBNF 字串字面還原成它所表示的字元序列。

    GBNF 的終端符號寫在雙引號內，內容為 UTF-8 字面；需要逸出的只有 `"` 與 `\\`。
    """
    assert terminal.startswith('"') and terminal.endswith('"'), terminal
    return terminal[1:-1].replace('\\"', '"').replace("\\\\", "\\")


def imported_modules(source: str) -> set[str]:
    """原始碼中**實際 import 的模組名稱**。

    以 `ast` 而非 `grep` 判定：本模組的 docstring 裡寫著「MUST NOT import
    推論引擎」這類說明，而一條會被自己的說明文字擋下的 grep 只會逼人刪掉說明。
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_field_names_are_fixed_and_the_verdict_comes_last() -> None:
    assert FIELD_NAMES == (
        "analysis_notes",
        "evidence_sentence_ids",
        "category_165",
        "label",
    )
    assert FIELD_NAMES.index("analysis_notes") < FIELD_NAMES.index("label")


def test_labels_are_three_and_the_order_comes_from_the_index() -> None:
    assert LABELS == ("無詐騙話術", "部分詐騙話術", "完整詐騙話術")
    assert [label_rank(label) for label in LABELS] == [0, 1, 2]
    # 序關係 MUST NOT 由字串比較推導：Unicode 碼位序與此處要的序無關。
    assert label_rank("完整詐騙話術") > label_rank("部分詐騙話術")
    assert ("完整詐騙話術" > "部分詐騙話術") != (
        label_rank("完整詐騙話術") > label_rank("部分詐騙話術")
    )


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        label_rank("可疑")


def test_copied_banned_word_tables_match_the_source() -> None:
    assert SPECULATIVE_TERMS_COPY == SPECULATIVE_TERMS
    assert VERDICT_CLAIMS_COPY == VERDICT_CLAIMS


def test_labels_hit_neither_banned_word_table() -> None:
    for label in LABELS:
        assert not [term for term in SPECULATIVE_TERMS_COPY if term in label]
        assert not [claim for claim in VERDICT_CLAIMS_COPY if claim in label]


def test_category_branches_equal_the_scam_type_values() -> None:
    values = {
        json.loads(unescape_terminal(branch)) for branch in branches("catval") if branch != '"null"'
    }
    assert values == {scam_type.value for scam_type in ScamType}


def test_category_branches_count_is_eighteen_plus_null() -> None:
    assert len(branches("catval")) == len(ScamType) + 1


def test_label_branches_come_from_the_labels_constant() -> None:
    assert {json.loads(unescape_terminal(branch)) for branch in branches("labval")} == set(LABELS)


def test_every_category_branch_survives_json_loads_and_scam_type() -> None:
    """兩層引號的端到端檢驗 —— 唯一擋得住這類錯誤的斷言。

    少一層 → 解逸出後是裸中文，`json.loads` 在下一行拋 `JSONDecodeError`。
    多一層 → `json.loads` 得到一個**帶引號**的字串，`ScamType()` 拋 `ValueError`。
    純字串比對兩者都擋不住，因為錯的 grammar 看起來也很像對的。
    """
    for branch in branches("catval"):
        if branch == '"null"':
            assert json.loads(unescape_terminal(branch)) is None
            continue
        document = json.loads('{"category_165":' + unescape_terminal(branch) + "}")
        assert ScamType(document["category_165"]) in ScamType


def test_slash_and_parentheses_appear_unescaped() -> None:
    grammar = build_grammar()
    assert "假檢警/假冒公務機關" in grammar
    assert "騙取金融帳戶(卡片)" in grammar


def test_grammar_has_exactly_two_repetition_bounds_and_no_bare_star() -> None:
    """防的是 `{m,n}` 不受支援時被改成無上界的 `*`。

    無上界的重複讓模型可以輸出數百個座標，而那是一次 decode 成本的爆炸。
    """
    grammar = build_grammar()
    assert "*" not in grammar
    assert set(re.findall(r"\{(\d+),(\d+)\}", grammar)) == {
        ("0", str(MAX_NOTES_CHARS)),
        ("0", str(MAX_EVIDENCE_IDS - 1)),
    }


def test_schema_imports_no_runtime_or_framework() -> None:
    assert imported_modules(SCHEMA_SOURCE).isdisjoint(BANNED_IMPORTS)


def test_tables_directory_has_no_grammar_file() -> None:
    tables = Path(schema.__file__).parent.parent / "tables"
    assert not list(tables.glob("*.gbnf"))
