"""以**真正的 llama.cpp grammar 引擎**檢驗 `build_grammar()` 接受什麼、拒絕什麼。

⚠️ **`llama_cpp.LlamaGrammar.from_string()` 在 0.3.35 不做任何解析** ——
它只是把字串存進一個屬性（`LlamaGrammar.__init__` 的全部內容就是
`self._grammar = _grammar`），真正的解析發生在 C 層的
`llama_sampler_init_grammar()`，而那需要一個 vocab，也就是需要模型檔。
所以本檔以 `llama_sampler_init_grammar()` + 逐 token 的
`llama_sampler_apply()` / `llama_sampler_accept()` 判定接受與否 ——
這正是解碼時真正發生的事，而不是它的一個近似。

代價：本檔需要一個 806 MB 的 GGUF，因此以 `SCAM_GUARD_GGUF` 指向本機檔案，
未設定時整檔跳過。**兩層引號的端到端斷言因此另有一份在
`tests/test_llm_schema.py`，那一份無條件執行。**
"""

import ctypes
import json
import os
from collections.abc import Iterable

import pytest

llama_cpp = pytest.importorskip("llama_cpp")

from scam_guard.llm.schema import MAX_EVIDENCE_IDS, build_grammar  # noqa: E402
from scam_guard.types import Coord, ScamType  # noqa: E402

GGUF_PATH = os.environ.get("SCAM_GUARD_GGUF")

pytestmark = pytest.mark.skipif(
    not GGUF_PATH,
    reason="需要一份本機 GGUF：設定 SCAM_GUARD_GGUF 指向 gemma-3-1b-it-Q4_K_M.gguf",
)

NEGATIVE_INFINITY = float("-inf")


class GrammarAcceptance:
    """對一段文字問「grammar 會不會讓模型產生它」。

    以類別而非 closure 持有 vocab 與 tokenizer —— 兩者是狀態。

    判定方式與解碼時相同：把候選集合縮成一個 token，套用 grammar sampler，
    被禁止的 token 其 logit 會被設為 `-inf`。全部 token 都通過之後再問一次
    EOG 是否被允許 —— grammar 只在某個堆疊為空（即到達接受狀態）時允許結束，
    所以這一問區分的正是「合法前綴」與「完整物件」。
    """

    def __init__(self, model_path: str) -> None:
        self._llama = llama_cpp.Llama(model_path=model_path, vocab_only=True, verbose=False)
        self._vocab = llama_cpp.llama_model_get_vocab(self._llama.model)
        self._eos = llama_cpp.llama_vocab_eos(self._vocab)

    def _allowed(self, sampler: ctypes.c_void_p, token: int) -> bool:
        data = (llama_cpp.llama_token_data * 1)()
        data[0].id = token
        data[0].logit = 0.0
        data[0].p = 0.0
        candidates = llama_cpp.llama_token_data_array(data=data, size=1, selected=-1, sorted=False)
        llama_cpp.llama_sampler_apply(sampler, ctypes.byref(candidates))
        return data[0].logit != NEGATIVE_INFINITY

    def accepts(self, grammar: str, text: str) -> bool:
        sampler = llama_cpp.llama_sampler_init_grammar(self._vocab, grammar.encode(), b"root")
        if not sampler:
            raise ValueError("llama.cpp 無法解析這段 grammar")
        try:
            for token in self._llama.tokenize(text.encode(), add_bos=False, special=False):
                if not self._allowed(sampler, token):
                    return False
                llama_cpp.llama_sampler_accept(sampler, token)
            return self._allowed(sampler, self._eos)
        finally:
            llama_cpp.llama_sampler_free(sampler)


@pytest.fixture(scope="module")
def acceptance() -> GrammarAcceptance:
    return GrammarAcceptance(GGUF_PATH)


@pytest.fixture(scope="module")
def grammar() -> str:
    return build_grammar()


def an_output(
    category: str | None = "假投資",
    label: str = "完整詐騙話術",
    ids: Iterable[Coord] = ((0, 0),),
    notes: str = "第 1 句自稱郵局",
) -> str:
    """一個符合 schema 的輸出，鍵序與 `FIELD_NAMES` 相同、不含空白。"""
    return json.dumps(
        {
            "analysis_notes": notes,
            "evidence_sentence_ids": [list(coord) for coord in ids],
            "category_165": category,
            "label": label,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_grammar_parses(acceptance: GrammarAcceptance, grammar: str) -> None:
    assert acceptance.accepts(grammar, an_output())


def test_every_scam_type_is_accepted_and_round_trips(
    acceptance: GrammarAcceptance, grammar: str
) -> None:
    """兩層引號唯一擋得住錯誤的形式：grammar 接受 + `json.loads` + `ScamType()`。"""
    for scam_type in ScamType:
        raw = an_output(category=scam_type.value)
        assert acceptance.accepts(grammar, raw), scam_type.name
        assert ScamType(json.loads(raw)["category_165"]) is scam_type


def test_null_category_is_accepted(acceptance: GrammarAcceptance, grammar: str) -> None:
    assert acceptance.accepts(grammar, an_output(category=None))


def test_the_maximum_number_of_coordinates_is_accepted(
    acceptance: GrammarAcceptance, grammar: str
) -> None:
    ids = [(0, index) for index in range(MAX_EVIDENCE_IDS)]
    assert acceptance.accepts(grammar, an_output(ids=ids))


def test_one_coordinate_too_many_is_rejected(acceptance: GrammarAcceptance, grammar: str) -> None:
    ids = [(0, index) for index in range(MAX_EVIDENCE_IDS + 1)]
    assert not acceptance.accepts(grammar, an_output(ids=ids))


def test_empty_coordinates_are_accepted(acceptance: GrammarAcceptance, grammar: str) -> None:
    """語法上合法，語意上是否合法由 `scam_guard.llm.validate` 判定。"""
    assert acceptance.accepts(grammar, an_output(ids=()))


def test_duplicate_coordinates_are_accepted(acceptance: GrammarAcceptance, grammar: str) -> None:
    """GBNF 表達不了「不重複」；去重屬輸出驗證層。"""
    assert acceptance.accepts(grammar, an_output(ids=((3, 0), (3, 0))))


def test_a_newline_in_the_notes_is_rejected(acceptance: GrammarAcceptance, grammar: str) -> None:
    raw = an_output().replace("第 1 句自稱郵局", "第 1 句\\n自稱郵局")
    assert not acceptance.accepts(grammar, raw)


def test_a_quote_in_the_notes_is_rejected(acceptance: GrammarAcceptance, grammar: str) -> None:
    raw = an_output().replace("第 1 句自稱郵局", '第 1 句\\"自稱郵局')
    assert not acceptance.accepts(grammar, raw)


def test_reordered_keys_are_rejected(acceptance: GrammarAcceptance, grammar: str) -> None:
    raw = (
        '{"label":"完整詐騙話術","analysis_notes":"x",'
        '"evidence_sentence_ids":[[0,0]],"category_165":"假投資"}'
    )
    assert not acceptance.accepts(grammar, raw)


def test_an_extra_key_is_rejected(acceptance: GrammarAcceptance, grammar: str) -> None:
    assert not acceptance.accepts(grammar, an_output()[:-1] + ',"confidence":0.9}')


def test_a_truncated_output_is_not_complete(acceptance: GrammarAcceptance, grammar: str) -> None:
    """grammar 不生效的三條路徑產出的是合法前綴，而前綴不是完整物件。"""
    assert not acceptance.accepts(grammar, an_output()[:-1])


def test_an_unknown_category_is_rejected(acceptance: GrammarAcceptance, grammar: str) -> None:
    assert not acceptance.accepts(grammar, an_output(category="網路購物"))


def test_an_unknown_label_is_rejected(acceptance: GrammarAcceptance, grammar: str) -> None:
    assert not acceptance.accepts(grammar, an_output(label="可疑"))
