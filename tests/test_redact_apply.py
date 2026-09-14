"""遮蔽的套用、句子座標的不變式，以及「不存在關閉途徑」。"""

import inspect

import pytest

from scam_guard import redact
from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import SENTENCE_END, Document, build_document
from scam_guard.pipeline import detect
from scam_guard.redact import PLACEHOLDERS, RedactedText, redact_document
from scam_guard.types import CheckResult, Message, Request

SIX_SENTENCES = (
    "您好。這是第二句！第三句？第四句;第五句。身分證A123456789，手機0912345678，請盡快回覆。"
)


class FakeCheck:
    """回傳固定結果的假檢查，供短路情境使用。"""

    def __init__(self, name: str, stage: Stage, results: list[CheckResult]) -> None:
        self.name = name
        self.stage = stage
        self.results = results

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return list(self.results)


def _doc(*texts: str) -> Document:
    return build_document([Message(text=text) for text in texts])


def test_sentence_count_is_unchanged() -> None:
    doc = _doc(SIX_SENTENCES)
    assert len(doc.sentences) == 6

    assert len(redact_document(doc).sentences) == 6


def test_redaction_changes_length_but_not_index() -> None:
    doc = _doc("身分證A123456789。這句沒有個資。")
    redacted = redact_document(doc)

    assert len(redacted.sentences) == len(doc.sentences)
    assert redacted.sentences[0] != doc.sentences[0]
    assert len(redacted.sentences[0]) != len(doc.sentences[0])
    assert "A123456789" not in redacted.sentences[0]
    assert redacted.sentences[1] == doc.sentences[1]


def test_coords_are_copied_item_by_item() -> None:
    doc = _doc("第一則。含手機0912345678。", "第二則的句子。")
    redacted = redact_document(doc)

    assert list(redacted.coords) == list(doc.coords)


def test_projection_resolves_coords_without_the_document() -> None:
    """讀 log 的人只拿得到投影，座標必須在沒有 `Document` 的情況下可解析。"""
    doc = _doc("第一句。第二句含身分證A123456789。")
    redacted = redact_document(doc)
    coord = doc.coords[1]
    del doc

    assert redacted.text_at(coord) == redacted.sentences[1]
    assert "<TW_ID>" in redacted.text_at(coord)


def test_document_is_not_modified() -> None:
    doc = _doc("身分證A123456789。")
    before_sentences = list(doc.sentences)
    before_raw = list(doc.raw_sentences)

    redact_document(doc)

    assert list(doc.sentences) == before_sentences
    assert list(doc.raw_sentences) == before_raw
    assert "A123456789" in doc.sentences[0]


def test_sentence_that_is_only_a_phone_number_does_not_become_empty() -> None:
    doc = _doc("0912345678")

    redacted = redact_document(doc)

    assert redacted.sentences == ("<TW_MOBILE>",)


def test_placeholders_contain_no_sentence_separator() -> None:
    for placeholder in PLACEHOLDERS.values():
        assert placeholder
        for character in SENTENCE_END + "\n":
            assert character not in placeholder


def test_document_without_pii_is_copied_verbatim() -> None:
    doc = _doc("點這裡領取您的中獎獎金。名額有限！")

    redacted = redact_document(doc)

    assert list(redacted.sentences) == list(doc.sentences)
    assert set(redacted.counts.values()) == {0}


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="必須等長"):
        RedactedText(sentences=["一", "二"], coords=[(0, 0)], counts={})


def test_counts_are_split_by_entity_type() -> None:
    doc = _doc("手機0912345678與0987654321。身分證A123456789。")

    counts = redact_document(doc).counts

    assert dict(counts) == {
        "TW_ID": 1,
        "TW_MOBILE": 2,
        "TW_LANDLINE": 0,
        "CREDIT_CARD": 0,
    }


def test_empty_document_yields_empty_projection() -> None:
    doc = _doc("   ")
    assert doc.sentences == ()

    redacted = redact_document(doc)

    assert redacted.sentences == ()
    assert redacted.coords == ()
    assert set(redacted.counts.values()) == {0}


def test_url_in_the_same_sentence_survives_intact() -> None:
    doc = _doc("請至 https://a.example/x?id=0912345678 並回報手機0987654321。")

    redacted = redact_document(doc)

    assert "https://a.example/x?id=0912345678" in redacted.sentences[0]
    assert "<TW_MOBILE>" in redacted.sentences[0]
    assert "0987654321" not in redacted.sentences[0]


def test_detect_attaches_a_complete_projection() -> None:
    req = Request(messages=[Message(text="身分證A123456789。請盡快回覆。")])

    verdict = detect(req, CheckRegistry())

    doc = build_document(req.messages)
    assert len(verdict.redacted.sentences) == len(doc.sentences)
    assert list(verdict.redacted.coords) == list(doc.coords)


def test_projection_is_complete_even_when_short_circuited() -> None:
    """短路略過的是 `EXPENSIVE` 階段的檢查，不是投影。"""
    registry = CheckRegistry()
    registry.register(
        FakeCheck(
            "blocklist",
            Stage.LOCAL,
            [CheckResult(name="blocklist", hit=True, weight=2.5, detail="命中清單", hard=True)],
        )
    )
    registry.register(FakeCheck("llm", Stage.EXPENSIVE, []))
    req = Request(messages=[Message(text="身分證A123456789。請盡快回覆。")])

    verdict = detect(req, registry)

    assert len(verdict.redacted.sentences) == 2
    assert "<TW_ID>" in verdict.redacted.sentences[0]


def test_there_is_no_way_to_turn_redaction_off() -> None:
    """「不可關閉」本身就是形狀上的性質，所以這條測的是形狀。

    一旦有人加回一個參數、或加回一個 `NullRedactor`，這條會紅。
    """
    parameters = inspect.signature(redact_document).parameters
    assert list(parameters) == ["doc"]

    nullish = [
        name for name in dir(redact) if "null" in name.lower() and callable(getattr(redact, name))
    ]
    assert nullish == []
