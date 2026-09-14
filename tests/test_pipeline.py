"""`detect()` 主流程與短路規則。"""

import math
from datetime import datetime

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document, Limits
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect
from scam_guard.types import CheckResult, Coord, Message, Request, ScamType
from scam_guard.weights import load_weights

TABLE = load_weights()


class FakeCheck:
    """可設定回傳結果並記錄被呼叫次數的假檢查。"""

    def __init__(self, name: str, stage: Stage, results: list[CheckResult]) -> None:
        self.name = name
        self.stage = stage
        self.results = results
        self.calls = 0
        self.docs: list[Document] = []

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        self.calls += 1
        self.docs.append(doc)
        return list(self.results)


def hard_hit(name: str) -> list[CheckResult]:
    return [
        CheckResult(
            name=name,
            hit=True,
            detail="命中 165 涉詐網站清單",
            scam_types=[ScamType.PHISHING_LINK],
            hard=True,
        )
    ]


def weak_hit(name: str) -> list[CheckResult]:
    return [
        CheckResult(
            name=name,
            hit=True,
            detail="語氣急迫",
            scam_types=[ScamType.PHISHING_LINK],
            hard=False,
        )
    ]


def a_request() -> Request:
    return Request(messages=[Message(text="您的包裹待領取 http://a.example")])


def detail_of(verdict_checks: list[CheckResult], name: str) -> str:
    return next(r.detail for r in verdict_checks if r.name == name)


def test_empty_registry_does_not_raise() -> None:
    verdict = detect(a_request(), CheckRegistry(), TABLE)

    assert verdict.checks == []


def test_registry_is_supplied_by_caller() -> None:
    first = CheckRegistry()
    first.register(FakeCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))
    second = CheckRegistry()
    second.register(FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))

    assert [r.name for r in detect(a_request(), first, TABLE).checks] == ["url_blocklist"]
    assert [r.name for r in detect(a_request(), second, TABLE).checks] == ["solicit_otp"]


def test_check_without_signal_is_recorded_as_unhit() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))
    registry.register(FakeCheck("evasion_invisible", Stage.LOCAL, []))
    registry.register(FakeCheck("quotation", Stage.LOCAL, []))

    checks = detect(a_request(), registry, TABLE).checks

    assert [r.name for r in checks] == ["url_blocklist", "evasion_invisible", "quotation"]
    unhit = [r for r in checks if not r.hit]
    assert [r.name for r in unhit] == ["evasion_invisible", "quotation"]
    assert all(r.detail == NOT_HIT for r in unhit)


def test_hard_evidence_skips_expensive_checks() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))
    llm = FakeCheck("domain_age", Stage.EXPENSIVE, weak_hit("domain_age"))
    registry.register(llm)

    checks = detect(a_request(), registry, TABLE).checks

    assert llm.calls == 0
    assert detail_of(checks, "domain_age") == SKIPPED
    assert SKIPPED != NOT_HIT


def test_local_checks_all_run_despite_short_circuit() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))
    later_local = FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp"))
    registry.register(later_local)
    last_local = FakeCheck("evasion_invisible", Stage.LOCAL, [])
    registry.register(last_local)

    detect(a_request(), registry, TABLE)

    assert later_local.calls == 1
    assert last_local.calls == 1


def test_weak_signal_does_not_short_circuit() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))
    llm = FakeCheck("domain_age", Stage.EXPENSIVE, [])
    registry.register(llm)

    checks = detect(a_request(), registry, TABLE).checks

    assert llm.calls == 1
    assert detail_of(checks, "domain_age") == NOT_HIT


def test_quotation_hit_vetoes_short_circuit_despite_hard_evidence() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))
    registry.register(FakeCheck("quotation", Stage.LOCAL, weak_hit("quotation")))
    llm = FakeCheck("domain_age", Stage.EXPENSIVE, weak_hit("domain_age"))
    registry.register(llm)

    checks = detect(a_request(), registry, TABLE).checks

    assert llm.calls == 1
    assert detail_of(checks, "domain_age") != SKIPPED


def test_quotation_hit_alone_runs_expensive_checks() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("quotation", Stage.LOCAL, weak_hit("quotation")))
    llm = FakeCheck("domain_age", Stage.EXPENSIVE, weak_hit("domain_age"))
    registry.register(llm)

    detect(a_request(), registry, TABLE)

    assert llm.calls == 1


def test_short_circuit_disabled_runs_everything() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))
    llm = FakeCheck("domain_age", Stage.EXPENSIVE, weak_hit("domain_age"))
    registry.register(llm)

    checks = detect(a_request(), registry, TABLE, short_circuit=False).checks

    assert llm.calls == 1
    assert detail_of(checks, "domain_age") != SKIPPED


def test_scam_probability_comes_from_the_score() -> None:
    """硬證據命中 → 分數 2.5 → 機率 sigmoid(2.5)。不再是佔位的 `None`。"""
    registry = CheckRegistry()
    registry.register(FakeCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))

    verdict = detect(a_request(), registry, TABLE)

    assert verdict.scam_probability == pytest.approx(1 / (1 + math.exp(-2.5)))


def test_end_to_end_with_one_local_and_one_expensive_check() -> None:
    registry = CheckRegistry()
    rule = FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp"))
    llm = FakeCheck("domain_age", Stage.EXPENSIVE, weak_hit("domain_age"))
    registry.register(rule)
    registry.register(llm)

    verdict = detect(a_request(), registry, TABLE)

    assert rule.calls == 1
    assert llm.calls == 1
    assert [r.name for r in verdict.checks] == ["solicit_otp", "domain_age"]
    assert all(r.hit for r in verdict.checks)
    assert verdict.scam_probability > 0.5
    assert verdict.scam_type is ScamType.PHISHING_LINK


def test_disabled_check_is_not_executed() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))
    evasion = FakeCheck("evasion_invisible", Stage.LOCAL, weak_hit("evasion_invisible"))
    registry.register(evasion)
    registry.disable("evasion_invisible")

    checks = detect(a_request(), registry, TABLE).checks

    assert evasion.calls == 0
    assert [r.name for r in checks] == ["solicit_otp"]


class CoordCheck:
    """回報指定座標的假檢查，同時保留它當時看到的 `Document`。"""

    stage = Stage.LOCAL

    def __init__(self, name: str, coord: Coord) -> None:
        self.name = name
        self.coord = coord
        self.doc: Document | None = None

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        self.doc = doc
        return [
            CheckResult(
                name=self.name,
                hit=True,
                detail=f"命中：{doc.text_at(self.coord)}",
                evidence=[self.coord],
            )
        ]


class SentAtCheck:
    """由座標回查 `req.messages` 取得該則訊息的時間。"""

    name = "relationship_building"
    stage = Stage.LOCAL

    def __init__(self) -> None:
        self.seen: list[datetime | None] = []

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        self.seen = [req.messages[message_index].sent_at for message_index, _ in doc.coords]
        return []


def a_conversation(count: int) -> Request:
    return Request(messages=[Message(text=f"第 {i} 則訊息") for i in range(count)])


def test_checks_receive_a_document_not_none() -> None:
    registry = CheckRegistry()
    check = FakeCheck("url_blocklist", Stage.LOCAL, [])
    registry.register(check)

    detect(a_request(), registry, TABLE)

    assert isinstance(check.docs[0], Document)
    assert check.docs[0].sentences == ("您的包裹待領取 http://a.example",)


def test_all_checks_share_one_document_instance() -> None:
    registry = CheckRegistry()
    names = ("safe_account", "atm_operation", "secrecy_demand")
    checks = [FakeCheck(name, Stage.LOCAL, []) for name in names]
    for check in checks:
        registry.register(check)

    detect(a_request(), registry, TABLE)

    first = checks[0].docs[0]
    assert all(check.docs[0] is first for check in checks)


def test_same_coordinate_resolves_to_the_same_sentence_across_checks() -> None:
    registry = CheckRegistry()
    first = CoordCheck("safe_account", (0, 1))
    second = CoordCheck("atm_operation", (0, 1))
    registry.register(first)
    registry.register(second)

    verdict = detect(Request.from_text("在嗎？我是你朋友"), registry, TABLE)

    assert first.doc is second.doc
    assert [r.evidence for r in verdict.checks] == [[(0, 1)], [(0, 1)]]
    assert first.doc.text_at((0, 1)) == "我是你朋友"


def test_default_limits_are_applied_when_not_given() -> None:
    registry = CheckRegistry()
    check = FakeCheck("url_blocklist", Stage.LOCAL, [])
    registry.register(check)

    detect(a_conversation(101), registry, TABLE)

    assert check.docs[0].truncated is True
    assert check.docs[0].dropped_messages == 1


def test_smaller_limits_truncate_the_document_the_checks_receive() -> None:
    registry = CheckRegistry()
    check = FakeCheck("url_blocklist", Stage.LOCAL, [])
    registry.register(check)

    detect(a_conversation(10), registry, TABLE, limits=Limits(max_messages=3))

    doc = check.docs[0]
    assert doc.truncated is True
    assert doc.dropped_messages == 7
    assert doc.coords[0] == (7, 0)


def test_blank_messages_do_not_interrupt_the_pipeline() -> None:
    registry = CheckRegistry()
    rule = FakeCheck("solicit_otp", Stage.LOCAL, [])
    llm = FakeCheck("domain_age", Stage.EXPENSIVE, [])
    registry.register(rule)
    registry.register(llm)

    verdict = detect(Request(messages=[Message(text="   "), Message(text="")]), registry, TABLE)

    assert rule.calls == 1
    assert llm.calls == 1
    assert [r.name for r in verdict.checks] == ["solicit_otp", "domain_age"]
    assert rule.docs[0].sentences == ()


def test_empty_document_without_hits_yields_no_probability() -> None:
    """無訊號 → 分數 0、機率 0.5，但信心 0.05 低於門檻 → 輸出「無法判定」。

    那個 0.5 不傳達任何資訊，而把它換成「無法判定」是**信心**的工作。
    拒答的理由是依據不足，不是計分尚未實作。
    """
    registry = CheckRegistry()
    registry.register(FakeCheck("solicit_otp", Stage.LOCAL, []))

    verdict = detect(Request.from_text("   "), registry, TABLE)

    assert verdict.scam_probability is None
    assert verdict.confidence == TABLE.threshold("base_no_hit")


def test_check_can_read_sent_at_through_the_coordinate() -> None:
    sent_at = datetime(2026, 9, 14, 10, 0, 0)
    registry = CheckRegistry()
    check = SentAtCheck()
    registry.register(check)

    detect(
        Request(messages=[Message(text="在嗎？"), Message(text="明天匯款", sent_at=sent_at)]),
        registry,
        TABLE,
    )

    assert check.seen == [None, sent_at]


def test_end_to_end_evidence_coordinate_resolves_to_the_raw_sentence() -> None:
    registry = CheckRegistry()
    check = CoordCheck("solicit_otp", (0, 1))
    registry.register(check)

    verdict = detect(Request.from_text("您好。請提供簡訊驗證碼１２３"), registry, TABLE)

    coord = verdict.checks[0].evidence[0]
    assert check.doc.raw_at(coord) == "請提供簡訊驗證碼１２３"
    assert check.doc.text_at(coord) == "請提供簡訊驗證碼123"
