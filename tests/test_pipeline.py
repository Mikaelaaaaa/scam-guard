"""`detect()` 主流程與短路規則。"""

from datetime import datetime

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document, Limits
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect
from scam_guard.types import CheckResult, Coord, Message, Request


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
    return [CheckResult(name=name, hit=True, detail="命中 165 涉詐網站清單", hard=True)]


def weak_hit(name: str) -> list[CheckResult]:
    return [CheckResult(name=name, hit=True, detail="語氣急迫", hard=False)]


def a_request() -> Request:
    return Request(messages=[Message(text="您的包裹待領取 http://a.example")])


def detail_of(verdict_checks: list[CheckResult], name: str) -> str:
    return next(r.detail for r in verdict_checks if r.name == name)


def test_empty_registry_does_not_raise() -> None:
    verdict = detect(a_request(), CheckRegistry())

    assert verdict.checks == []


def test_registry_is_supplied_by_caller() -> None:
    first = CheckRegistry()
    first.register(FakeCheck("blocklist", Stage.LOCAL, hard_hit("blocklist")))
    second = CheckRegistry()
    second.register(FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))

    assert [r.name for r in detect(a_request(), first).checks] == ["blocklist"]
    assert [r.name for r in detect(a_request(), second).checks] == ["solicit_otp"]


def test_check_without_signal_is_recorded_as_unhit() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("blocklist", Stage.LOCAL, hard_hit("blocklist")))
    registry.register(FakeCheck("evasion", Stage.LOCAL, []))
    registry.register(FakeCheck("quotation", Stage.LOCAL, []))

    checks = detect(a_request(), registry).checks

    assert [r.name for r in checks] == ["blocklist", "evasion", "quotation"]
    unhit = [r for r in checks if not r.hit]
    assert [r.name for r in unhit] == ["evasion", "quotation"]
    assert all(r.detail == NOT_HIT for r in unhit)


def test_hard_evidence_skips_expensive_checks() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("blocklist", Stage.LOCAL, hard_hit("blocklist")))
    llm = FakeCheck("llm", Stage.EXPENSIVE, weak_hit("llm"))
    registry.register(llm)

    checks = detect(a_request(), registry).checks

    assert llm.calls == 0
    assert detail_of(checks, "llm") == SKIPPED
    assert SKIPPED != NOT_HIT


def test_local_checks_all_run_despite_short_circuit() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("blocklist", Stage.LOCAL, hard_hit("blocklist")))
    later_local = FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp"))
    registry.register(later_local)
    last_local = FakeCheck("evasion", Stage.LOCAL, [])
    registry.register(last_local)

    detect(a_request(), registry)

    assert later_local.calls == 1
    assert last_local.calls == 1


def test_weak_signal_does_not_short_circuit() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))
    llm = FakeCheck("llm", Stage.EXPENSIVE, [])
    registry.register(llm)

    checks = detect(a_request(), registry).checks

    assert llm.calls == 1
    assert detail_of(checks, "llm") == NOT_HIT


def test_quotation_hit_vetoes_short_circuit_despite_hard_evidence() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("blocklist", Stage.LOCAL, hard_hit("blocklist")))
    registry.register(FakeCheck("quotation", Stage.LOCAL, weak_hit("quotation")))
    llm = FakeCheck("llm", Stage.EXPENSIVE, weak_hit("llm"))
    registry.register(llm)

    checks = detect(a_request(), registry).checks

    assert llm.calls == 1
    assert detail_of(checks, "llm") != SKIPPED


def test_quotation_hit_alone_runs_expensive_checks() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("quotation", Stage.LOCAL, weak_hit("quotation")))
    llm = FakeCheck("llm", Stage.EXPENSIVE, weak_hit("llm"))
    registry.register(llm)

    detect(a_request(), registry)

    assert llm.calls == 1


def test_short_circuit_disabled_runs_everything() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("blocklist", Stage.LOCAL, hard_hit("blocklist")))
    llm = FakeCheck("llm", Stage.EXPENSIVE, weak_hit("llm"))
    registry.register(llm)

    checks = detect(a_request(), registry, short_circuit=False).checks

    assert llm.calls == 1
    assert detail_of(checks, "llm") != SKIPPED


def test_scam_probability_is_placeholder_none() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("blocklist", Stage.LOCAL, hard_hit("blocklist")))

    verdict = detect(a_request(), registry)

    assert verdict.scam_probability is None


def test_end_to_end_with_one_local_and_one_expensive_check() -> None:
    registry = CheckRegistry()
    rule = FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp"))
    llm = FakeCheck("llm", Stage.EXPENSIVE, weak_hit("llm"))
    registry.register(rule)
    registry.register(llm)

    verdict = detect(a_request(), registry)

    assert rule.calls == 1
    assert llm.calls == 1
    assert [r.name for r in verdict.checks] == ["solicit_otp", "llm"]
    assert all(r.hit for r in verdict.checks)
    assert verdict.scam_probability is None
    assert verdict.scam_type is None


def test_disabled_check_is_not_executed() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))
    evasion = FakeCheck("evasion", Stage.LOCAL, weak_hit("evasion"))
    registry.register(evasion)
    registry.disable("evasion")

    checks = detect(a_request(), registry).checks

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

    name = "trajectory"
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
    check = FakeCheck("blocklist", Stage.LOCAL, [])
    registry.register(check)

    detect(a_request(), registry)

    assert isinstance(check.docs[0], Document)
    assert check.docs[0].sentences == ("您的包裹待領取 http://a.example",)


def test_all_checks_share_one_document_instance() -> None:
    registry = CheckRegistry()
    checks = [FakeCheck(name, Stage.LOCAL, []) for name in ("a", "b", "c")]
    for check in checks:
        registry.register(check)

    detect(a_request(), registry)

    first = checks[0].docs[0]
    assert all(check.docs[0] is first for check in checks)


def test_same_coordinate_resolves_to_the_same_sentence_across_checks() -> None:
    registry = CheckRegistry()
    first = CoordCheck("rule_a", (0, 1))
    second = CoordCheck("rule_b", (0, 1))
    registry.register(first)
    registry.register(second)

    verdict = detect(Request.from_text("在嗎？我是你朋友"), registry)

    assert first.doc is second.doc
    assert [r.evidence for r in verdict.checks] == [[(0, 1)], [(0, 1)]]
    assert first.doc.text_at((0, 1)) == "我是你朋友"


def test_default_limits_are_applied_when_not_given() -> None:
    registry = CheckRegistry()
    check = FakeCheck("blocklist", Stage.LOCAL, [])
    registry.register(check)

    detect(a_conversation(101), registry)

    assert check.docs[0].truncated is True
    assert check.docs[0].dropped_messages == 1


def test_smaller_limits_truncate_the_document_the_checks_receive() -> None:
    registry = CheckRegistry()
    check = FakeCheck("blocklist", Stage.LOCAL, [])
    registry.register(check)

    detect(a_conversation(10), registry, limits=Limits(max_messages=3))

    doc = check.docs[0]
    assert doc.truncated is True
    assert doc.dropped_messages == 7
    assert doc.coords[0] == (7, 0)


def test_blank_messages_do_not_interrupt_the_pipeline() -> None:
    registry = CheckRegistry()
    rule = FakeCheck("solicit_otp", Stage.LOCAL, [])
    llm = FakeCheck("llm", Stage.EXPENSIVE, [])
    registry.register(rule)
    registry.register(llm)

    verdict = detect(Request(messages=[Message(text="   "), Message(text="")]), registry)

    assert rule.calls == 1
    assert llm.calls == 1
    assert [r.name for r in verdict.checks] == ["solicit_otp", "llm"]
    assert rule.docs[0].sentences == ()


def test_empty_document_without_hits_yields_no_probability() -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("solicit_otp", Stage.LOCAL, []))

    verdict = detect(Request.from_text("   "), registry)

    assert verdict.scam_probability is None


def test_check_can_read_sent_at_through_the_coordinate() -> None:
    sent_at = datetime(2026, 9, 14, 10, 0, 0)
    registry = CheckRegistry()
    check = SentAtCheck()
    registry.register(check)

    detect(
        Request(messages=[Message(text="在嗎？"), Message(text="明天匯款", sent_at=sent_at)]),
        registry,
    )

    assert check.seen == [None, sent_at]


def test_end_to_end_evidence_coordinate_resolves_to_the_raw_sentence() -> None:
    registry = CheckRegistry()
    check = CoordCheck("solicit_otp", (0, 1))
    registry.register(check)

    verdict = detect(Request.from_text("您好。請提供簡訊驗證碼１２３"), registry)

    coord = verdict.checks[0].evidence[0]
    assert check.doc.raw_at(coord) == "請提供簡訊驗證碼１２３"
    assert check.doc.text_at(coord) == "請提供簡訊驗證碼123"
