"""`detect()` 主流程與短路規則。"""

from scam_guard.check import CheckRegistry, Stage
from scam_guard.pipeline import NOT_HIT, SKIPPED, detect
from scam_guard.types import CheckResult, Message, Request


class FakeCheck:
    """可設定回傳結果並記錄被呼叫次數的假檢查。"""

    def __init__(self, name: str, stage: Stage, results: list[CheckResult]) -> None:
        self.name = name
        self.stage = stage
        self.results = results
        self.calls = 0

    def __call__(self, req: Request, doc: object) -> list[CheckResult]:
        self.calls += 1
        return list(self.results)


def hard_hit(name: str) -> list[CheckResult]:
    return [CheckResult(name=name, hit=True, weight=2.5, detail="命中 165 涉詐網站清單", hard=True)]


def weak_hit(name: str) -> list[CheckResult]:
    return [CheckResult(name=name, hit=True, weight=0.6, detail="語氣急迫", hard=False)]


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
    assert all(r.detail == NOT_HIT and r.weight == 0.0 for r in unhit)


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
