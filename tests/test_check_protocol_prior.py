"""`wants_prior` 這個非破壞性的協定擴充，以及 pipeline 的分派。

既有的兩參數檢查一行不改 —— `tests/test_check_protocol.py` 與
`tests/test_pipeline.py` 的斷言全部仍然成立，本檔只加新的。
"""

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document
from scam_guard.pipeline import SKIPPED, detect
from scam_guard.types import CheckResult, Message, Request, ScamType
from scam_guard.weights import load_weights

TABLE = load_weights()


class PlainCheck:
    """既有形狀的檢查：兩個位置參數，沒有 `wants_prior`。"""

    def __init__(self, name: str, stage: Stage, results: list[CheckResult]) -> None:
        self.name = name
        self.stage = stage
        self.results = results
        self.calls = 0

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        self.calls += 1
        return list(self.results)


class PriorCheck:
    """宣告 `wants_prior` 的檢查，記下它收到的 `prior`。"""

    wants_prior = True

    def __init__(self, name: str, stage: Stage = Stage.EXPENSIVE) -> None:
        self.name = name
        self.stage = stage
        self.seen: list[tuple[CheckResult, ...]] = []

    def __call__(
        self, req: Request, doc: Document, *, prior: tuple[CheckResult, ...]
    ) -> list[CheckResult]:
        self.seen.append(prior)
        return []


def a_request() -> Request:
    return Request(messages=[Message(text="您的包裹待領取 http://a.example")])


def weak_hit(name: str) -> list[CheckResult]:
    return [
        CheckResult(name=name, hit=True, detail="語氣急迫", scam_types=[ScamType.PHISHING_LINK])
    ]


def hard_hit(name: str) -> list[CheckResult]:
    return [CheckResult(name=name, hit=True, detail="命中 165 涉詐網站清單", hard=True)]


def test_a_plain_check_is_still_called_with_two_positional_arguments() -> None:
    registry = CheckRegistry()
    plain = PlainCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp"))
    registry.register(plain)

    detect(a_request(), registry, TABLE)

    assert plain.calls == 1


def test_a_local_check_cannot_declare_wants_prior() -> None:
    registry = CheckRegistry()

    with pytest.raises(TypeError, match="solicit_otp"):
        registry.register(PriorCheck("solicit_otp", Stage.LOCAL))


def test_an_expensive_check_receives_every_local_result() -> None:
    registry = CheckRegistry()
    registry.register(PlainCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))
    registry.register(PlainCheck("url_tld_risk", Stage.LOCAL, []))
    consumer = PriorCheck("llm_scam")
    registry.register(consumer)

    verdict = detect(a_request(), registry, TABLE)

    local_results = [result for result in verdict.checks if result.name != "llm_scam"]
    assert consumer.seen == [tuple(local_results)]


def test_the_prior_excludes_other_expensive_results() -> None:
    for order in (0, 1):
        registry = CheckRegistry()
        registry.register(PlainCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))
        consumer = PriorCheck("llm_scam")
        other = PlainCheck("domain_age", Stage.EXPENSIVE, weak_hit("domain_age"))
        for check in (consumer, other) if order else (other, consumer):
            registry.register(check)

        detect(a_request(), registry, TABLE)

        assert [result.name for result in consumer.seen[0]] == ["solicit_otp"]


def test_the_prior_is_immutable() -> None:
    registry = CheckRegistry()
    registry.register(PlainCheck("solicit_otp", Stage.LOCAL, weak_hit("solicit_otp")))
    consumer = PriorCheck("llm_scam")
    registry.register(consumer)

    detect(a_request(), registry, TABLE)

    assert isinstance(consumer.seen[0], tuple)


def test_a_short_circuit_skips_the_prior_consumer_like_any_other() -> None:
    registry = CheckRegistry()
    registry.register(PlainCheck("url_blocklist", Stage.LOCAL, hard_hit("url_blocklist")))
    consumer = PriorCheck("llm_scam")
    registry.register(consumer)

    verdict = detect(a_request(), registry, TABLE)

    assert consumer.seen == []
    skipped = [result for result in verdict.checks if result.name == "llm_scam"]
    assert [result.detail for result in skipped] == [SKIPPED]
