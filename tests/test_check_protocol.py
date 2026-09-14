"""`Check` 協定與 `CheckRegistry` 的行為。"""

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.types import CheckResult, Message, Request


def solicit_otp(req: Request, doc: object) -> list[CheckResult]:
    """函式形式的檢查 —— 不繼承任何基底類別。"""
    return [CheckResult(name="solicit_otp", hit=True, weight=1.5, detail="第 2 句索取驗證碼")]


solicit_otp.name = "solicit_otp"  # type: ignore[attr-defined]
solicit_otp.stage = Stage.LOCAL  # type: ignore[attr-defined]


def no_signal(req: Request, doc: object) -> list[CheckResult]:
    """未命中的檢查回傳空陣列，不回傳佔位結果。"""
    return []


no_signal.name = "no_signal"  # type: ignore[attr-defined]
no_signal.stage = Stage.LOCAL  # type: ignore[attr-defined]


class UrlCheck:
    """類別形式的檢查 —— 一則訊息含三個 URL 時各產一筆結果。"""

    name = "url"
    stage = Stage.LOCAL

    def __call__(self, req: Request, doc: object) -> list[CheckResult]:
        return [
            CheckResult(
                name="url",
                hit=True,
                weight=2.5,
                detail=f"命中 165 涉詐網站清單：{host}",
                evidence=[i],
                hard=True,
            )
            for i, host in enumerate(("a.example", "b.example", "c.example"))
        ]


class TimingOutCheck:
    """依賴外部服務的檢查：自行吞例外並記錄，回傳空陣列。"""

    name = "domain_age"
    stage = Stage.EXPENSIVE

    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, req: Request, doc: object) -> list[CheckResult]:
        try:
            raise TimeoutError("RDAP 查詢逾時")
        except TimeoutError as exc:
            self.failures.append(str(exc))
            return []


def a_request() -> Request:
    return Request(messages=[Message(text="請提供簡訊驗證碼")])


def test_function_check_can_be_registered_and_listed() -> None:
    registry = CheckRegistry()
    registry.register(solicit_otp)

    assert [c.name for c in registry.enabled()] == ["solicit_otp"]


def test_class_check_can_be_registered() -> None:
    registry = CheckRegistry()
    registry.register(UrlCheck())

    assert [c.name for c in registry.enabled()] == ["url"]


def test_registries_are_isolated() -> None:
    first = CheckRegistry()
    second = CheckRegistry()
    first.register(solicit_otp)
    second.register(UrlCheck())

    assert [c.name for c in first.enabled()] == ["solicit_otp"]
    assert [c.name for c in second.enabled()] == ["url"]


def test_register_rejects_object_without_name() -> None:
    registry = CheckRegistry()

    with pytest.raises(TypeError, match="name"):
        registry.register(lambda req, doc: [])  # type: ignore[arg-type]


def test_register_rejects_non_callable() -> None:
    class NotCallable:
        name = "not_callable"
        stage = Stage.LOCAL

    registry = CheckRegistry()

    with pytest.raises(TypeError, match="可呼叫"):
        registry.register(NotCallable())  # type: ignore[arg-type]


def test_register_rejects_missing_or_invalid_stage() -> None:
    class NoStage:
        name = "no_stage"

        def __call__(self, req: Request, doc: object) -> list[CheckResult]:
            return []

    class BadStage(NoStage):
        name = "bad_stage"
        stage = "local"

    registry = CheckRegistry()

    with pytest.raises(TypeError, match="stage"):
        registry.register(NoStage())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="stage"):
        registry.register(BadStage())  # type: ignore[arg-type]


def test_register_rejects_duplicate_name() -> None:
    registry = CheckRegistry()
    registry.register(UrlCheck())

    with pytest.raises(ValueError, match="重複"):
        registry.register(UrlCheck())


def test_disable_then_enable() -> None:
    registry = CheckRegistry()
    registry.register(solicit_otp)
    registry.register(UrlCheck())
    registry.register(no_signal)

    registry.disable("url")
    assert [c.name for c in registry.enabled()] == ["solicit_otp", "no_signal"]

    registry.enable("url")
    assert [c.name for c in registry.enabled()] == ["solicit_otp", "url", "no_signal"]


def test_disable_unknown_name_raises() -> None:
    registry = CheckRegistry()
    registry.register(solicit_otp)

    with pytest.raises(KeyError, match="無此檢查"):
        registry.disable("不存在的檢查")


def test_enable_unknown_name_raises() -> None:
    registry = CheckRegistry()

    with pytest.raises(KeyError, match="無此檢查"):
        registry.enable("不存在的檢查")


def test_check_may_return_multiple_results() -> None:
    results = UrlCheck()(a_request(), None)

    assert len(results) == 3
    assert [r.detail for r in results] == [
        "命中 165 涉詐網站清單：a.example",
        "命中 165 涉詐網站清單：b.example",
        "命中 165 涉詐網站清單：c.example",
    ]
    assert [r.evidence for r in results] == [[0], [1], [2]]


def test_check_without_signal_returns_empty_list() -> None:
    assert no_signal(a_request(), None) == []


def test_external_failure_returns_empty_list_and_is_recorded() -> None:
    check = TimingOutCheck()

    assert check(a_request(), None) == []
    assert check.failures == ["RDAP 查詢逾時"]
