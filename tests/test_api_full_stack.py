"""API 四層組裝的回歸測試。

快照、LLM runtime 與 extra 探測全部在程序內構造或 monkeypatch；本檔不讀
operator-local ``data/``、不連網，也不需要 GGUF 或 llama-cpp-python。
"""

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.app
from api.schema import CheckResponse
from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry, Stage
from scam_guard.llm.check import LlmCheck
from scam_guard.llm.validate import SCAM_SIGNAL
from scam_guard.pipeline import SKIPPED, detect
from scam_guard.types import CheckResult, Message, Request
from scam_guard.url import PublicSuffixList

PSL_TEXT = "// ===BEGIN ICANN DOMAINS===\ncom\n// ===END ICANN DOMAINS===\n"
URL_CHECK_NAMES = frozenset(
    {"url_blocklist", "url_shortener", "url_tld_risk", "url_host_shape", "url_brand"}
)


class _StubRuntime:
    """不載模型；建構 ``LlmCheck`` 時 runtime 不會被呼叫。"""

    def generate(self, prompt: str, grammar: str, *, deadline_s: float) -> str:
        return ""


class _FakeLlamaModule:
    @staticmethod
    def LlamaCppRuntime(path: Path) -> _StubRuntime:  # noqa: N802
        return _StubRuntime()


class _PlainCheck:
    def __init__(self, name: str, results: list[CheckResult]) -> None:
        self.name = name
        self.stage = Stage.LOCAL
        self.results = results

    def __call__(self, req: Request, doc: object) -> list[CheckResult]:
        return list(self.results)


class _TrackingLlmCheck:
    name = SCAM_SIGNAL
    stage = Stage.EXPENSIVE
    wants_prior = True

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, req: Request, doc: object, *, prior: tuple[CheckResult, ...]
    ) -> list[CheckResult]:
        self.calls += 1
        return []


def _psl() -> PublicSuffixList:
    return PublicSuffixList.parse(PSL_TEXT)


def _snapshots() -> tuple[PublicSuffixList, BlocklistStore, RankAllowlist]:
    psl = _psl()
    return psl, BlocklistStore((), {"sources": {}}, psl), RankAllowlist({}, {})


def _llm_check() -> LlmCheck:
    return LlmCheck(
        runtime=_StubRuntime(),
        counter=api.app.LlmOutcomeCounter(),
        table=api.app.TABLE,
        budget=api.app.DEFAULT_BUDGET,
        deadline_s=api.app.LLM_DEADLINE_S,
    )


def _names(registry: CheckRegistry) -> set[str]:
    return {check.name for check in registry.enabled()}


def test_complete_inputs_register_classifier_urls_and_llm() -> None:
    psl, store, allowlist = _snapshots()
    registry = api.app.build_registry(psl, store, allowlist, _llm_check())
    names = _names(registry)

    assert {"ngram_classifier", SCAM_SIGNAL} | URL_CHECK_NAMES <= names
    assert api.app.build_unregistered("", llm_loaded=True) == (
        ("domain_age", "未注入外部查詢解析器"),
    )
    api.app.TABLE.validate_against(registry)


def test_missing_psl_skips_the_whole_url_layer_and_health_stays_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api.app, "PSL_DIR", tmp_path / "missing-psl")
    assert api.app.load_psl() is None

    registry = api.app.build_registry(None, None, None, None)
    names = _names(registry)
    assert not (URL_CHECK_NAMES & names)
    assert {"ngram_classifier", "quotation"} <= names

    monkeypatch.setattr(api.app, "REGISTRY", registry)
    monkeypatch.setattr(api.app, "BLOCKLIST_STORE", None)
    response = TestClient(api.app.app).get("/health")
    assert response.status_code == 200


@pytest.mark.parametrize("failed", ["blocklist", "allowlist"])
def test_either_list_failure_drops_only_url_blocklist(
    monkeypatch: pytest.MonkeyPatch, failed: str
) -> None:
    psl, store, allowlist = _snapshots()

    if failed == "blocklist":
        monkeypatch.setattr(
            api.app.BlocklistStore,
            "load",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("blocklist broken")),
        )
    else:
        monkeypatch.setattr(api.app.BlocklistStore, "load", lambda *args, **kwargs: store)
        monkeypatch.setattr(
            api.app.RankAllowlist,
            "load",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("allowlist broken")),
        )

    loaded_store, loaded_allowlist, reason = api.app.load_url_snapshots(psl)
    assert loaded_store is None
    assert loaded_allowlist is None
    assert f"{failed} broken" in reason

    registry = api.app.build_registry(psl, loaded_store, loaded_allowlist, None)
    names = _names(registry)
    assert "url_blocklist" not in names
    assert URL_CHECK_NAMES - {"url_blocklist"} <= names
    assert ("url_blocklist", reason) in api.app.build_unregistered(reason, llm_loaded=True)


def test_llm_is_absent_when_gguf_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(api.app.GGUF_ENV, raising=False)
    assert api.app.build_llm_check() is None


def test_llm_is_absent_when_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub")
    monkeypatch.setenv(api.app.GGUF_ENV, str(model))
    monkeypatch.setattr(api.app.importlib.util, "find_spec", lambda name: None)
    assert api.app.build_llm_check() is None


def test_configured_missing_gguf_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.gguf"
    monkeypatch.setenv(api.app.GGUF_ENV, str(missing))
    monkeypatch.setattr(api.app.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(FileNotFoundError, match="missing.gguf"):
        api.app.build_llm_check()


def test_llm_build_uses_an_injected_stub_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub")
    monkeypatch.setenv(api.app.GGUF_ENV, str(model))
    monkeypatch.setattr(api.app.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(api.app.importlib, "import_module", lambda name: _FakeLlamaModule())

    check = api.app.build_llm_check()
    assert isinstance(check, LlmCheck)
    assert check.name == SCAM_SIGNAL


def test_unregistered_list_marks_llm_as_required_but_unloaded() -> None:
    rows = dict(api.app.build_unregistered("", llm_loaded=False))
    assert SCAM_SIGNAL in rows
    assert "必備" in rows[SCAM_SIGNAL]
    assert "尚未載入" in rows[SCAM_SIGNAL]


def test_health_counts_loaded_llm_and_reports_unloaded_llm() -> None:
    loaded_registry = api.app.build_registry(None, None, None, _llm_check())
    loaded = api.app.build_health(
        loaded_registry,
        api.app.build_unregistered("", llm_loaded=True),
        None,
        api.app.BLOCKLIST_MAX_AGE_DAYS,
        api.app.TABLE,
    )
    assert SCAM_SIGNAL in _names(loaded_registry)
    assert SCAM_SIGNAL not in {item.name for item in loaded.checks_unregistered}
    assert loaded.checks_registered == len(loaded_registry.enabled())

    unloaded_registry = api.app.build_registry(None, None, None, None)
    unloaded = api.app.build_health(
        unloaded_registry,
        api.app.build_unregistered("", llm_loaded=False),
        None,
        api.app.BLOCKLIST_MAX_AGE_DAYS,
        api.app.TABLE,
    )
    assert SCAM_SIGNAL not in _names(unloaded_registry)
    assert SCAM_SIGNAL in {item.name for item in unloaded.checks_unregistered}
    assert unloaded.checks_registered == len(unloaded_registry.enabled())


def test_hard_evidence_skips_llm_but_no_hard_evidence_runs_it() -> None:
    hard_llm = _TrackingLlmCheck()
    hard_registry = CheckRegistry()
    hard_registry.register(
        _PlainCheck(
            "url_blocklist",
            [CheckResult(name="url_blocklist", hit=True, detail="名單命中", hard=True)],
        )
    )
    hard_registry.register(hard_llm)
    hard_verdict = detect(
        Request(messages=[Message(text="請點網址確認")]), hard_registry, api.app.TABLE
    )
    assert hard_llm.calls == 0
    assert [r.detail for r in hard_verdict.checks if r.name == SCAM_SIGNAL] == [SKIPPED]

    normal_llm = _TrackingLlmCheck()
    normal_registry = CheckRegistry()
    normal_registry.register(normal_llm)
    detect(Request(messages=[Message(text="猜猜我是誰")]), normal_registry, api.app.TABLE)
    assert normal_llm.calls == 1


def test_check_route_preserves_detect_default_short_circuit() -> None:
    tree = ast.parse(Path(api.app.__file__).read_text(encoding="utf-8"))
    route = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check"
    )
    calls = [
        node
        for node in ast.walk(route)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "detect"
    ]
    assert len(calls) == 1
    assert "short_circuit" not in {keyword.arg for keyword in calls[0].keywords}


def test_app_source_has_no_forbidden_handlers_import_guards_or_nested_defs() -> None:
    tree = ast.parse(Path(api.app.__file__).read_text(encoding="utf-8"))
    for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
        assert handler.type is not None, "裸 except 被禁止"
        caught = ast.dump(handler.type)
        assert "Exception" not in caught
        assert "BaseException" not in caught
    for guarded in (node for node in ast.walk(tree) if isinstance(node, ast.Try)):
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(guarded)
        ), "import 不得包在 try 內"
    for function in (
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        assert not any(
            isinstance(descendant, (ast.FunctionDef, ast.AsyncFunctionDef))
            for statement in function.body
            for descendant in ast.walk(statement)
        ), f"{function.name} 內出現巢狀 def"


def test_check_response_contract_still_has_exactly_six_fields() -> None:
    assert set(CheckResponse.model_fields) == {
        "scam_probability",
        "abstained",
        "confidence",
        "scam_type",
        "evidence",
        "actions",
    }
