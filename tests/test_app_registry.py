"""`app.py` 組裝層的註冊行為：URL 層與必備 LLM 層的掛載、缺席時的誠實降級，
以及未註冊清單的組出。

**本檔不打真實網路、不依賴 `data/` 快照、不需要 GGUF 模型檔。** 三份快照以
`PublicSuffixList.parse` 與直接建構的 `BlocklistStore` / `RankAllowlist` 就地捏出；
LLM 的兩個掛載條件（`SCAM_GUARD_GGUF` 與 `llm` extra）以 monkeypatch 模擬。
"""

import ast
import subprocess
from pathlib import Path

import pytest

import app
import demo_ui
from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.llm.check import LlmCheck
from scam_guard.llm.validate import SCAM_SIGNAL
from scam_guard.pipeline import SKIPPED
from scam_guard.types import CheckResult
from scam_guard.url import PublicSuffixList

PSL_TEXT = "// ===BEGIN ICANN DOMAINS===\ncom\n// ===END ICANN DOMAINS===\n"

URL_CHECK_NAMES = frozenset(
    {"url_blocklist", "url_shortener", "url_tld_risk", "url_host_shape", "url_brand"}
)


class _StubRuntime:
    """不載真模型的 `LlmRuntime` 替身。`LlmCheck.__init__` 只儲存 runtime，
    建構階段不呼叫 `generate`，所以一個什麼都不做的 `generate` 就夠。"""

    def generate(self, prompt: str, grammar: str, *, deadline_s: float) -> str:
        return ""


class _FakeLlamaModule:
    """`importlib.import_module("llm_runtime.llama_cpp_runtime")` 的替身模組，
    其 `LlamaCppRuntime(path)` 回傳一個 `_StubRuntime`，不觸碰 `llama_cpp`。"""

    @staticmethod
    def LlamaCppRuntime(path: Path) -> _StubRuntime:  # noqa: N802 - 對齊被替身的類別名
        return _StubRuntime()


def _dummy_find_spec(name: str) -> object:
    """`importlib.util.find_spec` 的替身：回傳非 `None` 代表 extra 已裝。"""
    return object()


def _fake_import_module(name: str) -> _FakeLlamaModule:
    return _FakeLlamaModule()


def _psl() -> PublicSuffixList:
    return PublicSuffixList.parse(PSL_TEXT)


def _names(registry) -> set[str]:
    return {check.name for check in registry.enabled()}


# ---------------------------------------------------------------------------
# URL 層註冊
# ---------------------------------------------------------------------------


def test_all_snapshots_valid_registers_five_url_checks() -> None:
    psl = _psl()
    store = BlocklistStore((), {"sources": {}}, psl)
    allowlist = RankAllowlist({}, {})
    registry = app.build_registry(psl, store, allowlist, None)
    assert URL_CHECK_NAMES <= _names(registry)
    unregistered = app.build_unregistered("", llm_loaded=True)
    assert "url_blocklist" not in {name for name, _label, _reason in unregistered}


def test_psl_missing_registers_no_url_checks_and_builds_no_empty_psl(tmp_path) -> None:
    """PSL 缺席 → registry 不含任何 `url_*`，三層規則與 n-gram 仍在。
    `load_psl()` 回 `None`（不建空 `PublicSuffixList`）。"""
    registry = app.build_registry(None, None, None, None)
    names = _names(registry)
    assert not any(name.startswith("url_") for name in names)
    assert {"quotation", "ngram_classifier"} <= names  # 規則與 n-gram 仍在
    app_psl_dir = tmp_path / "psl"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(app, "PSL_DIR", app_psl_dir)
        assert app.load_psl() is None


def test_blocklist_failure_drops_only_url_blocklist() -> None:
    """黑白名單缺席（`store`/`allowlist` 皆 `None`）→ `url_blocklist` 不註冊、
    其餘四個 URL 檢查照常；未註冊清單多一列且理由含例外訊息原文。
    沒有建出 `entry_count == 0` 的 `BlocklistStore`。"""
    psl = _psl()
    registry = app.build_registry(psl, None, None, None)
    names = _names(registry)
    assert "url_blocklist" not in names
    assert (URL_CHECK_NAMES - {"url_blocklist"}) <= names
    reason = "165 涉詐網址名單快照無法載入：manifest 缺少欄位 'sha256'"
    unregistered = app.build_unregistered(reason, llm_loaded=True)
    rows = {name: r for name, _label, r in unregistered}
    assert "url_blocklist" in rows
    assert reason in rows["url_blocklist"]


# ---------------------------------------------------------------------------
# LLM 層掛載
# ---------------------------------------------------------------------------


def test_llm_not_mounted_when_gguf_unset(monkeypatch) -> None:
    monkeypatch.delenv(app.GGUF_ENV, raising=False)
    assert app.build_llm_check() is None


def test_llm_not_mounted_when_extra_missing(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub")
    monkeypatch.setenv(app.GGUF_ENV, str(model))
    monkeypatch.setattr(app.importlib.util, "find_spec", lambda name: None)
    assert app.build_llm_check() is None


def test_llm_set_but_file_missing_raises(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "not-here.gguf"
    monkeypatch.setenv(app.GGUF_ENV, str(missing))
    monkeypatch.setattr(app.importlib.util, "find_spec", _dummy_find_spec)
    with pytest.raises(FileNotFoundError, match="not-here.gguf"):
        app.build_llm_check()


def test_llm_mounts_when_gguf_and_extra_present(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub")
    monkeypatch.setenv(app.GGUF_ENV, str(model))
    monkeypatch.setattr(app.importlib.util, "find_spec", _dummy_find_spec)
    monkeypatch.setattr(app.importlib, "import_module", _fake_import_module)
    check = app.build_llm_check()
    assert isinstance(check, LlmCheck)
    assert check.name == SCAM_SIGNAL


def test_llm_mounted_check_registers_in_the_same_registry(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub")
    monkeypatch.setenv(app.GGUF_ENV, str(model))
    monkeypatch.setattr(app.importlib.util, "find_spec", _dummy_find_spec)
    monkeypatch.setattr(app.importlib, "import_module", _fake_import_module)
    check = app.build_llm_check()
    registry = app.build_registry(None, None, None, check)
    assert SCAM_SIGNAL in _names(registry)


# ---------------------------------------------------------------------------
# 未註冊清單
# ---------------------------------------------------------------------------


def test_unregistered_flags_llm_as_mandatory_but_unloaded() -> None:
    unregistered = app.build_unregistered("", llm_loaded=False)
    rows = {name: reason for name, _label, reason in unregistered}
    assert SCAM_SIGNAL in rows
    assert "必備" in rows[SCAM_SIGNAL]


def test_unregistered_drops_llm_row_when_loaded() -> None:
    names = {name for name, _label, _reason in app.build_unregistered("", llm_loaded=True)}
    assert names == {"domain_age"}


# ---------------------------------------------------------------------------
# 語意格：短路「無需動用」 vs 未載入「未載入」（demo_ui 聚合，本 change 的實質區分）
# ---------------------------------------------------------------------------


def test_short_circuit_is_standby_not_unloaded() -> None:
    checks = [CheckResult(name=SCAM_SIGNAL, hit=False, detail=SKIPPED)]
    standby = demo_ui.source_statuses(checks, ())
    unloaded = demo_ui.source_statuses([], ((SCAM_SIGNAL, "語意判讀", "模型沒有載入"),))
    semantic_standby = next(s for s in standby if s.label == demo_ui.SOURCE_SEMANTIC)
    semantic_unloaded = next(s for s in unloaded if s.label == demo_ui.SOURCE_SEMANTIC)
    assert semantic_standby.state == demo_ui.STATE_STANDBY
    assert semantic_unloaded.state == demo_ui.STATE_UNLOADED
    assert semantic_standby.state != semantic_unloaded.state


# ---------------------------------------------------------------------------
# 原始碼約束：無 except Exception / 裸 except / try-import / 巢狀 def
# ---------------------------------------------------------------------------


def _app_tree() -> ast.Module:
    return ast.parse(Path(app.__file__).read_text(encoding="utf-8"))


def test_app_has_no_forbidden_exception_handlers() -> None:
    tree = _app_tree()
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        assert handler.type is not None, "裸 except 被禁止"
        caught = ast.dump(handler.type)
        assert "Exception" not in caught, "except Exception 被禁止"
        assert "BaseException" not in caught, "except BaseException 被禁止"


def test_app_never_wraps_an_import_in_try() -> None:
    tree = _app_tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for inner in ast.walk(node):
            assert not isinstance(inner, (ast.Import, ast.ImportFrom)), "import 不得包在 try 內"


def test_app_has_no_nested_def() -> None:
    tree = _app_tree()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in node.body:
            for inner in ast.walk(child):
                assert not isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)), (
                    f"{node.name} 內出現巢狀 def"
                )


def test_scam_guard_is_untouched_by_this_change() -> None:
    repo_root = Path(app.__file__).resolve().parent
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "scam_guard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", f"scam_guard/ 不該被本 change 改動：\n{result.stdout}"
