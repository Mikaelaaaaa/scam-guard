"""`domain_age` 的三種 outcome 處置、依據文案、粒度與短路行為。

全部以**注入的假 lookup** 測試，不觸發任何網路行為 —— 這正是把 RDAP 實作
放在 `net/` 的理由之一：核心的判斷邏輯不需要網路就能測完。
"""

import ast
from datetime import timedelta
from pathlib import Path

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.domain_age import (
    DEFAULT_THRESHOLD_DAYS,
    AgeOutcome,
    DomainAge,
    DomainAgeCheck,
    register_domain_age_check,
)
from scam_guard.normalize import Document, build_document
from scam_guard.pipeline import SKIPPED, detect
from scam_guard.types import CheckResult, Message, Request, ScamType
from scam_guard.url import utc_today
from scam_guard.url_check import load_tables
from tests.test_blocklist_store import psl_of

SHORTENERS = frozenset(load_tables().shorteners)

CORE_DIR = Path(__file__).resolve().parents[1] / "scam_guard"
BANNED_IN_CORE = frozenset(
    {
        "urllib.request",
        "urllib.error",
        "http.client",
        "socket",
        "ftplib",
        "smtplib",
        "requests",
        "httpx",
        "aiohttp",
        "net",
    }
)


class RecordingLookup:
    """記錄每一次呼叫的假 lookup。未預先安排的網域一律拋 `KeyError` ——
    「本來就不該被查的網域被查了」要大聲壞掉，不是安靜回一個空結果。"""

    def __init__(self, ages: dict[str, DomainAge]) -> None:
        self.calls: list[str] = []
        self._ages = ages

    def __call__(self, domain: str) -> DomainAge:
        self.calls.append(domain)
        if domain not in self._ages:
            raise KeyError(f"測試未安排此網域的結果：{domain!r}")
        return self._ages[domain]


class HardCheck:
    """一個必定命中硬證據的本機檢查，用來觸發 `detect()` 的短路。"""

    name = "fake_hard"
    stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return [CheckResult(name=self.name, hit=True, weight=1.0, detail="測試用硬證據", hard=True)]


def known(domain: str, days: int) -> DomainAge:
    return DomainAge(
        domain=domain,
        outcome=AgeOutcome.KNOWN,
        registered_on=utc_today() - timedelta(days=days),
    )


def imported_roots(source: str) -> set[str]:
    """一個模組直接 import 了哪些頂層模組路徑。只看 import 敘述 ——
    與 lint 規則同樣的範圍，`importlib.import_module()` 兩者都看不到。"""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            roots.add(node.module)
    return roots


def run(text: str, lookup: RecordingLookup, **kwargs: object) -> list[CheckResult]:
    check = DomainAgeCheck(lookup, psl_of(), SHORTENERS, weight=1.0, **kwargs)
    req = Request.from_text(text)
    return check(req, build_document(req.messages))


# --- 三種 outcome -----------------------------------------------------------


def test_known_and_new_hits_with_day_count_in_detail() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 6)})
    results = run("請點 https://evil.com/a 領取", lookup)
    assert len(results) == 1
    registered = (utc_today() - timedelta(days=6)).isoformat()
    assert results[0].detail == f"網域 evil.com 註冊於 6 天前（{registered}）"


def test_known_and_old_produces_no_result() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 900)})
    assert run("請點 https://evil.com/a 領取", lookup) == []


def test_no_data_produces_no_result() -> None:
    lookup = RecordingLookup({"evil.com": DomainAge("evil.com", AgeOutcome.NO_DATA)})
    assert run("請點 https://evil.com/a 領取", lookup) == []


def test_unavailable_produces_no_result_and_does_not_raise() -> None:
    lookup = RecordingLookup({"evil.com": DomainAge("evil.com", AgeOutcome.UNAVAILABLE)})
    assert run("請點 https://evil.com/a 領取", lookup) == []


def test_unknown_outcome_must_not_carry_a_date() -> None:
    for outcome in (AgeOutcome.NO_DATA, AgeOutcome.UNAVAILABLE):
        with pytest.raises(ValueError, match="registered_on MUST 為 None"):
            DomainAge(domain="evil.com", outcome=outcome, registered_on=utc_today())


def test_known_outcome_must_carry_a_date() -> None:
    with pytest.raises(ValueError, match="registered_on 不得為 None"):
        DomainAge(domain="evil.com", outcome=AgeOutcome.KNOWN)


def test_threshold_boundary_is_exclusive() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", DEFAULT_THRESHOLD_DAYS)})
    assert run("https://evil.com/a", lookup) == []
    lookup = RecordingLookup({"evil.com": known("evil.com", DEFAULT_THRESHOLD_DAYS - 1)})
    assert len(run("https://evil.com/a", lookup)) == 1


def test_threshold_is_a_parameter_not_a_constant() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 100)})
    assert run("https://evil.com/a", lookup) == []
    lookup = RecordingLookup({"evil.com": known("evil.com", 100)})
    assert len(run("https://evil.com/a", lookup, threshold_days=365)) == 1


# --- 結果的形狀 -------------------------------------------------------------


def test_hard_is_false() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 6)})
    assert run("https://evil.com/a", lookup)[0].hard is False


def test_scam_types_is_phishing_link_only() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 6)})
    assert run("https://evil.com/a", lookup)[0].scam_types == [ScamType.PHISHING_LINK]


def test_same_domain_two_urls_merge_into_one_result_with_both_coords() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 6)})
    results = run("先看 https://evil.com/a 領取。再看 https://evil.com/b 確認。", lookup)
    assert len(results) == 1
    assert results[0].evidence == [(0, 0), (0, 1)]
    assert lookup.calls == ["evil.com"]


def test_different_domains_produce_one_result_each() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 6), "bad.net": known("bad.net", 3)})
    results = run("看 https://evil.com/a 與 https://bad.net/b", lookup)
    assert [r.detail.split()[1] for r in results] == ["evil.com", "bad.net"]


def test_stage_is_expensive() -> None:
    assert DomainAgeCheck.stage is Stage.EXPENSIVE


# --- 不該被查詢的對象 -------------------------------------------------------


def test_ip_literal_is_never_looked_up() -> None:
    lookup = RecordingLookup({})
    assert run("請點 http://192.168.0.1/a 領取", lookup) == []
    assert lookup.calls == []


def test_shortener_domain_is_never_looked_up() -> None:
    lookup = RecordingLookup({})
    assert run("包裹配送失敗 reurl.cc/2xY3z", lookup) == []
    assert lookup.calls == []


def test_only_the_registrable_domain_is_sent() -> None:
    lookup = RecordingLookup({"evil.com": known("evil.com", 6)})
    run("https://login-esunbank.evil.com/a?token=abc", lookup)
    assert lookup.calls == ["evil.com"]


# --- 與 pipeline 的關係 -----------------------------------------------------


def test_hard_hit_short_circuits_before_any_lookup() -> None:
    lookup = RecordingLookup({})
    registry = CheckRegistry()
    registry.register(HardCheck())
    register_domain_age_check(registry, psl_of(), SHORTENERS, weight=1.0, lookup=lookup)
    verdict = detect(Request.from_text("請點 https://evil.com/a 領取"), registry)
    assert lookup.calls == []
    assert [r.detail for r in verdict.checks if r.name == "domain_age"] == [SKIPPED]


def test_core_imports_neither_net_nor_any_network_library() -> None:
    """架構界線的**測試**，與 lint 規則各守一邊。

    lint 規則可能被 `per-file-ignores` 改壞而沒有人發現（那正是本 change
    動過的那張表），所以界線不能只由設定檔守著。這個測試讀 `scam_guard/`
    的原始碼，與 ruff 看的是同一件事：直接的 import 敘述。
    """
    offenders: list[str] = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        for module in imported_roots(path.read_text(encoding="utf-8")):
            if module in BANNED_IN_CORE or module.split(".")[0] == "net":
                offenders.append(f"{path.name} → {module}")
    assert offenders == []


def test_without_lookup_the_check_is_not_registered_and_detect_still_works() -> None:
    registry = CheckRegistry()
    register_domain_age_check(registry, psl_of(), SHORTENERS, weight=1.0)
    assert [c.name for c in registry.enabled()] == []
    verdict = detect(Request(messages=[Message(text="請點 https://evil.com/a 領取")]), registry)
    assert verdict.checks == []
    assert verdict.scam_probability is None
