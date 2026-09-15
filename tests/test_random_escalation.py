"""隨機 escalation 對照組：可重現、能分出高判別力訊號、也能指出無判別力的訊號。"""

from dataclasses import dataclass

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document
from scam_guard.types import CheckResult, Request, ScamType
from scam_guard.weights import load_weights
from tools.eval.ablation import Outcome, contribution
from tools.eval.dataset import Sample
from tools.eval.random_escalation import DEFAULT_SEED, RandomEscalationCheck, replace_check
from tools.eval.run import run_over
from tools.eval.selectors import COFACTS_HAM_AD, COFACTS_SCAM
from tools.eval.stats import Rate

TABLE = load_weights()
SIGNAL = "solicit_otp"
SCAM_MARK = "把驗證碼傳給我"


@dataclass(frozen=True)
class KeywordCheck:
    """只看一個關鍵字的檢查。借 `solicit_otp` 的名稱，因此權重與群組都是真的。"""

    name: str = SIGNAL
    stage: Stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        if SCAM_MARK in req.latest.text:
            return [
                CheckResult(
                    name=self.name,
                    hit=True,
                    detail="命中關鍵字",
                    hard=True,
                    scam_types=[ScamType.PHISHING_LINK],
                )
            ]
        return []


@dataclass(frozen=True)
class CoinFlipCheck:
    """以文字長度的奇偶決定命中 —— 與標籤無關，判別力為零。"""

    name: str = SIGNAL
    stage: Stage = Stage.LOCAL

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        if len(req.latest.text) % 2 == 0:
            return [
                CheckResult(
                    name=self.name,
                    hit=True,
                    detail="拋硬幣",
                    hard=True,
                    scam_types=[ScamType.PHISHING_LINK],
                )
            ]
        return []


def samples() -> list[Sample]:
    """100 則正例（皆含關鍵字）與 100 則負例（皆不含），長度奇偶各半。

    **每一則的文字都不同。** 隨機對照組的命中與否由 `(seed, name, 訊息全文)`
    決定，重複的文字會得到重複的抽籤結果 —— 真實語料裡不會有兩則一模一樣的
    訊息（Cofacts 收錄時已以原文去重），但合成資料很容易踩到。
    """
    scam = [
        Sample(
            id=f"scam-{index}",
            text=f"{SCAM_MARK}{'。' * (index % 2)}{index:04d}",
            subset=COFACTS_SCAM.name,
        )
        for index in range(100)
    ]
    ham = [
        Sample(
            id=f"ham-{index}",
            text=f"這是一則正常的宣導訊息{'。' * (index % 2)}{index:04d}",
            subset=COFACTS_HAM_AD.name,
        )
        for index in range(100)
    ]
    return scam + ham


def registry_with(check: object) -> CheckRegistry:
    registry = CheckRegistry()
    registry.register(check)  # type: ignore[arg-type]
    return registry


def outcome_of(records: object) -> Outcome:
    from tools.eval.ablation import _outcome

    return _outcome(records)  # type: ignore[arg-type]


def test_a_fixed_seed_gives_identical_results() -> None:
    registry = registry_with(KeywordCheck())
    scoped = replace_check(
        registry,
        RandomEscalationCheck(
            name=SIGNAL, probability=0.5, hard=True, typed=True, seed=DEFAULT_SEED
        ),
    )
    first = [record.decided for record in run_over(samples(), scoped, TABLE)]
    second = [record.decided for record in run_over(samples(), scoped, TABLE)]
    assert first == second
    assert any(first)


def test_a_different_seed_gives_a_different_draw() -> None:
    registry = registry_with(KeywordCheck())
    left = replace_check(
        registry, RandomEscalationCheck(name=SIGNAL, probability=0.5, hard=True, typed=True, seed=1)
    )
    right = replace_check(
        registry, RandomEscalationCheck(name=SIGNAL, probability=0.5, hard=True, typed=True, seed=2)
    )
    assert [r.decided for r in run_over(samples(), left, TABLE)] != [
        r.decided for r in run_over(samples(), right, TABLE)
    ]


def test_the_control_never_reads_the_message_text() -> None:
    """不讀文本：同樣的機率下，文字完全不同的兩則有相同的命中分布統計。"""
    check = RandomEscalationCheck(name=SIGNAL, probability=1.0, hard=False, typed=False)
    long_request = Request.from_text("一段很長的詐騙話術，要求你把驗證碼傳過來")
    short_request = Request.from_text("嗨")
    assert check(long_request, None) and check(short_request, None)  # type: ignore[arg-type]


def test_a_discriminative_signal_beats_the_same_rate_random_control() -> None:
    """只命中 scam 的合成訊號，召回提升顯著高於同命中率的隨機注入。"""
    registry = registry_with(KeywordCheck())
    baseline = run_over(samples(), registry, TABLE)
    result = contribution(
        SIGNAL,
        "signal",
        [SIGNAL],
        samples(),
        registry,
        TABLE,
        baseline,
        outcome_of(baseline),
        seed=DEFAULT_SEED,
    )
    assert result.hit_rate == Rate(numerator=100, denominator=200)
    assert result.distinguishable_from_random is True
    assert result.delta_recall < 0


def test_a_signal_equivalent_to_a_coin_flip_is_not_distinguishable() -> None:
    """在 scam 與 ham 上命中率相同的訊號，與隨機注入無法區分。"""
    registry = registry_with(CoinFlipCheck())
    baseline = run_over(samples(), registry, TABLE)
    result = contribution(
        SIGNAL,
        "signal",
        [SIGNAL],
        samples(),
        registry,
        TABLE,
        baseline,
        outcome_of(baseline),
        seed=DEFAULT_SEED,
    )
    assert result.hit_rate == Rate(numerator=100, denominator=200)
    assert result.distinguishable_from_random is False


def test_replace_check_swaps_by_name_and_leaves_the_original_alone() -> None:
    registry = registry_with(KeywordCheck())
    replacement = RandomEscalationCheck(name=SIGNAL, probability=0.5, hard=True, typed=True)
    scoped = replace_check(registry, replacement)
    assert [type(check) for check in scoped.enabled()] == [RandomEscalationCheck]
    assert [type(check) for check in registry.enabled()] == [KeywordCheck]


def test_the_control_is_not_registered_in_the_weight_table() -> None:
    """`weights.toml` 中沒有任何名為 random_escalation 的訊號。"""
    assert "random_escalation" not in TABLE.signals
    assert not [name for name in TABLE.signals if "random" in name]
