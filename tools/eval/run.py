"""對測試集跑一次 `detect()`，把每一則的判定攤平成一筆可重算的紀錄。

**`short_circuit=False`。** 沒有 LLM 可省，唯一的 `Stage.EXPENSIVE` 是
`domain_age`（預設不註冊）；關掉短路換取 `Verdict.checks` 的完整訊號圖，
使群組分群正確性分析（`GroupContribution.shadowed`）不因短路而缺材料。

**`RunRecord` 保留 `score` 與 `confidence` 的原始值，而不是只保留判定結果。**
門檻掃描因此是一次純算術（`decided ⟺ confidence >= floor 且 score >= 門檻`），
不需要為每一個切點重跑一次 `detect()`。這不是最佳化，是正確性：重跑會讓
掃描的五個切點各自看到一份可能不同的 `Document`，而掃描要問的是
「同一份判定在不同門檻下會怎樣」。

**組裝層在這裡，不在 `scam_guard/`。** `register_url_checks` 需要一份 PSL 與
黑名單快照，`register_domain_age_check` 需要一個查詢器 —— 三者都是本機檔案或
對外查詢，偵測核心不知道它們從哪來。
"""

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.normalize import DEFAULT_LIMITS, Limits
from scam_guard.pipeline import detect
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.scoring import compute_score, is_decision
from scam_guard.types import Request
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import load_tables, register_url_checks
from scam_guard.weights import WeightTable
from tools.eval.dataset import Sample, file_sha256
from tools.eval.selectors import LABEL_SCAM

SNAPSHOT_MAX_AGE_DAYS = 400
"""本機快照的最大年齡。見 `tools/eval/promo_scan.py` 的同名常數。

評估用的是**當下手上這一份**快照，它有多舊是一個要被 `run_id` 記錄的事實，
不是一個要在這裡擋下來的條件。
"""

NO_CLAUSE_MARKER = "本句未切出子句"
"""`scam_guard.rules.speech_act._detail()` 在整句只有一個子句時加上的標記。

該函式不匯出這個字串，所以此處是一份刻意的重複，由
`tests/test_eval_report.py` 的一條測試以真實命中釘住 ——
標記改了而這裡沒改時，那條測試會失敗，而不是這個比率悄悄變成 0。
"""

SIMPLIFIED_MARKERS = frozenset("发会银账验码个这为将应该网门题产")
"""判定簡體的高頻字集合。**粗略，沒有依據，是一個起點。**

本專案刻意不做繁簡轉換，也沒有繁簡對照表，所以這只能是一個啟發式。
`add-evasion-check` 已記錄「`賬號` 是中國大陸標準寫法不是規避」，
兩處的判斷要一致 —— 這裡同樣把簡體當成**來源特徵**，不當成可疑訊號。
"""

SIMPLIFIED_MIN_HITS = 2
"""出現幾個上列字才算簡體。1 個會把引用單一簡體詞的繁體訊息掃進來。"""


@dataclass(frozen=True)
class RunRecord:
    """一則樣本跑完 `detect()` 之後的全部可重算事實。

    刻意攤平成基本型別：報告層、掃描層與消融層都只讀這個型別，
    三者因此不需要各自持有一份 `Verdict`，也不可能各自重算一次分數。
    """

    id: str
    subset: str
    split: str
    label: str
    score: float
    confidence: float
    probability: float | None
    abstained: bool
    decided: bool
    scam_type: str | None
    contradicted: bool
    quotation_hit: bool
    blocklist_hit: bool
    simplified: bool
    hit_signals: tuple[str, ...]
    hard_signals: tuple[str, ...]
    shadowed_signals: tuple[str, ...]
    speech_act_hits: int
    no_clause_hits: int
    evidence: tuple[str, ...]
    actions: tuple[str, ...]


@dataclass(frozen=True)
class RunId:
    """一次評估的可重現性座標。四者缺一即無法宣稱可重現。"""

    testset_manifest_sha256: str
    weights_sha256: str
    blocklist_data_through: str
    blocklist_manifest_sha256: str
    git_commit: str

    def __str__(self) -> str:
        return (
            f"testset={self.testset_manifest_sha256[:12]}"
            f"-weights={self.weights_sha256[:12]}"
            f"-blocklist={self.blocklist_data_through}"
            f"@{self.blocklist_manifest_sha256[:12]}"
            f"-code={self.git_commit[:12]}"
        )


def build_registry(
    psl: PublicSuffixList,
    *,
    store: BlocklistStore | None = None,
) -> CheckRegistry:
    """組出評估用的完整 registry。

    **`domain_age` 不在此註冊。** 它需要一個 `DomainAgeLookup`，而那是本系統
    唯一的對外查詢入口；注入它是一個顯式的決定，由 `add-ablation` 的網域年齡
    實驗自行負責。未註冊的後果是 `Verdict.checks` 裡沒有它的記錄 ——
    與「跑了沒命中」在型別上可區分，這正是 `register_domain_age_check` 的設計。
    """
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    register_url_checks(registry, psl, load_tables(), store=store)
    return registry


def load_psl(psl_dir: Path) -> PublicSuffixList:
    return PublicSuffixList.load(psl_dir, max_age_days=SNAPSHOT_MAX_AGE_DAYS)


def load_blocklist(blocklist_dir: Path, psl: PublicSuffixList) -> BlocklistStore:
    return BlocklistStore.load(blocklist_dir, psl, max_age_days=SNAPSHOT_MAX_AGE_DAYS)


def _speech_act_names() -> frozenset[str]:
    """21 條言語行為規則的名稱。由註冊函式本身取得，不另抄一份清單。"""
    probe = CheckRegistry()
    register_speech_act_rules(probe)
    return frozenset(check.name for check in probe.enabled())


SPEECH_ACT_SIGNALS: frozenset[str] = _speech_act_names()
"""未切出子句的比例只在言語行為規則上有意義 —— 只有它們做子句切分。

在此算一次而不是每則算一次；建構規則不做任何 I/O，不違反
`scam_guard.weights` 那條「import 時不讀檔」的界線。
"""


def _is_simplified(text: str) -> bool:
    return sum(1 for character in text if character in SIMPLIFIED_MARKERS) >= SIMPLIFIED_MIN_HITS


def run_one(
    sample: Sample,
    registry: CheckRegistry,
    table: WeightTable,
    *,
    limits: Limits = DEFAULT_LIMITS,
) -> RunRecord:
    """跑一則。`short_circuit=False`，理由見模組 docstring。"""
    request = Request.from_text(sample.text)
    verdict = detect(request, registry, table, short_circuit=False, limits=limits)
    score = compute_score(verdict.checks, table)
    hits = [result for result in verdict.checks if result.hit]
    blocklist_name = table.roles["blocklist_exact"]
    quotation_name = table.roles["quotation"]
    speech_act_hits = [result for result in hits if result.name in SPEECH_ACT_SIGNALS]
    return RunRecord(
        id=sample.id,
        subset=sample.subset,
        split=sample.split,
        label=sample.label,
        score=score.value,
        confidence=verdict.confidence,
        probability=verdict.scam_probability,
        abstained=verdict.scam_probability is None,
        decided=is_decision(score, verdict, table),
        scam_type=None if verdict.scam_type is None else verdict.scam_type.name,
        contradicted=score.contradicted,
        quotation_hit=any(result.name == quotation_name for result in hits),
        blocklist_hit=any(result.name == blocklist_name for result in hits),
        simplified=_is_simplified(sample.text),
        hit_signals=tuple(result.name for result in hits),
        hard_signals=tuple(result.name for result in hits if result.hard),
        shadowed_signals=tuple(
            shadowed.name
            for contribution in score.group_contributions
            for shadowed in contribution.shadowed
        ),
        speech_act_hits=len(speech_act_hits),
        no_clause_hits=sum(1 for result in speech_act_hits if NO_CLAUSE_MARKER in result.detail),
        evidence=tuple(verdict.evidence),
        actions=tuple(verdict.actions),
    )


def run_over(
    samples: Sequence[Sample],
    registry: CheckRegistry,
    table: WeightTable,
    *,
    limits: Limits = DEFAULT_LIMITS,
) -> tuple[RunRecord, ...]:
    return tuple(run_one(sample, registry, table, limits=limits) for sample in samples)


def _git_commit() -> str:
    """目前的 commit。取不到即 raise —— 一份不知道自己跑在哪個版本上的報告
    不可重現，而「unknown」這個字串看起來像一個值。"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"取不到 git commit：{result.stderr.strip()!r}")
    return result.stdout.strip()


def build_run_id(
    manifest_path: Path, table: WeightTable, blocklist_manifest: Path, store: BlocklistStore
) -> RunId:
    """組出 `run_id`。黑名單的 `data_through` 取三個來源中最舊的一個。

    取最舊而不是最新：黑名單的召回受限於**最落後**的那一份資料，
    取最新會讓一份含已停更資料集的快照看起來比實際新鮮。
    """
    sources = store.manifest["sources"]
    data_through = min(str(source["data_through"]) for source in sources.values())  # type: ignore[union-attr]
    return RunId(
        testset_manifest_sha256=file_sha256(manifest_path),
        weights_sha256=file_sha256(table.path),
        blocklist_data_through=data_through,
        blocklist_manifest_sha256=file_sha256(blocklist_manifest),
        git_commit=_git_commit(),
    )


def write_records(path: Path, records: Sequence[RunRecord]) -> None:
    """把逐則紀錄落地，供重跑報告而不必重跑 `detect()`。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def scam_records(records: Sequence[RunRecord]) -> tuple[RunRecord, ...]:
    return tuple(record for record in records if record.label == LABEL_SCAM)
