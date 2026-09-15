"""消融實驗 —— 逐訊號、逐群組關閉，以及上游指名的門檻掃描。

**本模組 MUST NOT 修改 `scam_guard/tables/weights.toml`。** 把某條訊號的 `basis`
從 `placeholder` 改為 `measured` 需要附兩個條件機率與 `measured_on`，那是一次會被
`load_weights()` 驗證的正式變更，值得有自己的 commit 與 review，
不該是消融腳本的副作用。本模組的產出是一份**建議表**。

**全部重跑一律 `short_circuit=False`**（由 `tools.eval.run.run_one` 保證），
使 `Verdict.checks` 保留完整訊號圖，`GroupContribution.shadowed` 才有材料。

**頭條結論一律引 `holdout`。** `tune` 是用來調參的那一側，而消融的結論要拿來
決定「這條規則值不值得留」，那個決定必須基於沒被用來調過任何東西的那一側。

⚠️ **本輪消融跑在 `placeholder` 權重下。** 33 個訊號的權重只有四個值，關閉任何
一條 Tier-A 訊號對分數的影響幾乎相同，所以「訊號 A 被關閉後召回率降了 5%」
有很大一部分反映的是**訊號 A 的命中率**而不是它的判別力。隨機對照組緩解了
「任何命中都有效」這個最粗的混淆，但緩解不了「兩個命中率相同的訊號誰更強」。
那需要 `measured` 權重，而本輪產生的正是那些權重的原始資料 ——
**這是一個順序上無法迴避的雞生蛋問題**，見 `PLACEHOLDER_LIMITATION`。
"""

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from scam_guard.check import CheckRegistry
from scam_guard.domain_age import DomainAge, register_domain_age_check
from scam_guard.blocklist import BlocklistStore
from scam_guard.normalize import DEFAULT_LIMITS, Limits, build_document
from scam_guard.types import Message
from scam_guard.weights import WeightTable, load_weights
from net.rdap import HttpsTransport, RdapBootstrap, RdapLookup
from net.rdap_cache import RdapCache
from tools.eval.dataset import MANIFEST_FILENAME, Sample, load_testset
from scam_guard.url import PublicSuffixList, extract_urls
from scam_guard.url_check import UrlBrandCheck, group_by_registrable_domain, load_tables
from tools.eval.baseline_tfidf import EVIDENCE_NOTE, BaselineResult, train_and_evaluate
from tools.eval.random_escalation import DEFAULT_SEED, RandomEscalationCheck, replace_check
from tools.eval.recommendations import (
    DISCLAIMER,
    RECOMMENDATION_COLUMNS,
    recommend,
)
from tools.eval.report import MULTI_MESSAGE_MINIMUM, subset_report
from tools.eval.run import (
    RunRecord,
    build_registry,
    build_run_id,
    load_blocklist,
    load_psl,
    run_over,
)
from tools.eval.selectors import HAM_SUBSETS, HOLDOUT, LABEL_SCAM
from tools.eval.stats import Rate

PLACEHOLDER_LIMITATION = (
    '本輪消融執行時全部訊號的權重仍為 `basis = "placeholder"` 的封閉集合四個值'
    "（2.5、0.6、0.0、-1.5）。因此**逐訊號消融量到的主要是各訊號的命中率，"
    "而不是它的判別力**；訊號之間的相對排序在權重改為 `measured` 之後可能改變。"
    "這是一個順序上無法迴避的雞生蛋問題：可估性資料要等這一輪消融完成才有，"
    "而消融的解讀精確度又受限於還沒被這批資料更新過的權重。"
)

CONFIDENCE_SWEEPS: Mapping[str, tuple[float, ...]] = {
    "base_hard": (0.45, 0.60, 0.75, 0.90, 1.00),
    "base_multi_group": (0.45, 0.55, 0.70, 0.85),
    "base_single_group": (0.05, 0.20, 0.35, 0.39),
    "base_no_hit": (0.00, 0.05, 0.20, 0.39),
    "cap_contradiction": (0.05, 0.20, 0.30, 0.39),
    "cap_unseen_pattern": (0.05, 0.20, 0.35, 0.39),
    "cap_truncated": (0.41, 0.50, 0.70, 0.90),
    "confidence_floor": (0.10, 0.31, 0.34, 0.40, 0.45, 0.60, 0.70, 0.95),
}
"""八個信心相關門檻的單變數掃描範圍。

範圍守住 `add-confidence` 定義的六條相對關係 —— 例如 `base_no_hit` 的上界不超過
`confidence_floor`（0.40），否則違反「完全無訊號必定拒答」這條不變式，
而掃出來的那個點量到的是一個**不同的系統**，不是這個系統的一個設定。
`confidence_floor` 本身是被掃的對象，因此它的範圍不受那六條關係限制。
"""

PRIOR_SWEEP: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
"""`prior_log_odds` 的掃描範圍。**掃描但不建議變更。**

先驗一旦非 0，`add-score-compute` 定的「完全無訊號時可能性為 0.5」這條語意就被
打破 —— 那是一個語意決定，不是掃描能替代的。
"""

URL_BRAND_SHORT_LABEL_LENGTHS: tuple[int, ...] = (4, 5, 6)
"""`url_brand` 的「官方主標籤短於 N 個字元時不做編輯距離判定」。

`add-url-check` 記錄這個 5 沒有實驗依據。
"""

DOMAIN_AGE_THRESHOLDS: tuple[int, ...] = (7, 14, 30, 60, 90)
"""網域年齡門檻。30 天沒有實驗依據；165027 的天數中位數 44 落在 30–60 之間，
因此不另外加一個 44 天的掃描點 —— 它已經被這個範圍涵蓋。"""

BLOCKED: Mapping[str, str] = {
    "add-tranco-allowlist": (
        "排名門檻 N（1,000/5,000/10,000）的掃描**此刻不可執行**："
        "url-feeds PR 未合併，`scam_guard/` 中沒有任何 tranco 相關實作。"
        '`blocked_on = "add-tranco-allowlist"`。'
    ),
    "pii_nlp": (
        "啟用/不啟用 `pii_nlp/` 的對照組**此刻不可執行**：該目錄從未建立"
        "（`add-pii-recognizers` 已明文記錄 `opf` 契約驗證未通過），"
        '沒有「啟用」那一側可以比較。`blocked_on = "pii_nlp/ 的實際建立"`。'
    ),
    "add-verdict-render": (
        "`max_evidence_lines` / `max_actions` 的掃描**不執行**："
        "`add-verdict-render` 自己已判斷這兩個參數不影響 `scam_probability` 或 "
        "`confidence`，只影響呈現行數，對「訊號值不值得留」沒有貢獻。"
    ),
    "log-redaction": (
        "`NullRedactor` 對照組**不重開**：`add-redact-apply` 已把遮蔽定在全部檢查"
        "跑完之後，對 `Verdict` 與 LLM 輸出皆無影響，「關掉遮蔽差多少」的答案"
        "恆等於零，且是結構上的零。"
    ),
    "llm-layer": (
        "LLM 相關的全部掃描項目**此刻不可執行**：`llm-layer` 排在本 PR 之後，"
        "`registry` 中唯一的 `Stage.EXPENSIVE` 是 `domain_age`。"
    ),
}


@dataclass(frozen=True)
class Outcome:
    """一次重跑的三個數字。頭條誤判率取三個 ham 子集中 Wilson 上界最大者。"""

    headline_subset: str
    false_positive_rate: Rate
    recall: Rate
    abstention_rate: Rate


@dataclass(frozen=True)
class SignalContribution:
    """關閉一個訊號（或一整個群組）之後的變化。"""

    name: str
    kind: str
    registered: bool
    hit_rate: Rate | None
    delta_false_positive_rate: float
    delta_recall: float
    shadowed_count: int
    random_recall: Rate | None
    distinguishable_from_random: bool | None
    note: str


def _upper_of(entry: tuple[str, Rate]) -> float:
    """排序鍵：Wilson 上界。`Rate` 刻意不可排序 —— 比較兩個比率要說清楚比的是哪一端。"""
    return entry[1].upper


def _outcome(records: Sequence[RunRecord]) -> Outcome:
    """以 `add-metrics` 的 `subset_report()` 算三個數字，不重新實作任何統計量。"""
    scam = [record for record in records if record.label == LABEL_SCAM]
    candidates = []
    for name in HAM_SUBSETS:
        subset = [record for record in records if record.subset == name]
        if not subset:
            continue
        report = subset_report(name, subset)
        if report.false_positive_rate is not None:
            candidates.append((name, report.false_positive_rate))
    if not candidates or not scam:
        raise ValueError("消融需要至少一個 ham 子集與非空的 scam 子集")
    headline_subset, headline = max(candidates, key=_upper_of)
    return Outcome(
        headline_subset=headline_subset,
        false_positive_rate=headline,
        recall=Rate(numerator=sum(1 for record in scam if record.decided), denominator=len(scam)),
        abstention_rate=Rate(
            numerator=sum(1 for record in records if record.abstained), denominator=len(records)
        ),
    )


def _disabled(
    registry: CheckRegistry, names: Sequence[str]
) -> tuple[CheckRegistry | None, tuple[str, ...]]:
    """複製一份 registry 並停用指定的名稱，回傳 `(registry, 未註冊的名稱)`。

    `CheckRegistry.disable()` 對未註冊的名稱拋 `KeyError` —— 那是刻意的
    （拼錯名稱時安靜忽略會讓整組實驗悄悄變成對照組）。這裡把「未註冊」
    轉成一個**被記錄的結果**而不是一次中止：`domain_age` 在預設組裝下確實
    沒有註冊，而那件事本身就是報告要寫的一行。

    **一個群組裡有成員未註冊時照樣跑剩下的成員。** 未註冊等同「已經被關掉」，
    拒絕整組會讓 `url_reputation`（含 `domain_age`）這種群組永遠量不到 ——
    而那是六個非單元群組裡最大的一個。缺席的成員名稱一併回傳，寫進 `note`。
    """
    rebuilt = CheckRegistry()
    for check in registry.enabled():
        rebuilt.register(check)
    available = {check.name for check in rebuilt.enabled()}
    missing = tuple(name for name in names if name not in available)
    present = [name for name in names if name in available]
    if not present:
        return None, missing
    for name in present:
        rebuilt.disable(name)
    return rebuilt, missing


def ablate(
    names: Sequence[str],
    samples: Sequence[Sample],
    registry: CheckRegistry,
    table: WeightTable,
) -> tuple[tuple[RunRecord, ...] | None, tuple[str, ...]]:
    """停用一組訊號後重跑。全部名稱皆未註冊時第一個回傳值為 `None`。"""
    scoped, missing = _disabled(registry, names)
    if scoped is None:
        return None, missing
    return run_over(samples, scoped, table), missing


def _shadowed_count(records: Sequence[RunRecord], name: str) -> int:
    """該訊號在其群組內被別的訊號蓋過的次數。

    `add-score-compute` 指名這是「判斷分群對不對的唯一材料」：一個群組若長期
    都是同一條規則在提供 max，其他成員可能根本不該在裡面。
    """
    return sum(1 for record in records if name in record.shadowed_signals)


def contribution(
    name: str,
    kind: str,
    members: Sequence[str],
    samples: Sequence[Sample],
    registry: CheckRegistry,
    table: WeightTable,
    baseline: Sequence[RunRecord],
    baseline_outcome: Outcome,
    *,
    seed: int,
) -> SignalContribution:
    """關閉 `members` 後重跑，並（對單一訊號）跑一次同命中率的隨機對照組。"""
    ablated, missing = ablate(members, samples, registry, table)
    if ablated is None:
        return SignalContribution(
            name=name,
            kind=kind,
            registered=False,
            hit_rate=None,
            delta_false_positive_rate=0.0,
            delta_recall=0.0,
            shadowed_count=0,
            random_recall=None,
            distinguishable_from_random=None,
            note=(
                f"未註冊於評估用的 registry，無法消融："
                f"{'、'.join(missing)}（`domain_age` 需要注入 RDAP 查詢器）"
            ),
        )
    after = _outcome(ablated)
    hits = Rate(
        numerator=sum(1 for record in baseline if name in record.hit_signals),
        denominator=len(baseline),
    )
    random_recall = None
    distinguishable = None
    if kind == "signal" and hits.numerator > 0:
        # `hard` 取**實測的多數**而不是 `hard_capable`：一條 hard_capable 的規則
        # 若在這份語料上多半因自帶碼豁免而降級，讓對照組一律 hard=True 會給它
        # 2.5 的權重，那是在跟一個比真訊號更強的東西比。
        hard_hits = sum(1 for record in baseline if name in record.hard_signals)
        typed = any(name in record.typed_signals for record in baseline)
        scoped = replace_check(
            registry,
            RandomEscalationCheck(
                name=name,
                probability=hits.value,
                hard=hard_hits * 2 >= hits.numerator and table.signals[name].hard_capable,
                typed=typed,
                seed=seed,
            ),
        )
        random_recall = _outcome(run_over(samples, scoped, table)).recall
        distinguishable = not baseline_outcome.recall.overlaps(random_recall)
    return SignalContribution(
        name=name,
        kind=kind,
        registered=True,
        hit_rate=hits,
        delta_false_positive_rate=(
            after.false_positive_rate.value - baseline_outcome.false_positive_rate.value
        ),
        delta_recall=after.recall.value - baseline_outcome.recall.value,
        shadowed_count=_shadowed_count(baseline, name),
        random_recall=random_recall,
        distinguishable_from_random=distinguishable,
        note=(
            ""
            if not missing
            else f"群組中未註冊、因此本來就處於關閉狀態的成員：{'、'.join(missing)}"
        ),
    )


def split_groups(table: WeightTable) -> WeightTable:
    """把每個訊號各自視為獨立群組的衍生表。

    `WeightTable.with_overrides()` 只能改值，改不了分群，所以這裡直接建一個新的
    `WeightTable`。原表不動 —— 它是 frozen 的，而衍生表只活在記憶體裡。
    """
    return WeightTable(
        path=table.path,
        signals={name: replace(signal, group=name) for name, signal in table.signals.items()},
        roles=dict(table.roles),
        thresholds=dict(table.thresholds),
        type_priority=table.type_priority,
    )


def threshold_sweep(
    key: str,
    values: Sequence[float],
    samples: Sequence[Sample],
    registry: CheckRegistry,
    table: WeightTable,
) -> list[tuple[str, float, Outcome]]:
    """單變數掃描一個門檻。信心門檻會改變判定，因此**必須重跑**，不能純算術。"""
    rows: list[tuple[str, float, Outcome]] = []
    for value in values:
        derived = table.with_overrides(thresholds={key: value})
        rows.append((key, value, _outcome(run_over(samples, registry, derived))))
    return rows


def prior_sweep(
    baseline: Sequence[RunRecord], table: WeightTable, values: Sequence[float]
) -> list[tuple[str, float, Outcome]]:
    """`prior_log_odds` 的掃描。

    先驗只是把每一則的分數平移同一個量（現行值為 0），所以這裡是純算術 ——
    重跑 `detect()` 會得到完全相同的結果，只是慢六倍。
    """
    floor = table.threshold("confidence_floor")
    decision = table.threshold("decision_score")
    rows: list[tuple[str, float, Outcome]] = []
    for value in values:
        shifted = tuple(
            replace(
                record,
                score=record.score + value,
                decided=record.confidence >= floor and record.score + value >= decision,
            )
            for record in baseline
        )
        rows.append(("prior_log_odds", value, _outcome(shifted)))
    return rows


def contradiction_rate(records: Sequence[RunRecord]) -> Rate:
    """scam 子集上 `Score.contradicted` 為真的比例 —— 引述兩段式處置的真實代價。

    這些樣本本來會被判為詐騙，卻因為同時命中引述而降級為「無法判定」。
    `scoring.py` 的 docstring 稱它為「已知損失，不粉飾」，這裡是那句話的數字。
    """
    scam = [record for record in records if record.label == LABEL_SCAM]
    return Rate(numerator=sum(1 for record in scam if record.contradicted), denominator=len(scam))


def truncation_count(samples: Sequence[Sample], limits: Limits = DEFAULT_LIMITS) -> int:
    """實際觸發 `add-context-limits` 截斷的樣本數。

    截斷需要單一請求超過 100 則訊息或 50,000 字元，而測試集的每一則都是單則
    轉傳。這個數字幾乎確定是 0，而那本身就是要寫進報告的結論：
    **材料不足以產出門檻敏感度曲線。**
    """
    return sum(
        1 for sample in samples if build_document([Message(text=sample.text)], limits).truncated
    )


def truncation_invariant_holds(message_count: int, limits: Limits = DEFAULT_LIMITS) -> bool:
    """材料不足時改為驗證截斷機制本身：`truncated == (dropped_messages > 0)`。"""
    document = build_document(
        [Message(text=f"第 {index} 則訊息") for index in range(message_count)], limits
    )
    return document.truncated == (document.dropped_messages > 0)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _rate_text(rate: Rate | None) -> str:
    return "" if rate is None else f"{rate.value:.6f}"


def ablation_rows(contributions: Sequence[SignalContribution]) -> list[list[str]]:
    return [
        [
            item.name,
            item.kind,
            "yes" if item.registered else "no",
            _rate_text(item.hit_rate),
            f"{item.delta_false_positive_rate:.6f}",
            f"{item.delta_recall:.6f}",
            str(item.shadowed_count),
            _rate_text(item.random_recall),
            ""
            if item.distinguishable_from_random is None
            else ("yes" if item.distinguishable_from_random else "no"),
            item.note,
        ]
        for item in contributions
    ]


ABLATION_COLUMNS = (
    "name",
    "kind",
    "registered",
    "hit_rate",
    "delta_false_positive_rate",
    "delta_recall",
    "shadowed_count",
    "random_escalation_recall",
    "distinguishable_from_random",
    "note",
)

THRESHOLD_COLUMNS = (
    "threshold",
    "value",
    "headline_subset",
    "false_positive_rate",
    "recall",
    "abstention_rate",
)


def threshold_rows(rows: Sequence[tuple[str, float, Outcome]]) -> list[list[str]]:
    return [
        [
            key,
            f"{value:.4f}",
            outcome.headline_subset,
            f"{outcome.false_positive_rate.value:.6f}",
            f"{outcome.recall.value:.6f}",
            f"{outcome.abstention_rate.value:.6f}",
        ]
        for key, value, outcome in rows
    ]


@dataclass(frozen=True)
class SnapshotLookup:
    """一份**凍結的**網域年齡快照，供五個門檻在同一批資料上重新分類。

    五個門檻各查一次 RDAP 是錯的：查詢之間註冊局的回應可能不同，而那個差異
    會被記成門檻造成的差異。查一次、凍結、重分類五次，差異就只剩門檻。

    未在快照中的網域拋 `KeyError` —— 回一個 `NO_DATA` 會把「這一輪沒查到它」
    偽裝成「註冊局沒有這筆資料」。
    """

    ages: Mapping[str, DomainAge]

    def __call__(self, domain: str) -> DomainAge:
        if domain not in self.ages:
            raise KeyError(f"網域不在年齡快照中：{domain!r}")
        return self.ages[domain]


def holdout_domains(samples: Sequence[Sample], psl: PublicSuffixList) -> list[str]:
    """holdout 中全部可解析的可註冊網域，順序穩定。"""
    seen: list[str] = []
    for sample in samples:
        document = build_document([Message(text=sample.text)])
        for domain in group_by_registrable_domain(extract_urls(document, psl)):
            if domain not in seen:
                seen.append(domain)
    return seen


def domain_age_sweep(
    samples: Sequence[Sample],
    psl: PublicSuffixList,
    store: BlocklistStore,
    table: WeightTable,
    snapshot: SnapshotLookup,
) -> list[tuple[int, Rate, Outcome]]:
    """以五個門檻對同一份年齡快照重新分類。"""
    tables = load_tables()
    rows: list[tuple[int, Rate, Outcome]] = []
    for days in DOMAIN_AGE_THRESHOLDS:
        registry = build_registry(psl, store=store)
        register_domain_age_check(
            registry, psl, tables.shorteners, lookup=snapshot, threshold_days=days
        )
        records = run_over(samples, registry, table)
        rows.append(
            (
                days,
                Rate(
                    numerator=sum(1 for r in records if "domain_age" in r.hit_signals),
                    denominator=len(records),
                ),
                _outcome(records),
            )
        )
    return rows


def _domain_age_lines(
    rows: Sequence[tuple[int, Rate, Outcome]],
    attempted: int,
    resolved: int,
    outcomes: Mapping[str, int],
) -> list[str]:
    if not rows:
        return [
            "**未執行。** `domain_age` 在預設組裝下不註冊（需要注入 RDAP 查詢器），"
            "本輪未以 `--domain-age` 啟用。holdout 中可解析的可註冊網域共 "
            f"{attempted} 個。不以部分結果冒充完整結果。",
            "",
        ]
    lines = [
        f"批次查詢了 {attempted} 個可註冊網域，其中 {resolved} 個取得註冊日期；"
        f"結果分布：{'、'.join(f'{key} {value}' for key, value in sorted(outcomes.items()))}。",
        "",
        "同一份年齡快照以五個門檻重新分類（查一次、凍結、重分類五次 ——"
        "五個門檻各查一次 RDAP 會把註冊局回應的差異記成門檻造成的差異）：",
        "",
        "| 門檻（天） | domain_age 命中率 | 頭條誤判率 | 召回率 | 棄權率 |",
        "|---|---|---|---|---|",
    ]
    for days, hits, outcome in rows:
        lines.append(
            f"| {days} | {hits} | {outcome.false_positive_rate} | "
            f"{outcome.recall} | {outcome.abstention_rate} |"
        )
    return lines + [""]


def url_brand_sweep(
    samples: Sequence[Sample],
    psl: object,
    store: object,
    table: WeightTable,
) -> list[tuple[int, Rate, Outcome]]:
    """以 4/5/6 三個 `short_label_length` 重跑，記錄 `url_brand` 的命中率與整體表現。

    `add-url-check` 記錄「官方主標籤短於 5 個字元時不做編輯距離判定」的 5
    沒有實驗依據。三個值都在同一份 holdout 上跑，差別只有這一個參數。
    """
    rows: list[tuple[int, Rate, Outcome]] = []
    for length in URL_BRAND_SHORT_LABEL_LENGTHS:
        registry = build_registry(psl, store=store)  # type: ignore[arg-type]
        scoped = replace_check(
            registry,
            UrlBrandCheck(load_tables(), psl, short_label_length=length),  # type: ignore[arg-type]
        )
        records = run_over(samples, scoped, table)
        rows.append(
            (
                length,
                Rate(
                    numerator=sum(1 for r in records if "url_brand" in r.hit_signals),
                    denominator=len(records),
                ),
                _outcome(records),
            )
        )
    return rows


def _baseline_rows(
    records: Sequence[RunRecord], baseline_result: BaselineResult
) -> list[list[str]]:
    """規則系統 vs. TF-IDF，逐 ham 子集並列誤判率，另加召回率與棄權率。"""
    rows: list[list[str]] = []
    for name in HAM_SUBSETS:
        subset = [record for record in records if record.subset == name]
        if not subset or name not in baseline_result.false_positive_rates:
            continue
        report = subset_report(name, subset)
        assert report.false_positive_rate is not None
        rows.append(
            [
                name,
                "false_positive_rate",
                str(report.false_positive_rate),
                str(baseline_result.false_positive_rates[name]),
            ]
        )
    outcome = _outcome(records)
    rows.append(["cofacts_scam", "recall", str(outcome.recall), str(baseline_result.recall)])
    rows.append(
        [
            "all",
            "abstention_rate",
            str(outcome.abstention_rate),
            str(baseline_result.abstention_rate),
        ]
    )
    return rows


def _rule_rates(records: Sequence[RunRecord]) -> dict[str, Rate]:
    """規則系統在各 ham 子集上的誤判率，供與 TF-IDF 並列。"""
    rates: dict[str, Rate] = {}
    for name in HAM_SUBSETS:
        subset = [record for record in records if record.subset == name]
        if not subset:
            continue
        report = subset_report(name, subset)
        if report.false_positive_rate is not None:
            rates[name] = report.false_positive_rate
    return rates


def _handoff_markdown(
    *,
    run_id: str,
    seed: int,
    baseline_outcome: Outcome,
    split_outcome: Outcome,
    brand_rows: Sequence[tuple[int, Rate, Outcome]],
    contradiction: Rate,
    truncated: int,
    baseline_result: BaselineResult,
    rule_rates: Mapping[str, Rate],
    contributions: Sequence[SignalContribution],
    domain_age_lines: Sequence[str],
) -> str:
    """交辦清單章節：每條上游交辦一列，含處置或 blocked 理由。"""
    lines = [
        "# 消融：上游交辦清單與各自的結果",
        "",
        f"`run_id`：`{run_id}`；隨機對照組 `seed = {seed}`",
        "",
        "## 解讀限制（固定段落）",
        "",
        PLACEHOLDER_LIMITATION,
        "",
        "## baseline（holdout，全部訊號啟用）",
        "",
        f"- 頭條誤判率（{baseline_outcome.headline_subset}）："
        f"{baseline_outcome.false_positive_rate}",
        f"- 召回率：{baseline_outcome.recall}",
        f"- 棄權率：{baseline_outcome.abstention_rate}",
        "",
        "## 逐條交辦",
        "",
        "### `add-url-check`：`url_tld_risk` 的貢獻是否為零或負",
        "",
        _contribution_line(contributions, "url_tld_risk"),
        "",
        "### `add-url-check`：`url_brand` 的編輯距離門檻（4/5/6）",
        "",
        "| short_label_length | url_brand 命中率 | 頭條誤判率 | 召回率 |",
        "|---|---|---|---|",
    ]
    for length, hits, outcome in brand_rows:
        lines.append(f"| {length} | {hits} | {outcome.false_positive_rate} | {outcome.recall} |")
    lines += [
        "",
        "### `add-evasion-check`：`evasion_zhuyin` 的貢獻",
        "",
        _contribution_line(contributions, "evasion_zhuyin"),
        "",
        "### `add-context-limits`：`max_messages` / `max_chars`",
        "",
        f"holdout 中實際觸發截斷的樣本數：**{truncated}**。",
        "",
        "樣本不足，僅驗證截斷機制正確性，**未產出門檻敏感度曲線**。"
        if truncated < MULTI_MESSAGE_MINIMUM
        else "材料足夠，見 thresholds.csv。",
        "",
        f"截斷不變式（`truncated == (dropped_messages > 0)`）："
        f"{'成立' if truncation_invariant_holds(DEFAULT_LIMITS.max_messages + 5) else '不成立'}。",
        "",
        "### `scoring.py`：引述兩段式處置的真實代價",
        "",
        f"holdout 的 scam 子集上 `Score.contradicted` 為真的比例：{contradiction}。"
        f"這些樣本本來會被判為詐騙，卻因為同時命中引述而降級為「無法判定」。"
        f"逐則案例見 `add-metrics` 產出的 `cases.md`。",
        "",
        "### `add-weight-table`：6 個非單元群組「合併 vs 拆開」",
        "",
        f"把每個訊號各自視為獨立群組之後：頭條誤判率 "
        f"{split_outcome.false_positive_rate}、召回率 {split_outcome.recall}"
        f"（合併時為 {baseline_outcome.false_positive_rate} / {baseline_outcome.recall}）。",
        f"`game_code` 的逐訊號結果：{_contribution_line(contributions, 'game_code')}",
        "",
        "### `add-score-compute`：`prior_log_odds`",
        "",
        "已掃描（見 `thresholds.csv`），**不建議變更**：先驗一旦非 0，"
        "「完全無訊號時可能性為 0.5」這條語意就被打破，而那是一個語意決定。",
        "",
        "### `add-confidence`：八個信心門檻",
        "",
        "已逐一單變數掃描，每個切點皆同時報告誤判率、召回率與棄權率"
        "（見 `thresholds.csv`）。掃描範圍守住 `add-confidence` 的六條相對關係。",
        "",
        "### `add-domain-age`：30 天門檻（`add-blocklist-store` 的 44 天中位數涵蓋於此）",
        "",
        *domain_age_lines,
        "### 此刻不可執行的交辦",
        "",
    ]
    for key, reason in BLOCKED.items():
        lines.append(f"- **{key}**：{reason}")
    lines += [
        "",
        "## TF-IDF baseline",
        "",
        "| 子集 | 指標 | 規則系統 | TF-IDF |",
        "|---|---|---|---|",
    ]
    for row in _baseline_rows_for_markdown(baseline_result, baseline_outcome, rule_rates):
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        f"訓練集（`tune`）{baseline_result.train_size} 則。",
        "",
        EVIDENCE_NOTE,
        "",
    ]
    return "\n".join(lines) + "\n"


def _baseline_rows_for_markdown(
    baseline_result: BaselineResult, outcome: Outcome, rule_rates: Mapping[str, Rate]
) -> list[list[str]]:
    rows = [
        [name, "誤判率", str(rule_rates[name]), str(rate)]
        for name, rate in baseline_result.false_positive_rates.items()
        if name in rule_rates
    ]
    rows.append(["cofacts_scam", "召回率", str(outcome.recall), str(baseline_result.recall)])
    rows.append(
        ["all", "棄權率", str(outcome.abstention_rate), str(baseline_result.abstention_rate)]
    )
    return rows


def _contribution_line(contributions: Sequence[SignalContribution], name: str) -> str:
    for item in contributions:
        if item.name == name:
            if not item.registered:
                return f"`{name}`：{item.note}"
            return (
                f"`{name}`：命中率 {item.hit_rate}、關閉後 Δ召回率 "
                f"{item.delta_recall:+.4%}、Δ誤判率 {item.delta_false_positive_rate:+.4%}、"
                f"群組內被蓋過 {item.shadowed_count} 次、"
                f"與隨機對照組"
                + (
                    "無法區分"
                    if item.distinguishable_from_random is False
                    else ("可區分" if item.distinguishable_from_random else "未比較（零命中）")
                )
            )
    return f"`{name}`：未出現在消融結果中"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.eval.ablation",
        description="逐訊號與逐群組消融、隨機對照組、門檻掃描與 TF-IDF baseline。",
    )
    parser.add_argument("--testset-dir", default="testset")
    parser.add_argument("--data-dir", default="data/testset")
    parser.add_argument("--psl-dir", default="data/psl")
    parser.add_argument("--blocklist-dir", default="data/blocklist")
    parser.add_argument("--out-root", default="data/reports")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--domain-age",
        action="store_true",
        help=(
            "對 holdout 的全部可註冊網域批次查詢 RDAP 並掃描五個門檻。"
            "這是本專案唯一會對外查詢的評估步驟，因此**必須顯式開啟**。"
        ),
    )
    parser.add_argument("--rdap-dir", default="data/rdap")
    args = parser.parse_args(argv)

    testset_dir = Path(args.testset_dir)
    testset = load_testset(Path(args.data_dir), testset_dir / MANIFEST_FILENAME)
    samples = tuple(sample for sample in testset.all_samples() if sample.split == HOLDOUT)
    if not samples:
        print("holdout 為空，請先重建測試集。", file=sys.stderr)
        return 1

    psl = load_psl(Path(args.psl_dir))
    store = load_blocklist(Path(args.blocklist_dir), psl)
    table = load_weights()
    registry = build_registry(psl, store=store)
    run_id = build_run_id(
        testset_dir / MANIFEST_FILENAME, table, Path(args.blocklist_dir) / "manifest.json", store
    )
    out_dir = Path(args.out_root) / str(run_id) / "ablation"

    baseline = run_over(samples, registry, table)
    baseline_outcome = _outcome(baseline)
    print(
        f"baseline：誤判率 {baseline_outcome.false_positive_rate}"
        f"（{baseline_outcome.headline_subset}）、召回率 {baseline_outcome.recall}、"
        f"棄權率 {baseline_outcome.abstention_rate}",
        file=sys.stderr,
    )

    contributions: list[SignalContribution] = []
    for name in sorted(table.signals):
        contributions.append(
            contribution(
                name,
                "signal",
                [name],
                samples,
                registry,
                table,
                baseline,
                baseline_outcome,
                seed=args.seed,
            )
        )
        print(f"  訊號 {name} 完成", file=sys.stderr)
    for group, members in sorted(table.groups.items()):
        if len(members) <= 1:
            continue
        contributions.append(
            contribution(
                group,
                "group",
                list(members),
                samples,
                registry,
                table,
                baseline,
                baseline_outcome,
                seed=args.seed,
            )
        )
        print(f"  群組 {group} 完成", file=sys.stderr)

    _write_csv(out_dir / "ablation.csv", ABLATION_COLUMNS, ablation_rows(contributions))

    thresholds: list[tuple[str, float, Outcome]] = []
    for key, values in CONFIDENCE_SWEEPS.items():
        thresholds += threshold_sweep(key, values, samples, registry, table)
        print(f"  門檻 {key} 完成", file=sys.stderr)
    thresholds += prior_sweep(baseline, table, PRIOR_SWEEP)
    _write_csv(out_dir / "thresholds.csv", THRESHOLD_COLUMNS, threshold_rows(thresholds))

    advice = [
        recommend(
            item.name,
            item.kind,
            registered=item.registered,
            hit_rate=item.hit_rate,
            delta_recall=item.delta_recall,
            shadowed_count=item.shadowed_count,
            distinguishable_from_random=item.distinguishable_from_random,
        )
        for item in contributions
    ]
    recommendations_path = out_dir / "recommendations.csv"
    recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    with recommendations_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(DISCLAIMER + "\n")
        writer = csv.writer(stream)
        writer.writerow(RECOMMENDATION_COLUMNS)
        writer.writerows([item.name, item.kind, item.verdict, item.reason] for item in advice)

    baseline_result = train_and_evaluate(testset.all_samples())
    _write_csv(
        out_dir / "baseline_comparison.csv",
        ("subset", "metric", "rule_system", "tfidf_baseline"),
        _baseline_rows(baseline, baseline_result),
    )

    domain_age_rows: list[tuple[int, Rate, Outcome]] = []
    domains = holdout_domains(samples, psl)
    outcomes: dict[str, int] = {}
    resolved = 0
    if args.domain_age:
        lookup = RdapLookup(
            RdapBootstrap.load(Path(args.rdap_dir)),
            RdapCache(Path(args.rdap_dir) / "cache.sqlite3"),
            HttpsTransport(),
        )
        ages: dict[str, DomainAge] = {}
        for domain in domains:
            age = lookup(domain)
            ages[domain] = age
            outcomes[age.outcome.name] = outcomes.get(age.outcome.name, 0) + 1
            resolved += int(age.registered_on is not None)
        print(f"  RDAP 查詢完成：{len(ages)} 個網域，{outcomes}", file=sys.stderr)
        domain_age_rows = domain_age_sweep(samples, psl, store, table, SnapshotLookup(ages=ages))

    split_outcome = _outcome(run_over(samples, registry, split_groups(table)))
    brand_rows = url_brand_sweep(samples, psl, store, table)
    (out_dir / "handoff.md").write_text(
        _handoff_markdown(
            run_id=str(run_id),
            seed=args.seed,
            baseline_outcome=baseline_outcome,
            split_outcome=split_outcome,
            brand_rows=brand_rows,
            contradiction=contradiction_rate(baseline),
            truncated=truncation_count(samples),
            baseline_result=baseline_result,
            rule_rates=_rule_rates(baseline),
            contributions=contributions,
            domain_age_lines=_domain_age_lines(domain_age_rows, len(domains), resolved, outcomes),
        ),
        encoding="utf-8",
    )
    print(f"消融結果寫入 {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
