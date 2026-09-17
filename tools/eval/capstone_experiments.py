"""專題兩軌實驗 + 分類器鑑別力 + TF-IDF 基準(2026-09-17)。

評測harness 的 `build_registry()` 只掛規則 + 網址;本腳本額外掛上 `ngram_classifier`
(Track A/B 的「本系統」),並可選擇性掛上 Gemini 語意層(Track B),用來量三層各自的
邊際貢獻。**不修改核心 `build_registry()` 的語意**——既有報告是規則層基線,本腳本是
另一組明確標示「含分類器 / 含 LLM」的量測。

前置:先重建測試集內容(`python -m tools.eval.build_testset rebuild`);Track B 需要
`GEMINI_API_KEY`(伺服器端語意層)。

    python -m tools.eval.capstone_experiments tracks [--sample-n 300] [--delay 4.0]
    python -m tools.eval.capstone_experiments discrimination
    python -m tools.eval.capstone_experiments baseline

`tracks` 的 `--delay` 是 Track B 每次 Gemini 呼叫之間的秒數;免費層 RPM 有限,一次
burst 打數百筆會被 429 限速,建議 ~4 秒(約 15 RPM)。
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import sleep

from tools.eval.baseline_tfidf import train_and_evaluate
from tools.eval.dataset import HOLDOUT, MANIFEST_FILENAME, load_testset
from tools.eval.run import build_registry, load_blocklist, load_psl
from tools.eval.stats import Rate
from scam_guard.ngram import (
    NGRAM_THRESHOLD,
    document_text,
    load_model,
    register_ngram_check,
    score,
)
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.pipeline import detect
from scam_guard.types import Message, Request
from scam_guard.weights import load_weights

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "testset"
MANIFEST_PATH = ROOT / "testset" / MANIFEST_FILENAME
PSL_DIR = ROOT / "data" / "psl"
BLOCKLIST_DIR = ROOT / "data" / "blocklist"


def _load():
    psl = load_psl(PSL_DIR)
    store = load_blocklist(BLOCKLIST_DIR, psl)
    table = load_weights()
    testset = load_testset(DATA_DIR, MANIFEST_PATH)
    return psl, store, table, testset


def _registry_with_ngram(psl, store, table):
    """規則 + 網址 + 分類器(不含 LLM)。"""
    registry = build_registry(psl, store=store)
    register_ngram_check(registry, table, model=load_model())
    return registry


def _judge(sample, registry, table):
    request = Request(messages=[Message(text=sample.text)])
    return detect(request, registry, table, short_circuit=False, limits=DEFAULT_LIMITS)


def _rate_dict(numerator, denominator):
    rate = Rate(numerator=numerator, denominator=denominator)
    return {"value": rate.value, "lower": rate.lower, "upper": rate.upper,
            "num": numerator, "den": denominator}


def _holdout(testset, subset):
    return [s for s in testset.samples(subset) if s.split == HOLDOUT]


def cmd_tracks(args):
    from llm_runtime.gemini import GeminiCallFailed, GeminiRuntime
    from scam_guard.llm.check import LlmCheck
    from scam_guard.llm.prompt import DEFAULT_BUDGET
    from scam_guard.llm.validate import LlmOutcomeCounter

    psl, store, table, testset = _load()
    reg_a = _registry_with_ngram(psl, store, table)

    # Track A:全 holdout,無 LLM。
    track_a = {}
    for subset in testset.subsets:
        samples = _holdout(testset, subset)
        if not samples:
            continue
        decided = sum(_judge(s, reg_a, table).scam_probability is not None for s in samples)
        n = len(samples)
        track_a[subset] = {
            "label": samples[0].label, "n": n,
            "decided_rate": _rate_dict(decided, n),
            "abstention_rate": (n - decided) / n,
        }

    # Track B:分層決定性抽樣 --sample-n,±Gemini 對照。
    reg_b = _registry_with_ngram(psl, store, table)
    reg_b.register(LlmCheck(runtime=GeminiRuntime(), counter=LlmOutcomeCounter(),
                            table=table, budget=DEFAULT_BUDGET, deadline_s=45.0))
    picked = _stratified_sample(testset, args.sample_n)
    rows = []
    failed = 0
    for sample in picked:
        v_a = _judge(sample, reg_a, table)
        try:
            b_decided = _judge(sample, reg_b, table).scam_probability is not None
        except GeminiCallFailed:
            b_decided = v_a.scam_probability is not None
            failed += 1
        rows.append({"label": sample.label,
                     "a": v_a.scam_probability is not None, "b": b_decided})
        if args.delay > 0:
            sleep(args.delay)

    scam = [r for r in rows if r["label"] == "scam"]
    ham = [r for r in rows if r["label"] == "ham"]
    track_b = {
        "n": len(rows), "llm_failed": failed, "delay_s": args.delay,
        "scam_recall_3layer": _rate_dict(sum(r["a"] for r in scam), len(scam)),
        "scam_recall_4layer": _rate_dict(sum(r["b"] for r in scam), len(scam)),
        "ham_fpr_3layer": _rate_dict(sum(r["a"] for r in ham), len(ham)),
        "ham_fpr_4layer": _rate_dict(sum(r["b"] for r in ham), len(ham)),
        "rescued_scam": sum(1 for r in scam if (not r["a"]) and r["b"]),
        "added_false_positive": sum(1 for r in ham if (not r["a"]) and r["b"]),
    }
    return {"track_a": track_a, "track_b": track_b, "missing": list(testset.missing)}


def _stratified_sample(testset, sample_n):
    by_subset = {
        subset: sorted(_holdout(testset, subset),
                       key=lambda s: hashlib.sha256(s.id.encode()).hexdigest())
        for subset in testset.subsets
    }
    by_subset = {k: v for k, v in by_subset.items() if v}
    total = sum(len(v) for v in by_subset.values())
    picked = []
    for samples in by_subset.values():
        picked.extend(samples[: round(sample_n * len(samples) / total)])
    return picked


def _auc(positives, negatives):
    """Mann-Whitney AUC = P(隨機 scam 分數 > 隨機 ham 分數);平手給 0.5。"""
    labelled = sorted([(v, 1) for v in positives] + [(v, 0) for v in negatives])
    ranks = [0.0] * len(labelled)
    i = 0
    while i < len(labelled):
        j = i
        while j < len(labelled) and labelled[j][0] == labelled[i][0]:
            j += 1
        average = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = average
        i = j
    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, labelled) if lab == 1)
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cmd_discrimination(args):
    _, _, table, testset = _load()
    model = load_model()
    threshold = table.threshold(NGRAM_THRESHOLD)

    scores = {}
    for subset in testset.subsets:
        values = []
        for sample in _holdout(testset, subset):
            doc = build_document([Message(text=sample.text)], DEFAULT_LIMITS)
            values.append(score(model, document_text(doc)).value)
        if values:
            scores[subset] = values

    scam = scores.get("cofacts_scam", [])
    cofacts_ham = scores.get("cofacts_ham_ad", []) + scores.get("cofacts_ham_suspected", [])
    conversation = scores.get("conversation_ham", [])
    all_ham = cofacts_ham + conversation

    def fire(values):
        return sum(1 for v in values if v >= threshold) / len(values) if values else None

    return {
        "threshold": threshold,
        "n": {k: len(v) for k, v in scores.items()},
        "classifier_fire_rate": {
            "cofacts_scam": fire(scam),
            "cofacts_ham(ad+suspected)": fire(cofacts_ham),
            "conversation_ham": fire(conversation),
        },
        "auc": {
            "scam_vs_all_ham": _auc(scam, all_ham) if scam and all_ham else None,
            "scam_vs_cofacts_ham": _auc(scam, cofacts_ham) if scam and cofacts_ham else None,
            "scam_vs_conversation": _auc(scam, conversation) if scam and conversation else None,
        },
    }


def cmd_baseline(args):
    _, _, _, testset = _load()
    result = train_and_evaluate(testset.all_samples())

    def rate(r):
        return {"value": r.value, "lower": r.lower, "upper": r.upper,
                "num": r.numerator, "den": r.denominator}

    return {
        "recall": rate(result.recall),
        "false_positive_rates": {k: rate(v) for k, v in result.false_positive_rates.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.eval.capstone_experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)
    tracks = sub.add_parser("tracks", help="Track A(全 holdout,無 LLM)+ Track B(抽樣,±Gemini)")
    tracks.add_argument("--sample-n", type=int, default=300)
    tracks.add_argument("--delay", type=float, default=4.0,
                        help="Track B 每次 Gemini 呼叫間的秒數(避開免費層 RPM 限速)")
    sub.add_parser("discrimination", help="分類器單獨 fire rate + AUC")
    sub.add_parser("baseline", help="TF-IDF 基準(char (2,4) + LogReg)")
    args = parser.parse_args(argv)

    if not load_testset(DATA_DIR, MANIFEST_PATH).subsets:
        parser.error(f"測試集尚未重建:{DATA_DIR} 為空。先跑 "
                     "`python -m tools.eval.build_testset rebuild`。")

    handler = {"tracks": cmd_tracks, "discrimination": cmd_discrimination,
               "baseline": cmd_baseline}[args.cmd]
    print(json.dumps(handler(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
