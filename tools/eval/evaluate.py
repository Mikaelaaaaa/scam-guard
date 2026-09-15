"""跑一次完整評估：對測試集執行 `detect()`，兩個切分各產出一份報告。

    python -m tools.eval.evaluate

**兩個切分都跑，而且都寫出來。** `add-testset` 已定：報告 MUST 同時列出
`tune` 與 `holdout` 上的每一個指標，兩者的差距就是過擬合的量。只寫一句
「holdout 未被逐則檢視」沒有任何人能驗證 —— 規則層的詞表是看著同一組
selector 的另一批樣本調出來的，holdout 是 held-out，不是 out-of-distribution。

產出寫進 `data/reports/<run_id>/`，該路徑被 git 忽略。
"""

import argparse
import sys
from pathlib import Path

from scam_guard.weights import load_weights
from tools.eval.dataset import MANIFEST_FILENAME, load_testset
from tools.eval.report import SPLITS, render_report
from tools.eval.run import (
    build_registry,
    build_run_id,
    load_blocklist,
    load_psl,
    run_over,
    write_records,
)

DEFAULT_TESTSET_DIR = Path("testset")
DEFAULT_DATA_DIR = Path("data/testset")
DEFAULT_PSL_DIR = Path("data/psl")
DEFAULT_BLOCKLIST_DIR = Path("data/blocklist")
DEFAULT_REPORT_ROOT = Path("data/reports")

MISSING_SUBSET_NOTE = (
    "下列子集未蒐集，其誤判率**未被量測**：{names}。"
    "`self_sms_ham` 是唯一覆蓋真實銀行、物流、電信與政府通知的子集，"
    "而那正是 solicit_otp、parcel_notice、atm_operation 與 secrecy_demand "
    "最可能誤判的一類。頭條誤判率取自 Cofacts 的兩個 ham 子集，"
    "MUST NOT 被當成這一類的代理值。"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.eval.evaluate",
        description="對測試集跑一次 detect() 並產出報告（tune 與 holdout 各一份）。",
    )
    parser.add_argument("--testset-dir", default=str(DEFAULT_TESTSET_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--psl-dir", default=str(DEFAULT_PSL_DIR))
    parser.add_argument("--blocklist-dir", default=str(DEFAULT_BLOCKLIST_DIR))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args(argv)

    testset_dir = Path(args.testset_dir)
    manifest_path = testset_dir / MANIFEST_FILENAME
    testset = load_testset(Path(args.data_dir), manifest_path)
    if not testset.subsets:
        print(
            f"測試集尚未重建：{args.data_dir} 為空。請先執行 "
            f"`python -m tools.eval.build_testset rebuild`。",
            file=sys.stderr,
        )
        return 1

    psl = load_psl(Path(args.psl_dir))
    store = load_blocklist(Path(args.blocklist_dir), psl)
    table = load_weights()
    registry = build_registry(psl, store=store)
    run_id = build_run_id(manifest_path, table, Path(args.blocklist_dir) / "manifest.json", store)

    samples = testset.all_samples()
    records = run_over(samples, registry, table)
    texts = {sample.id: sample.text for sample in samples}

    out_root = Path(args.report_root) / str(run_id)
    write_records(out_root / "records.jsonl", records)
    for split in SPLITS:
        report = render_report(
            records,
            texts,
            table,
            run_id,
            out_root / split,
            split=split,
            missing_subsets=testset.missing,
            notes=(MISSING_SUBSET_NOTE.format(names="、".join(testset.missing)),)
            if testset.missing
            else (),
        )
        name, rate = report.headline_false_positive_rate
        print(
            f"[{split}] 頭條誤判率 {name} {rate}；棄權率 {report.subsets[name].abstention_rate}",
            file=sys.stderr,
        )
    print(f"報告寫入 {out_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
