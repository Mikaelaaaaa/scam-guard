"""主動尋找「宣導文含 165 黑名單網址」的樣本。

    python -m tools.eval.promo_scan

`add-score-compute` 的黑名單例外條款（「`blocklist_exact` 的硬證據命中
MUST NOT 標記矛盾」）建立在一句**沒有量測支撐**的推論上：

> 防詐宣導文不會把仍在運作的惡意網址原樣貼出來。

它自己寫下：「`add-testset` MUST 在 hard negative 中特別檢查這種樣本是否存在；
若存在，例外條款要重審。」本模組是那條檢查。

結果三種，各有明確後果：

- **0 則** —— 例外條款的推論在這份語料上成立。報告寫樣本數與零命中的
  95% Wilson 上界，**不得**寫「不存在這種樣本」。
- **1 至數則** —— 逐則人工檢視，區分「確為宣導文貼出黑名單網址」與
  「其實是詐騙被誤標為 ham」；前者交回 `add-score-compute` 重審例外條款。
- **大量** —— 例外條款錯誤。這是計分層的 bug，不是測試集的問題。

**只跑兩個檢查，不跑整條 pipeline。** 這個查詢問的是「這兩個訊號會不會同時
命中」，跑整條 pipeline 會把答案埋進一個 `Verdict` 裡，還得先組出一份完整的
registry 與權重表。
"""

import argparse
import json
import sys
from pathlib import Path

from scam_guard.blocklist import BlocklistStore
from scam_guard.normalize import build_document
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.types import Request
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import UrlBlocklistCheck, load_tables
from tools.eval.dataset import load_testset, write_jsonl
from tools.eval.selectors import HAM_SUBSETS

DEFAULT_OUT_DIR = Path("data/testset")
DEFAULT_TESTSET_DIR = Path("testset")
DEFAULT_PSL_DIR = Path("data/psl")
DEFAULT_BLOCKLIST_DIR = Path("data/blocklist")
OUT_FILENAME = "promo_with_blocklist.jsonl"

MAX_AGE_DAYS = 400
"""本機快照的最大年齡。

評估是離線、一次性、由人執行的批次工作，用的是**當下手上這一份**快照 ——
它的新鮮度是一個要被記錄的事實（`run_id` 含黑名單 manifest 的 `data_through`
與 sha256），不是一個要在這裡擋下來的條件。放寬到 400 天是為了讓一份舊快照
仍然跑得完並如實報出它有多舊，而不是讓評估在快照過期時安靜地少一個訊號。
"""


def scan(testset_dir: Path, out_dir: Path, psl_dir: Path, blocklist_dir: Path) -> tuple[int, int]:
    """回傳 `(同時命中的筆數, 掃描的 ham 樣本總數)`。清單即使是空的也會被寫出。"""
    psl = PublicSuffixList.load(psl_dir, max_age_days=MAX_AGE_DAYS)
    store = BlocklistStore.load(blocklist_dir, psl, max_age_days=MAX_AGE_DAYS)
    blocklist_check = UrlBlocklistCheck(store, psl, load_tables())
    quotation_check = QuotationCheck()

    testset = load_testset(out_dir, testset_dir / "manifest.json")
    rows: list[dict] = []
    scanned = 0
    for subset in HAM_SUBSETS:
        if subset in testset.missing:
            continue
        for sample in testset.samples(subset):
            scanned += 1
            request = Request.from_text(sample.text)
            document = build_document(request.messages)
            blocklist_hits = [result for result in blocklist_check(request, document) if result.hit]
            if not blocklist_hits:
                continue
            quotation_hits = [result for result in quotation_check(request, document) if result.hit]
            if not quotation_hits:
                continue
            rows.append(
                {
                    "id": sample.id,
                    "subset": sample.subset,
                    "split": sample.split,
                    "blocklist_detail": [result.detail for result in blocklist_hits],
                    "blocklist_hard": [result.hard for result in blocklist_hits],
                    "quotation_detail": [result.detail for result in quotation_hits],
                }
            )
    write_jsonl(out_dir / OUT_FILENAME, rows)
    return len(rows), scanned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.eval.promo_scan",
        description="列出同時命中黑名單與引述的 ham 樣本。清單即使為空也會寫出。",
    )
    parser.add_argument("--testset-dir", default=str(DEFAULT_TESTSET_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--psl-dir", default=str(DEFAULT_PSL_DIR))
    parser.add_argument("--blocklist-dir", default=str(DEFAULT_BLOCKLIST_DIR))
    args = parser.parse_args(argv)

    found, scanned = scan(
        Path(args.testset_dir), Path(args.out_dir), Path(args.psl_dir), Path(args.blocklist_dir)
    )
    print(
        json.dumps(
            {
                "both_hit": found,
                "ham_samples_scanned": scanned,
                "out": str(Path(args.out_dir) / OUT_FILENAME),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
