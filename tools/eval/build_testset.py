"""建立與重建測試集。

    # 第一次：抓池子、選子集、寫出版控的識別字清單與 manifest
    python -m tools.eval.build_testset select

    # 之後：由識別字清單重建內容，逐則比對 sha256
    python -m tools.eval.build_testset rebuild

**版控的只有公開識別字與雜湊，內容不進版控。** `testset/` 放在 repo 根目錄，
**不是** `data/` 之下 —— `.gitignore` 第 222 行是 `data/`，無前導斜線、任何深度
生效，`add-url-check` 與 `add-weight-table` 已各踩過一次。內容一律寫進
`data/testset/`，且每一個產出檔都以 `git check-ignore` **實測**被忽略，
不以目視確認 `.gitignore` 取代。

**重建時雜湊不符一律大聲失敗**：非零結束碼、列出全部不符的識別字與原因，
不跳過該則繼續。接受差異是一個**明確的人為動作**（`rebuild --accept-drift`），
被接受的差異寫進 manifest 與識別字清單，MUST NOT 靜默套用。

**可重現性有到期日。** Cofacts 的文章可被編輯與撤銷，時間夠久之後任何一份
id 清單都會 drift。報告 MUST 記錄 `rebuilt_at` 與接受的 drift 筆數；
drift 超過 1% 時，該份報告的數字 MUST NOT 與更早的報告直接比較。
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from tools.cofacts_fetch import CofactsFetchError, fetch_selector
from tools.eval.dataset import (
    IDS_FIELDS,
    IDS_FILENAME,
    MANIFEST_FILENAME,
    Sample,
    file_sha256,
    read_jsonl,
    split_counts,
    subset_path,
    text_sha256,
    write_jsonl,
)
from tools.eval.selectors import (
    COFACTS_DERIVED_NAMES,
    COFACTS_SUBSETS,
    CONVERSATION_HAM,
    EXCLUDED_POOLS,
    LABEL_SCAM,
    MULTI_MESSAGE_HAM,
    MULTI_MESSAGE_SCAM,
    NO_BALANCED_SAMPLING,
    NO_TYPE_STRATIFICATION,
    SELF_SUBSETS,
    SUBSETS,
    SYNTH_LOGISTICS_REFUSAL,
    Subset,
)
from tools.message_filter import flag_message

DEFAULT_TESTSET_DIR = Path("testset")
DEFAULT_OUT_DIR = Path("data/testset")
DEFAULT_POOL_DIR = Path("data/testset/pools")
DEFAULT_DEV_SAMPLE = Path("data/dev_sample.jsonl")
DEFAULT_CONVERSATION_DIR = Path("data/conversation")
CONVERSATION_SUBSET_FILENAME = "conversation_ham.jsonl"
CONVERSATION_MANIFEST_FILENAME = "manifest.json"

DRIFT_FILENAME = "drift.json"
SNIPPET_LENGTH = 40
DRIFT_COMPARABLE_SHARE = 0.01
"""drift 超過此比例時，該份報告的數字不得與更早的報告直接比較。"""

MULTI_MESSAGE_FLAG = "line_export"
MULTI_MESSAGE_SUFFICIENT_MINIMUM = 10
"""少於此數時 manifest 標記 `sufficient = false`，多則情境的任何指標一律不報。

`add-message-filter` 在 1,200 則中只標到 1 則（0.2%），而它自己記錄「真實比例
可能是 0.05% 也可能是 1%」。因此樣本量在建立測試集之前就是未知的，
而「少於 10 則就不報指標」這條規則 MUST 在看到數字之前定好 ——
定在之後就成了看著結果挑門檻。
"""


class TestsetBuildError(Exception):
    """建立或重建失敗。訊息一律含足以定位問題的資訊。"""


def _require_git_ignored(path: Path) -> None:
    """產出路徑必須被 git 忽略。以 `git check-ignore` 實測，不目視確認 `.gitignore`。"""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        return
    if result.returncode == 1:
        raise TestsetBuildError(
            f"產出路徑 {path} 未被 git 忽略。測試集的內容是真實民眾送進 Cofacts 的訊息，"
            f"且 Cofacts 開放資料為 CC BY-SA 4.0，不得進版控。請改用 data/ 之下的路徑。"
        )
    raise TestsetBuildError(
        f"git check-ignore 無法判定 {path}："
        f"returncode={result.returncode}、stderr={result.stderr.strip()!r}"
    )


def _require_not_git_ignored(path: Path) -> None:
    """版控目錄必須**不**被 git 忽略。`data/` 這條忽略規則在任何深度生效。"""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode == 1:
        return
    if result.returncode == 0:
        raise TestsetBuildError(
            f"版控目錄 {path} 被 git 忽略。識別字清單與 manifest 必須進版控，"
            f"否則測試集無法被第三人重建。目錄名不得為 data 或位於任何名為 data 的目錄之下。"
        )
    raise TestsetBuildError(
        f"git check-ignore 無法判定 {path}："
        f"returncode={result.returncode}、stderr={result.stderr.strip()!r}"
    )


def _pool_path(pool_dir: Path, subset: Subset) -> Path:
    return pool_dir / f"pool_{subset.name}.jsonl"


def fetch_pool(subset: Subset, pool_dir: Path, *, reuse: bool) -> list[dict]:
    """抓一個 Cofacts 池子的**全部**文章。

    抓全池而不是抓到目標筆數為止，因為 `multi_message` 的樣本來源 MUST 是完整
    訊息池：`add-message-filter` 在開發樣本的 1,200 則中只標到 1 則，
    在六頁裡找一個 0.2% 的東西找不到。

    `reuse` 為真且池檔已存在時**不重抓**。這是一個人明確指定的旗標，不是自動
    判斷 —— 自動重用會讓「這份數字是哪一天的池子算的」變成沒有人知道的事。
    """
    if subset.cofacts_filter is None:
        raise TestsetBuildError(f"子集 {subset.name!r} 不是由 Cofacts selector 定義的，無法抓取")
    path = _pool_path(pool_dir, subset)
    if reuse and path.is_file():
        print(f"重用既有池檔：{path}", file=sys.stderr)
        return read_jsonl(path)
    _require_git_ignored(path)
    fetch_selector(
        dict(subset.cofacts_filter),
        subset.name,
        path,
        f"python -m tools.eval.build_testset select --pool-dir {path.parent}",
    )
    return read_jsonl(path)


def _dev_ids(dev_sample: Path) -> set[str]:
    """開發樣本的識別字集合。缺檔即 raise —— 沒有它就無法斷言交集為空。"""
    if not dev_sample.is_file():
        raise TestsetBuildError(
            f"開發樣本不存在：{dev_sample}。測試集 MUST 與開發樣本的識別字不相交，"
            f"缺少它就無法驗證這件事。產生方式見 tests/test_dev_sample_smoke.py 的 skip 訊息。"
        )
    return {record["id"] for record in read_jsonl(dev_sample)}


def _is_multi_message(text: str) -> bool:
    return MULTI_MESSAGE_FLAG in flag_message(text)


def select(
    pool_dir: Path,
    dev_sample: Path,
    *,
    reuse_pools: bool,
) -> tuple[dict[str, list[dict]], dict[str, object]]:
    """挑出六個子集，回傳 `(子集 → 紀錄, 選取統計)`。

    一則同時符合某個扁平子集與 `line_export` 時**只歸入 multi_message**：
    重複計入會讓同一則在兩個分母裡各算一次。受影響的筆數是個位數，
    但「每一則恰好屬於一個子集」這個性質讓 manifest 的筆數可以直接相加。

    `cofacts_ham_suspected` **全取不取樣**（`target` 為 `None`）；
    另外兩個取到 `target` 為止，順序即 `createdAt DESC`。
    不做平衡取樣，理由見 `selectors.NO_BALANCED_SAMPLING`。
    """
    excluded = _dev_ids(dev_sample)
    chosen: dict[str, list[dict]] = {name: [] for name in SUBSETS}
    stats: dict[str, object] = {}
    for subset in COFACTS_SUBSETS:
        pool = fetch_pool(subset, pool_dir, reuse=reuse_pools)
        overlap = 0
        flagged = 0
        for record in pool:
            multi_flagged = _is_multi_message(record["text"])
            flagged += int(multi_flagged)
            if record["id"] in excluded:
                overlap += 1
                continue
            if multi_flagged:
                multi = MULTI_MESSAGE_SCAM if subset.label == LABEL_SCAM else MULTI_MESSAGE_HAM
                chosen[multi.name].append(record)
                continue
            if subset.target is not None and len(chosen[subset.name]) >= subset.target:
                continue
            chosen[subset.name].append(record)
        stats[subset.name] = {
            "pool_size": len(pool),
            "removed_dev_overlap": overlap,
            "selected": len(chosen[subset.name]),
            "target": subset.target,
            "line_export_in_pool": flagged,
        }
    return chosen, stats


def _sample_record(record: Mapping[str, object], subset: str) -> dict:
    """一則落地紀錄。**不含 label、不含類型欄位**，兩者的理由見 `dataset` 與
    `selectors.NO_TYPE_STRATIFICATION`。`source_uri` 是 CC BY-SA 4.0 的姓名標示。"""
    return {
        "id": record["id"],
        "text": record["text"],
        "subset": subset,
        "source_uri": record["source_uri"],
    }


def write_subsets(out_dir: Path, chosen: Mapping[str, Sequence[Mapping[str, object]]]) -> dict:
    """寫出 `data/testset/<subset>.jsonl`，回傳每個子集的筆數與切分分布。"""
    written: dict[str, object] = {}
    for name, records in chosen.items():
        if not records:
            continue
        path = subset_path(out_dir, name)
        _require_git_ignored(path)
        write_jsonl(path, (_sample_record(record, name) for record in records))
        samples = [Sample(id=r["id"], text=r["text"], subset=name) for r in records]
        written[name] = {"count": len(records), "splits": split_counts(samples)}
    return written


def write_ids(testset_dir: Path, chosen: Mapping[str, Sequence[Mapping[str, object]]]) -> int:
    """寫出版控的識別字清單。每行只有識別字、子集名稱與內容雜湊。"""
    path = testset_dir / IDS_FILENAME
    _require_not_git_ignored(path)
    rows = [
        {"id": record["id"], "subset": name, "text_sha256": text_sha256(record["text"])}
        for name, records in chosen.items()
        if name in COFACTS_DERIVED_NAMES
        for record in records
    ]
    return write_jsonl(path, rows)


def _self_subset_status(out_dir: Path) -> dict:
    """自建子集的狀態。**缺檔就是缺檔**，不以空集合冒充已蒐集。

    自建子集可驗證但不可重現：有檔案時記錄 sha256 與筆數（證明跑完實驗之後
    沒有被改過），沒有檔案時記錄它沒有被蒐集 —— 而報告 MUST 據此寫出
    「最危險的那一類誤判未被量測」，MUST NOT 以其餘子集的合併數字取代它。
    """
    status: dict[str, object] = {}
    for subset in SELF_SUBSETS:
        path = subset_path(out_dir, subset.name)
        if not path.is_file():
            status[subset.name] = {
                "collected": False,
                "target": subset.target,
                "note": "未蒐集。此子集不可重現（需要蒐集者本人的手機），缺席 MUST 於報告中明說。",
            }
            continue
        status[subset.name] = {
            "collected": True,
            "count": len(read_jsonl(path)),
            "target": subset.target,
            "file_sha256": file_sha256(path),
        }
    return status


def _conversation_status(out_dir: Path, conversation_dir: Path) -> dict:
    """把對話語料落地為 `conversation_ham` 子集並回傳狀態與語料 pinning。

    **可重現但非 Cofacts**：語料由 `tools/fetch_conversation_corpus.py` 從 pinned
    版本的公開非 gated 語料經內容閘門過濾與精確去重產生，因此**不進版控的識別字
    清單**（那份只放 Cofacts 識別字，見 `write_ids`），改由本狀態把語料的
    release/commit、是否轉繁、過濾與去重規則、子集 sha256 記入 `testset/manifest.json`，
    使第三人可重建同一份輸入。切分不在此處 —— 由 `dataset.Sample.split` 依
    `sha256(正規化文字)` 首位元組確定性計算。

    語料未取得（`data/conversation/` 無檔）時回傳 `collected = False`，MUST 於報告
    中明說對話這一類的誤判未被量測，MUST NOT 以其餘子集冒充。
    """
    subset_source = conversation_dir / CONVERSATION_SUBSET_FILENAME
    manifest_source = conversation_dir / CONVERSATION_MANIFEST_FILENAME
    if not subset_source.is_file():
        return {
            "collected": False,
            "target": CONVERSATION_HAM.target,
            "note": (
                "未取得。執行 `python -m tools.fetch_conversation_corpus`；"
                "缺席時對話這一類的誤判未被量測，MUST 於報告中明說。"
            ),
        }
    records = read_jsonl(subset_source)
    out_path = subset_path(out_dir, CONVERSATION_HAM.name)
    _require_git_ignored(out_path)
    write_jsonl(
        out_path,
        (
            {"id": record["id"], "text": record["text"], "subset": CONVERSATION_HAM.name}
            for record in records
        ),
    )
    samples = [
        Sample(id=record["id"], text=record["text"], subset=CONVERSATION_HAM.name)
        for record in records
    ]
    corpus_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    return {
        "collected": True,
        "count": len(records),
        "target": CONVERSATION_HAM.target,
        "splits": split_counts(samples),
        "file_sha256": file_sha256(out_path),
        "corpus": corpus_manifest["corpus"],
        "repo": corpus_manifest["repo"],
        "commit": corpus_manifest["commit"],
        "license": corpus_manifest["license"],
        "converted_to_traditional": corpus_manifest["converted_to_traditional"],
        "opencc_config": corpus_manifest["opencc_config"],
        "content_gate": corpus_manifest["content_gate"],
        "corpus_counts": corpus_manifest["counts"],
        "subset_sha256": corpus_manifest["subset_sha256"],
    }


def build_manifest(
    written: Mapping[str, object],
    stats: Mapping[str, object],
    dev_sample: Path,
    out_dir: Path,
    dev_intersection: int,
    conversation_dir: Path,
) -> dict:
    """組出版控的 manifest。selector 常數寫進去，使子集的定義與筆數在同一份檔案裡。"""
    multi_total = sum(
        int(written[name]["count"])  # type: ignore[index]
        for name in (MULTI_MESSAGE_SCAM.name, MULTI_MESSAGE_HAM.name)
        if name in written
    )
    return {
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selectors": {
            subset.name: dict(subset.cofacts_filter)
            for subset in COFACTS_SUBSETS
            if subset.cofacts_filter is not None
        },
        "selection": dict(stats),
        "subsets": dict(written),
        "self_built": _self_subset_status(out_dir),
        "conversation_ham": _conversation_status(out_dir, conversation_dir),
        "multi_message": {
            "count": multi_total,
            "sufficient": multi_total >= MULTI_MESSAGE_SUFFICIENT_MINIMUM,
            "minimum": MULTI_MESSAGE_SUFFICIENT_MINIMUM,
            "line_export_in_full_pools": sum(
                int(stats[name]["line_export_in_pool"])  # type: ignore[index]
                for name in stats
            ),
            "note": (
                "樣本數少於 minimum 時，多則情境的任何指標一律不報，只列逐則輸出。"
                "這條規則在看到數字之前就定好，定在之後就成了看著結果挑門檻。"
            ),
        },
        "dev_sample": {
            "path": str(dev_sample),
            "file_sha256": file_sha256(dev_sample),
            "id_intersection": dev_intersection,
            "note": (
                "識別字交集為 0，但兩者來自同一組 selector 與相近的時間窗。"
                "holdout **未被逐則檢視**，卻**與調過門檻的樣本同分布** —— "
                "規則層的詞表是看著開發樣本補充的。報告 MUST 同時列出 tune 與 "
                "holdout 上的每一個指標，兩者的差距就是過擬合的量。"
            ),
        },
        "excluded_pools": dict(EXCLUDED_POOLS),
        "not_done": {
            "synth_logistics": SYNTH_LOGISTICS_REFUSAL,
            "type_stratification": NO_TYPE_STRATIFICATION,
            "balanced_sampling": NO_BALANCED_SAMPLING,
        },
        "accepted_drift": [],
    }


def _load_ids(testset_dir: Path) -> list[dict]:
    path = testset_dir / IDS_FILENAME
    if not path.is_file():
        raise TestsetBuildError(f"識別字清單不存在：{path}。請先執行 `select`。")
    rows = read_jsonl(path)
    for index, row in enumerate(rows, start=1):
        for field in IDS_FIELDS:
            if field not in row:
                raise TestsetBuildError(f"{path} 第 {index} 行缺少欄位 {field!r}")
        if "label" in row:
            raise TestsetBuildError(
                f"{path} 第 {index} 行含 label 欄位。標籤由 subset 決定，不得寫入識別字清單。"
            )
    return rows


def _fetch_by_id(pool_dir: Path, *, reuse: bool) -> dict[str, dict]:
    """重抓三個池子並攤平成 `id → 紀錄`。重建只認識別字，不認它原本在哪一頁。"""
    found: dict[str, dict] = {}
    for subset in COFACTS_SUBSETS:
        for record in fetch_pool(subset, pool_dir, reuse=reuse):
            found[record["id"]] = record
    return found


def detect_drift(rows: Sequence[Mapping[str, object]], found: Mapping[str, dict]) -> dict:
    """逐則比對，回傳 `{"deleted": [...], "edited": [...]}`。

    `edited` 帶兩段內文的前 40 字元，使人得以在接受之前看出改了什麼。
    """
    deleted: list[str] = []
    edited: list[dict] = []
    for row in rows:
        article_id = str(row["id"])
        if article_id not in found:
            deleted.append(article_id)
            continue
        actual = text_sha256(found[article_id]["text"])
        if actual != row["text_sha256"]:
            edited.append(
                {
                    "id": article_id,
                    "expected_sha256": row["text_sha256"],
                    "actual_sha256": actual,
                    "actual_prefix": found[article_id]["text"][:SNIPPET_LENGTH],
                }
            )
    return {"deleted": deleted, "edited": edited}


def _accepted(accept_path: Path | None) -> dict:
    if accept_path is None:
        return {"deleted": [], "edited": []}
    if not accept_path.is_file():
        raise TestsetBuildError(f"要接受的 drift 檔不存在：{accept_path}")
    accepted = json.loads(accept_path.read_text(encoding="utf-8"))
    for key in ("deleted", "edited"):
        if key not in accepted:
            raise TestsetBuildError(f"{accept_path} 缺少欄位 {key!r}")
    return accepted


def _unaccepted(drift: Mapping[str, object], accepted: Mapping[str, object]) -> dict:
    """尚未被接受的那一部分。接受一份 drift 不等於接受之後出現的新 drift。"""
    accepted_deleted = set(accepted["deleted"])  # type: ignore[arg-type]
    accepted_edited = {entry["id"]: entry["actual_sha256"] for entry in accepted["edited"]}  # type: ignore[union-attr]
    return {
        "deleted": [item for item in drift["deleted"] if item not in accepted_deleted],  # type: ignore[union-attr]
        "edited": [
            entry
            for entry in drift["edited"]  # type: ignore[union-attr]
            if entry["id"] not in accepted_edited
            or accepted_edited[entry["id"]] != entry["actual_sha256"]
        ],
    }


def rebuild(
    testset_dir: Path,
    out_dir: Path,
    pool_dir: Path,
    conversation_dir: Path,
    *,
    reuse_pools: bool,
    accept_drift: Path | None,
) -> int:
    """由識別字清單重建內容。有未被接受的 drift 即回傳非零結束碼。"""
    rows = _load_ids(testset_dir)
    found = _fetch_by_id(pool_dir, reuse=reuse_pools)
    drift = detect_drift(rows, found)
    accepted = _accepted(accept_drift)
    outstanding = _unaccepted(drift, accepted)

    drift_path = out_dir / DRIFT_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    _require_git_ignored(drift_path)
    drift_path.write_text(
        json.dumps(
            {**drift, "rebuilt_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if outstanding["deleted"] or outstanding["edited"]:
        for article_id in outstanding["deleted"]:
            print(f"已撤銷：{article_id}", file=sys.stderr)
        for entry in outstanding["edited"]:
            print(
                f"已編輯：{entry['id']}，清單雜湊 {entry['expected_sha256'][:12]}、"
                f"實際雜湊 {entry['actual_sha256'][:12]}，實際內文前 {SNIPPET_LENGTH} 字元："
                f"{entry['actual_prefix']!r}",
                file=sys.stderr,
            )
        print(
            f"重建中止：{len(outstanding['deleted'])} 則已撤銷、"
            f"{len(outstanding['edited'])} 則已編輯。差異已寫入 {drift_path}。"
            f"檢視之後以 `--accept-drift {drift_path}` 明確接受它，"
            f"接受的差異會寫入 manifest 與識別字清單。**不跳過壞的繼續**。",
            file=sys.stderr,
        )
        return 1

    accepted_ids = set(accepted["deleted"]) | {entry["id"] for entry in accepted["edited"]}
    kept = [row for row in rows if row["id"] not in set(accepted["deleted"])]
    chosen: dict[str, list[dict]] = {name: [] for name in SUBSETS}
    for row in kept:
        chosen[str(row["subset"])].append(found[str(row["id"])])
    written = write_subsets(out_dir, chosen)

    if accepted_ids:
        write_jsonl(
            testset_dir / IDS_FILENAME,
            [
                {
                    "id": row["id"],
                    "subset": row["subset"],
                    "text_sha256": text_sha256(found[str(row["id"])]["text"]),
                }
                for row in kept
            ],
        )
    manifest_path = testset_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rebuilt_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["subsets"] = written
    manifest["self_built"] = _self_subset_status(out_dir)
    manifest["conversation_ham"] = _conversation_status(out_dir, conversation_dir)
    manifest["accepted_drift"] = [
        {"id": article_id, "kind": "deleted"} for article_id in accepted["deleted"]
    ] + [{"id": entry["id"], "kind": "edited"} for entry in accepted["edited"]]
    manifest["drift_share"] = len(accepted_ids) / len(rows) if rows else 0.0
    manifest["drift_comparable"] = manifest["drift_share"] <= DRIFT_COMPARABLE_SHARE
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"重建完成：{sum(len(records) for records in chosen.values())} 則", file=sys.stderr)
    return 0


def run_select(
    testset_dir: Path,
    out_dir: Path,
    pool_dir: Path,
    dev_sample: Path,
    conversation_dir: Path,
    *,
    reuse_pools: bool,
) -> int:
    _require_not_git_ignored(testset_dir / IDS_FILENAME)
    chosen, stats = select(pool_dir, dev_sample, reuse_pools=reuse_pools)
    excluded = _dev_ids(dev_sample)
    intersection = sorted(
        {record["id"] for records in chosen.values() for record in records} & excluded
    )
    if intersection:
        raise TestsetBuildError(
            f"測試集與開發樣本的識別字交集不為空，共 {len(intersection)} 筆："
            f"{intersection[:5]}。測試集 MUST 與開發樣本不相交。"
        )
    written = write_subsets(out_dir, chosen)
    count = write_ids(testset_dir, chosen)
    manifest = build_manifest(
        written, stats, dev_sample, out_dir, len(intersection), conversation_dir
    )
    (testset_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, info in written.items():
        print(f"{name}：{info['count']} 則、切分 {info['splits']}", file=sys.stderr)  # type: ignore[index]
    print(f"識別字清單 {count} 行寫入 {testset_dir / IDS_FILENAME}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.eval.build_testset",
        description="建立與重建測試集。內容不進版控，識別字與雜湊進版控。",
    )
    parser.add_argument("mode", choices=("select", "rebuild"))
    parser.add_argument("--testset-dir", default=str(DEFAULT_TESTSET_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    parser.add_argument("--dev-sample", default=str(DEFAULT_DEV_SAMPLE))
    parser.add_argument("--conversation-dir", default=str(DEFAULT_CONVERSATION_DIR))
    parser.add_argument(
        "--reuse-pools",
        action="store_true",
        help="池檔已存在時不重抓。人明確指定，不自動判斷。",
    )
    parser.add_argument("--accept-drift", default=None, help="明確接受一份已檢視過的 drift")
    args = parser.parse_args(argv)

    testset_dir = Path(args.testset_dir)
    testset_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "select":
            return run_select(
                testset_dir,
                Path(args.out_dir),
                Path(args.pool_dir),
                Path(args.dev_sample),
                Path(args.conversation_dir),
                reuse_pools=args.reuse_pools,
            )
        return rebuild(
            testset_dir,
            Path(args.out_dir),
            Path(args.pool_dir),
            Path(args.conversation_dir),
            reuse_pools=args.reuse_pools,
            accept_drift=None if args.accept_drift is None else Path(args.accept_drift),
        )
    except (TestsetBuildError, CofactsFetchError) as error:
        print(f"失敗：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
