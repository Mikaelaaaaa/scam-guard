"""測試集的建立、切分、重現機制與界線。

真實內容不進版控（`data/`），因此吃資料的測試在缺檔時 skip，skip 訊息帶重建
命令 —— 否則它會變成一個永遠綠燈、永遠沒跑過的測試。不吃資料的測試
（切分的性質、清單的形狀、解析器、drift 判定）一律無條件執行。
"""

import ast
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from scam_guard.rules.speech_act import SELF_CONTAINED_CODE
from scam_guard.weights import BASIS_MEASURED, load_weights
from tools.eval.build_testset import (
    MULTI_MESSAGE_SUFFICIENT_MINIMUM,
    detect_drift,
)
from tools.eval.dataset import (
    CODE_BEARING_FLOOR,
    IDS_FIELDS,
    IDS_FILENAME,
    MANIFEST_FILENAME,
    PROVENANCE_VALUES,
    SELF_NOTICE_FIELDS,
    SELF_NOTICE_MINIMUM,
    SELF_NOTICE_TARGET,
    Sample,
    load_testset,
    read_jsonl,
    split_of,
    text_sha256,
)
from tools.eval.line_export import LineExportError, parse
from tools.eval.selectors import (
    COFACTS_SUBSETS,
    HAM_SUBSETS,
    HOLDOUT,
    LABEL_CASE_ONLY,
    SELF_SMS_HAM,
    SELF_SMS_SCAM,
    SUBSETS,
    TUNE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTSET_DIR = REPO_ROOT / "testset"
OUT_DIR = REPO_ROOT / "data" / "testset"

HOW_TO_BUILD = f"""找不到重建後的測試集 {OUT_DIR}。重建方式：

  python -m tools.eval.build_testset rebuild

內容不進版控（真實民眾訊息，Cofacts 開放資料為 CC BY-SA 4.0），缺檔是正常的。
版控的部分是 {TESTSET_DIR}/{IDS_FILENAME} 與 {MANIFEST_FILENAME}。"""


def _require_data() -> None:
    if not OUT_DIR.is_dir():
        pytest.skip(HOW_TO_BUILD)


# --- 切分：確定性、可加樣本、與內容無關 -----------------------------------


def test_split_is_deterministic_and_two_valued() -> None:
    for article_id in ("abc", "p2ym1pvt03iz", "37gk89sqhimce", ""):
        assert split_of(article_id) == split_of(article_id)
        assert split_of(article_id) in (TUNE, HOLDOUT)


def test_adding_samples_does_not_move_existing_ones() -> None:
    """加入新樣本後，既有樣本的歸屬全部不變 —— 這是不用 `random.shuffle` 的理由。"""
    original = [f"id-{index}" for index in range(200)]
    before = {article_id: split_of(article_id) for article_id in original}
    extended = original + [f"new-{index}" for index in range(400)]
    after = {article_id: split_of(article_id) for article_id in extended}
    assert all(after[article_id] == before[article_id] for article_id in original)


def test_split_is_roughly_balanced_and_independent_of_length() -> None:
    """切分與內容無關，因此與長度無關。門檻寫在測試裡：兩側各不少於四成。"""
    ids = [f"article-{index}" for index in range(2000)]
    tune = sum(1 for article_id in ids if split_of(article_id) == TUNE)
    assert 0.40 < tune / len(ids) < 0.60
    long_ids = [f"article-{index}-{'x' * index}" for index in range(2000)]
    long_tune = sum(1 for article_id in long_ids if split_of(article_id) == TUNE)
    assert 0.40 < long_tune / len(long_ids) < 0.60


def test_self_built_notices_are_all_holdout() -> None:
    """自建通知子集不切分 —— 用真實通知調過門檻的系統，在真實通知上的誤判率
    不再是一個獨立的量測。"""
    for index in range(50):
        sample = Sample(id=f"self-{index}", text="您的驗證碼為 482913", subset=SELF_SMS_HAM.name)
        assert sample.split == HOLDOUT


def test_case_only_subset_never_counts_in_rates() -> None:
    assert SELF_SMS_SCAM.label == LABEL_CASE_ONLY
    sample = Sample(id="x", text="y", subset=SELF_SMS_SCAM.name)
    assert sample.counts_in_rates is False
    for name in HAM_SUBSETS:
        assert Sample(id="x", text="y", subset=name).counts_in_rates is True


# --- 樣本量的推導 ---------------------------------------------------------


def test_self_notice_target_is_derived_not_chosen() -> None:
    """189 由 `z² / (n + z²) ≤ 0.02` 推得，不是一個選定的整數。"""
    z_squared = 1.959963984540054**2
    assert z_squared / (SELF_NOTICE_TARGET + z_squared) <= 0.02
    assert z_squared / (SELF_NOTICE_TARGET - 1 + z_squared) > 0.02
    assert SELF_NOTICE_MINIMUM < SELF_NOTICE_TARGET


# --- 版控的識別字清單：不含標籤、不含內文 ---------------------------------


def test_ids_file_has_exactly_three_fields_and_no_label() -> None:
    rows = read_jsonl(TESTSET_DIR / IDS_FILENAME)
    assert rows
    for row in rows:
        assert set(row) == set(IDS_FIELDS)
        assert "label" not in row


def test_ids_file_contains_no_message_text() -> None:
    """每一行的值皆為 ASCII 或已知的子集名稱。Cofacts 內文必為中文，會被抓到。"""
    rows = read_jsonl(TESTSET_DIR / IDS_FILENAME)
    for row in rows:
        assert row["subset"] in SUBSETS
        assert row["id"].isascii()
        assert row["text_sha256"].isascii()
        assert len(row["text_sha256"]) == 64


def test_versioned_testset_directory_is_not_git_ignored() -> None:
    """`.gitignore` 的 `data/` 無前導斜線、任何深度生效，`*.jsonl` 亦然。
    版控目錄被吃掉的話整個重現機制就落空。"""
    for path in (TESTSET_DIR / IDS_FILENAME, TESTSET_DIR / MANIFEST_FILENAME):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, f"{path} 被 git 忽略"


def test_produced_content_is_git_ignored() -> None:
    _require_data()
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{path} 未被 git 忽略"


def test_no_split_file_is_produced() -> None:
    """歸屬由識別字現算，不存檔 —— 不存就不會有「split 檔與資料檔不同步」。"""
    _require_data()
    names = [path.name for path in OUT_DIR.iterdir()]
    assert not [name for name in names if "split" in name or "tune" in name or "holdout" in name]


# --- 不得含類型欄位 -------------------------------------------------------


def test_no_scam_type_column_anywhere_in_the_testset() -> None:
    """連一個恆為空的佔位欄位都不留 —— 留了遲早有人拿 `.get()` 去讀它。"""
    _require_data()
    forbidden = {"scam_type", "type", "scam_types", "case_title", "category"}
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        for row in read_jsonl(path):
            assert not forbidden & set(row), f"{path} 含類型欄位"
    for row in read_jsonl(TESTSET_DIR / IDS_FILENAME):
        assert not forbidden & set(row)


# --- 沒有任何接受人工標籤的介面 -------------------------------------------


def test_no_code_path_writes_a_label_field() -> None:
    """標籤完全由子集決定 —— `tools/eval/` 中沒有任何一處產出帶 `label` 的紀錄。

    以 AST 判定而非字串比對：`if "label" in row: raise` 是一條**擋下**標籤的
    檢查，字串比對分不出它與一行寫入。
    """
    for path in sorted((REPO_ROOT / "tools" / "eval").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
                assert "label" not in keys, f"{path} 有一個帶 label 鍵的字典字面值"
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                assert node.slice.value != "label" or not isinstance(node.ctx, ast.Store), (
                    f"{path} 有一處寫入 label 欄位"
                )


# --- 與開發樣本不相交 -----------------------------------------------------


def test_testset_ids_are_disjoint_from_the_dev_sample() -> None:
    manifest = json.loads((TESTSET_DIR / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["dev_sample"]["id_intersection"] == 0
    assert manifest["dev_sample"]["file_sha256"]
    dev_sample = REPO_ROOT / manifest["dev_sample"]["path"]
    if not dev_sample.is_file():
        pytest.skip(f"找不到開發樣本 {dev_sample}，無法重新驗證交集")
    dev_ids = {record["id"] for record in read_jsonl(dev_sample)}
    testset_ids = {row["id"] for row in read_jsonl(TESTSET_DIR / IDS_FILENAME)}
    assert not dev_ids & testset_ids


# --- manifest 的必要欄位 --------------------------------------------------


def test_manifest_records_selectors_counts_and_the_things_not_done() -> None:
    manifest = json.loads((TESTSET_DIR / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for subset in COFACTS_SUBSETS:
        assert subset.name in manifest["selectors"]
        assert manifest["selectors"][subset.name]["articleTypes"] == ["TEXT"]
    assert set(manifest["not_done"]) == {
        "synth_logistics",
        "type_stratification",
        "balanced_sampling",
    }
    assert manifest["multi_message"]["minimum"] == MULTI_MESSAGE_SUFFICIENT_MINIMUM
    for name in (SELF_SMS_HAM.name, SELF_SMS_SCAM.name):
        assert name in manifest["self_built"]


def test_suspected_pool_is_taken_whole_without_sampling() -> None:
    """`cofacts_ham_suspected` 全取不取樣：選取數 = 池大小 - 開發樣本重疊。"""
    manifest = json.loads((TESTSET_DIR / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    entry = manifest["selection"]["cofacts_ham_suspected"]
    assert entry["target"] is None
    assert entry["selected"] == entry["pool_size"] - entry["removed_dev_overlap"]


# --- 重建與 drift ---------------------------------------------------------


def test_edited_article_is_reported_as_drift() -> None:
    rows = [{"id": "a", "subset": "cofacts_scam", "text_sha256": text_sha256("原文")}]
    drift = detect_drift(rows, {"a": {"text": "被編輯過的內文"}})
    assert drift["deleted"] == []
    assert drift["edited"][0]["id"] == "a"
    assert drift["edited"][0]["actual_prefix"] == "被編輯過的內文"


def test_deleted_article_is_reported_as_drift() -> None:
    rows = [{"id": "gone", "subset": "cofacts_scam", "text_sha256": text_sha256("原文")}]
    drift = detect_drift(rows, {})
    assert drift["deleted"] == ["gone"]
    assert drift["edited"] == []


def test_matching_hashes_produce_no_drift() -> None:
    rows = [{"id": "a", "subset": "cofacts_scam", "text_sha256": text_sha256("原文")}]
    assert detect_drift(rows, {"a": {"text": "原文"}}) == {"deleted": [], "edited": []}


def test_rebuild_reports_nonzero_exit_and_names_every_drifting_id(tmp_path: Path) -> None:
    """有 drift 時**非零結束碼**、印出全部 id、**不跳過繼續**。"""
    from tools.eval import build_testset

    testset_dir = tmp_path / "testset"
    testset_dir.mkdir()
    (testset_dir / IDS_FILENAME).write_text(
        json.dumps({"id": "a", "subset": "cofacts_scam", "text_sha256": text_sha256("原文")})
        + "\n",
        encoding="utf-8",
    )
    (testset_dir / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")

    calls: list[str] = []

    def fake_fetch_by_id(pool_dir: Path, *, reuse: bool) -> dict[str, dict]:
        calls.append(str(pool_dir))
        return {"a": {"id": "a", "text": "改過了", "source_uri": "u"}}

    original = build_testset._fetch_by_id
    original_ignored = build_testset._require_git_ignored
    build_testset._fetch_by_id = fake_fetch_by_id
    build_testset._require_git_ignored = lambda path: None
    try:
        code = build_testset.rebuild(
            testset_dir,
            tmp_path / "out",
            tmp_path / "pools",
            tmp_path / "conversation",
            reuse_pools=False,
            accept_drift=None,
        )
        assert code == 1
        drift = json.loads((tmp_path / "out" / "drift.json").read_text(encoding="utf-8"))
        assert drift["edited"][0]["id"] == "a"
        assert not (tmp_path / "out" / "cofacts_scam.jsonl").exists()

        accepted = tmp_path / "accepted.json"
        accepted.write_text(json.dumps(drift), encoding="utf-8")
        code = build_testset.rebuild(
            testset_dir,
            tmp_path / "out",
            tmp_path / "pools",
            tmp_path / "conversation",
            reuse_pools=False,
            accept_drift=accepted,
        )
        assert code == 0
        manifest = json.loads((testset_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["accepted_drift"] == [{"id": "a", "kind": "edited"}]
        assert (tmp_path / "out" / "cofacts_scam.jsonl").exists()
    finally:
        build_testset._fetch_by_id = original
        build_testset._require_git_ignored = original_ignored
    assert calls


# --- LINE 匯出檔解析 ------------------------------------------------------

EXPORT = (
    "[LINE] 與金融正義處理陳先生的聊天記錄\n"
    "儲存日期： 2024/09/10 11:35\n"
    "\n"
    "2024/08/19（一）\n"
    "上午09:53\t小麥\t您好\n"
    "下午01:54\t金融正義處理陳先生\t我這邊有個穩賺的標的\n"
)


def test_line_export_parses_three_columns_into_messages() -> None:
    messages = parse(EXPORT)
    assert [message.sender for message in messages] == ["小麥", "金融正義處理陳先生"]
    assert messages[0].sent_at == datetime(2024, 8, 19, 9, 53)
    assert messages[1].sent_at == datetime(2024, 8, 19, 13, 54)
    assert messages[1].text == "我這邊有個穩賺的標的"


def test_line_export_rejects_a_continuation_line_with_its_line_number() -> None:
    broken = EXPORT + "這是一行跨行續行\n"
    with pytest.raises(LineExportError, match="第 7 行"):
        parse(broken)


def test_line_export_rejects_a_nickname_containing_a_tab() -> None:
    broken = EXPORT + "上午10:00\t暱稱\t含tab\t內容\n"
    with pytest.raises(LineExportError, match="第 7 行"):
        parse(broken)


def test_line_export_requires_the_header() -> None:
    with pytest.raises(LineExportError, match="第 1 行"):
        parse("上午09:53\t小麥\t您好\n")


# --- 自建子集的欄位與去識別規則 -------------------------------------------


def test_self_notice_schema_records_consent_but_never_the_provider() -> None:
    assert "consent" in SELF_NOTICE_FIELDS
    assert PROVENANCE_VALUES == ("self", "granted", "public")
    for forbidden in ("provider", "phone", "name", "contact"):
        assert forbidden not in SELF_NOTICE_FIELDS


def test_self_notice_subset_keeps_its_digits(  # noqa: E501
) -> None:
    """含 4 至 8 位獨立數字的樣本比例 MUST 高於 50%；低於即代表去識別遮到了數字。

    判定用的是規則層**自帶碼豁免的同一個樣式**，不是另寫一個 —— 這個檢查要問的
    正是「豁免會不會生效」。
    """
    path = OUT_DIR / f"{SELF_SMS_HAM.name}.jsonl"
    if not path.is_file():
        pytest.skip(
            f"{SELF_SMS_HAM.name} 未蒐集（目標 {SELF_NOTICE_TARGET} 則、"
            f"最低 {SELF_NOTICE_MINIMUM} 則）。此子集需要蒐集者本人的手機，"
            f"不可由程式產生；缺席已記錄於 manifest 的 self_built。"
        )
    rows = read_jsonl(path)
    assert rows
    for row in rows:
        assert set(SELF_NOTICE_FIELDS) <= set(row)
        assert row["provenance"] in PROVENANCE_VALUES
    bearing = sum(1 for row in rows if SELF_CONTAINED_CODE.search(row["text"]) is not None)
    assert bearing / len(rows) > CODE_BEARING_FLOOR


def test_deidentification_never_calls_the_detection_core_redactor() -> None:
    """去識別以人工執行 —— `tools/eval/` 不 import 也不呼叫核心的遮蔽元件。

    同樣以 AST 判定：模組 docstring 裡**寫著**「MUST NOT 呼叫 redact_document()」
    是規則的陳述，不是一次呼叫。
    """
    forbidden_modules = {"scam_guard.pii", "scam_guard.redact"}
    for path in sorted((REPO_ROOT / "tools" / "eval").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_modules, f"{path} import 了 {node.module}"
            if isinstance(node, ast.Import):
                assert not forbidden_modules & {alias.name for alias in node.names}, path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "redact_document", f"{path} 呼叫了 redact_document()"


# --- 權重表的 measured_on 必須指向 tune -----------------------------------


def test_measured_weights_point_at_the_tuning_split_only() -> None:
    """載入層不知道資料集存在，這條只能由評估側驗。"""
    table = load_weights()
    for signal in table.signals.values():
        for weight in (signal.weight_soft, signal.weight_hard):
            if weight is None or weight.basis != BASIS_MEASURED:
                continue
            assert weight.measured_on is not None
            assert TUNE in weight.measured_on, signal.name
            assert HOLDOUT not in weight.measured_on, signal.name
    for key, threshold in table.thresholds.items():
        if threshold.basis != BASIS_MEASURED:
            continue
        assert threshold.measured_on is not None
        assert TUNE in threshold.measured_on, key
        assert HOLDOUT not in threshold.measured_on, key


# --- 實際載入一次 ---------------------------------------------------------


def test_loading_the_rebuilt_testset_yields_labelled_samples() -> None:
    _require_data()
    testset = load_testset(OUT_DIR, TESTSET_DIR / MANIFEST_FILENAME)
    assert testset.subsets
    for name, samples in testset.subsets.items():
        assert all(sample.subset == name for sample in samples)
        assert all(sample.label == SUBSETS[name].label for sample in samples)
