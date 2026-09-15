"""GitHub Pages 站台的黑名單載入：註冊、不註冊，以及兩者的分界。

**以構造的小型快照測試，不打真實網路、不依賴 `data/`。** CI 是一個沒跑過取得
程式的環境，而測試若讀真實快照，同一份測試在兩天之內會給出兩個答案。

**為什麼要 patch `PublicSuffixList.load` 才 import 得動 `docs/pages_app.py`。**
那一側在 import 階段就要 `/psl` 這個絕對路徑，而測試程序寫不到檔案系統根目錄。
patch 只涵蓋 import 那一刻，其餘每一條測試都用真的 PSL 與真的
`BlocklistStore.load()` / `RankAllowlist.load()`。

被驗的性質只有一個，但它有兩面：**快照可用時 `url_blocklist` 要在**，
**快照不可用時它要不在，而且不能變成一個零筆的名單** —— 「這個網域不在名單上」
與「這裡沒有名單」在 `Verdict.checks` 裡長得一模一樣，而前者是一次成功的推論、
後者是一次缺席。
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from scam_guard.url import PublicSuffixList
from tests.test_allowlist_store import write_snapshot as write_allowlist_snapshot
from tests.test_blocklist_store import days_ago, psl_of
from tests.test_blocklist_store import write_snapshot as write_blocklist_snapshot

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _fixed_psl(path: str | Path, *, max_age_days: int) -> PublicSuffixList:
    """`PublicSuffixList.load` 的替身，只在 import `pages_app` 的那一刻生效。

    參數收下但不使用：被取代的那個方法會去讀 `/psl`，而這裡要的就是不去讀它。
    """
    return psl_of()


@pytest.fixture(scope="module")
def pages_app() -> ModuleType:
    """import `docs/pages_app.py`。

    import 當下 `/blocklist` 與 `/allowlist` 都不存在，所以模組層級那一次載入
    走的是失敗路徑 —— 那正是「快照檔不存在」這個情境的實地驗證，
    由 `test_module_level_load_without_snapshots_does_not_register` 斷言。
    """
    sys.path.insert(0, str(DOCS))
    with patch.object(PublicSuffixList, "load", _fixed_psl):
        module = importlib.import_module("pages_app")
    yield module
    del sys.modules["pages_app"]
    sys.path.remove(str(DOCS))


def registered_names(pages_app: ModuleType, store: object, allowlist: object) -> list[str]:
    registry = pages_app.build_registry(psl_of(), store, allowlist)
    return [check.name for check in registry.enabled()]


def both_snapshots(tmp_path: Path, **blocklist_overrides: dict[str, object]) -> tuple[str, str]:
    """寫出一對可載入的構造快照，回傳兩個目錄路徑。"""
    blocklist = write_blocklist_snapshot(tmp_path / "blocklist", **blocklist_overrides)
    allowlist = write_allowlist_snapshot(tmp_path / "allowlist")
    return str(blocklist), str(allowlist)


# --- 兩份快照都可用 -------------------------------------------------------


def test_both_snapshots_valid_registers_the_check(pages_app: ModuleType, tmp_path: Path) -> None:
    blocklist_dir, allowlist_dir = both_snapshots(tmp_path)
    store, allowlist, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert reason == ""
    assert store is not None
    assert allowlist is not None
    assert "url_blocklist" in registered_names(pages_app, store, allowlist)


def test_unregistered_list_keeps_only_domain_age_when_loaded(pages_app: ModuleType) -> None:
    """成功時清單上只剩 `domain_age`。

    原本寫死在這裡的「名單有十萬筆，在瀏覽器裡載入太重」MUST 消失 ——
    那個理由實測是假的（gzip 後 1,268,967 B、Pyodide 內常駐約 28 MB），
    而一個寫著假理由的揭露比不揭露更糟。
    """
    names = [name for name, _, _ in pages_app.unregistered_checks("")]
    assert names == ["domain_age"]


def test_reason_always_comes_from_an_exception(pages_app: ModuleType, tmp_path: Path) -> None:
    """未註冊理由 MUST 來自當次載入拋出的例外，不是一段寫死的字串。

    寫死的那一句是「名單有十萬筆，在瀏覽器裡載入太重」，它已經被移除；
    這條測試守的是它不會以任何形式回來 —— 一個不隨情境改變的理由，
    就是一段寫死的字串。
    """
    blocklist_dir, allowlist_dir = both_snapshots(tmp_path)
    (Path(blocklist_dir) / "entries.jsonl").unlink()
    _, _, missing = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    stale_dir, fresh_allowlist = both_snapshots(
        tmp_path / "stale", **{"176455": {"data_through": days_ago(400)}}
    )
    _, _, expired = pages_app.load_snapshots(psl_of(), stale_dir, fresh_allowlist)
    assert "載入太重" not in missing
    assert "載入太重" not in expired
    assert missing != expired


# --- 黑名單不可用 ---------------------------------------------------------


def test_module_level_load_without_snapshots_does_not_register(pages_app: ModuleType) -> None:
    """import 當下 `/blocklist` 不存在 —— 這是線上快照沒被部署上去的形狀。"""
    assert pages_app.STORE is None
    assert pages_app.ALLOWLIST is None
    assert "tools.fetch_blocklist" in pages_app.BLOCKLIST_UNREGISTERED_REASON
    assert "url_blocklist" not in [check.name for check in pages_app.REGISTRY.enabled()]
    assert "url_blocklist" in [name for name, _, _ in pages_app.UNREGISTERED_CHECKS]


def test_missing_entries_file_names_the_path(pages_app: ModuleType, tmp_path: Path) -> None:
    blocklist_dir, allowlist_dir = both_snapshots(tmp_path)
    (Path(blocklist_dir) / "entries.jsonl").unlink()
    store, allowlist, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert store is None
    assert allowlist is None
    assert str(Path(blocklist_dir) / "entries.jsonl") in reason
    assert "url_blocklist" not in registered_names(pages_app, store, allowlist)


def test_sha256_mismatch_reason_carries_both_digests(pages_app: ModuleType, tmp_path: Path) -> None:
    blocklist_dir, allowlist_dir = both_snapshots(tmp_path)
    (Path(blocklist_dir) / "entries.jsonl").write_text(
        '{"host": "x.com", "url": "x.com", "source": "176455", '
        '"first_seen": "2026-08", "last_seen": "2026-08"}\n',
        encoding="utf-8",
    )
    store, allowlist, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert store is None
    assert "sha256" in reason
    # 兩個雜湊值都要在訊息裡：只印一個的話看不出是哪一邊變了。
    assert reason.count("為 ") >= 2


def test_stale_blocklist_reason_names_the_dataset(pages_app: ModuleType, tmp_path: Path) -> None:
    blocklist_dir, allowlist_dir = both_snapshots(
        tmp_path, **{"176455": {"data_through": days_ago(400)}}
    )
    store, allowlist, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert store is None
    assert "176455" in reason
    assert days_ago(400) in reason
    assert "60" in reason
    assert "url_blocklist" not in registered_names(pages_app, store, allowlist)


def test_reason_is_not_rewritten_into_a_generic_phrase(
    pages_app: ModuleType, tmp_path: Path
) -> None:
    """過期與損毀共用同一條處置，但訊息**不得**被改寫成同一句話。

    `BlocklistStore.load()` 的四種 `ValueError` 各自已經指名了是哪一種，
    而以字串比對去分辨它們再改寫，等於在這裡重寫一次那四條規則。
    """
    blocklist_dir, allowlist_dir = both_snapshots(
        tmp_path, **{"165027": {"data_through": days_ago(300)}}
    )
    _, _, stale = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    (Path(blocklist_dir) / "entries.jsonl").write_bytes(b"")
    _, _, broken = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert stale != broken
    assert "過期" in stale
    assert "sha256" in broken


# --- 白名單不可用，黑名單正常 ---------------------------------------------


def test_stale_allowlist_takes_the_blocklist_with_it(pages_app: ModuleType, tmp_path: Path) -> None:
    """白名單是成對的那一半，而它的 30 天比黑名單的 60 天緊。

    沒有白名單的 `url_blocklist` 會重現 `add-tranco-allowlist` 實測到的那一筆
    硬證據偽陽性（160055 收錄 `play.google.com`），而白名單不是 `Check` ——
    畫面上沒有任何地方報告得了它的缺席。
    """
    blocklist_dir = str(write_blocklist_snapshot(tmp_path / "blocklist"))
    allowlist_dir = str(
        write_allowlist_snapshot(
            tmp_path / "allowlist", data_through=f"{days_ago(100)}T22:25:00+00:00"
        )
    )
    store, allowlist, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert store is None
    assert allowlist is None
    assert "Tranco" in reason
    assert "白名單過期" in reason
    assert "url_blocklist" not in registered_names(pages_app, store, allowlist)


def test_missing_allowlist_reason_points_at_the_allowlist(
    pages_app: ModuleType, tmp_path: Path
) -> None:
    blocklist_dir = str(write_blocklist_snapshot(tmp_path / "blocklist"))
    allowlist_dir = str(tmp_path / "allowlist")
    store, allowlist, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert store is None
    assert "tools.fetch_tranco" in reason
    assert "tools.fetch_blocklist" not in reason


# --- 失敗路徑 MUST NOT 產生一個零筆的名單 ---------------------------------


def test_no_failure_path_produces_an_empty_store(pages_app: ModuleType, tmp_path: Path) -> None:
    """六條失敗路徑，每一條都必須回傳 `None`，不是一個 `entry_count == 0` 的 store。"""
    missing_blocklist = str(tmp_path / "absent")
    missing_allowlist = str(tmp_path / "absent")

    stale_blocklist, fresh_allowlist = both_snapshots(
        tmp_path / "stale", **{"176455": {"data_through": days_ago(400)}}
    )
    corrupt_blocklist, _ = both_snapshots(tmp_path / "corrupt")
    (Path(corrupt_blocklist) / "entries.jsonl").write_bytes(b"")
    no_license_blocklist, _ = both_snapshots(
        tmp_path / "license", **{"165027": {"redistributable": False}}
    )
    good_blocklist, good_allowlist = both_snapshots(tmp_path / "good")
    stale_allowlist = str(
        write_allowlist_snapshot(
            tmp_path / "stale-allowlist", data_through=f"{days_ago(100)}T22:25:00+00:00"
        )
    )

    failures = (
        (missing_blocklist, fresh_allowlist),
        (stale_blocklist, fresh_allowlist),
        (corrupt_blocklist, fresh_allowlist),
        (no_license_blocklist, fresh_allowlist),
        (good_blocklist, missing_allowlist),
        (good_blocklist, stale_allowlist),
    )
    for blocklist_dir, allowlist_dir in failures:
        store, allowlist, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
        assert store is None, blocklist_dir
        assert allowlist is None, blocklist_dir
        assert reason != ""

    # 對照組：同一組寫法在快照有效時確實會註冊，否則上面六條全過也證明不了什麼。
    store, allowlist, reason = pages_app.load_snapshots(psl_of(), good_blocklist, good_allowlist)
    assert store is not None
    assert store.entry_count > 0
    assert reason == ""


# --- 門檻沿用既有部署的數字 -----------------------------------------------


def test_thresholds_match_the_existing_deployment(pages_app: ModuleType) -> None:
    """60 取自 `api/app.py`、30 取自 `RankAllowlist.load()` 的 docstring。

    160055 為 `retired`，`_check_freshness` 會跳過它 —— 對應表裡要是多了它，
    讀的人會以為它還會更新。
    """
    assert pages_app.BLOCKLIST_MAX_AGE_DAYS == {"176455": 60, "165027": 60}
    assert pages_app.ALLOWLIST_MAX_AGE_DAYS == 30


def test_redistributable_check_is_not_relaxed(pages_app: ModuleType) -> None:
    """本站台的產出就是給第三方看的判斷依據，`require_redistributable` 不得被放寬。"""
    source = (DOCS / "pages_app.py").read_text(encoding="utf-8")
    assert "require_redistributable=False" not in source


def test_unlicensed_source_is_refused(pages_app: ModuleType, tmp_path: Path) -> None:
    blocklist_dir, allowlist_dir = both_snapshots(
        tmp_path, **{"165027": {"redistributable": False}}
    )
    store, _, reason = pages_app.load_snapshots(psl_of(), blocklist_dir, allowlist_dir)
    assert store is None
    assert "165027" in reason
