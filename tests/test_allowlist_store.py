"""`RankAllowlist` 的載入檢查、新鮮度判定與查詢。以**構造的快照**測試。

真實快照不進版控，而且 Tranco 每天換一份 —— 若測試依賴真實資料，
同一份測試在兩天之內會給出兩個答案。
"""

import json
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from scam_guard.allowlist import ENTRIES_FILENAME, MANIFEST_FILENAME, RankAllowlist
from scam_guard.url import utc_today

DEFAULT_ENTRIES = (
    {"domain": "google.com", "rank": 1},
    {"domain": "weebly.com", "rank": 365},
    {"domain": "example.com", "rank": 900},
)


def days_ago(days: int) -> str:
    return (utc_today() - timedelta(days=days)).isoformat()


def write_snapshot(
    directory: Path,
    entries: tuple[dict[str, object], ...] = DEFAULT_ENTRIES,
    **manifest_overrides: object,
) -> Path:
    """寫出一份可載入的構造快照，供各測試改造。"""
    directory.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n").encode(
        "utf-8"
    )
    (directory / ENTRIES_FILENAME).write_bytes(payload)
    manifest: dict[str, object] = {
        "list_id": "GQNVK",
        "permalink": "https://tranco-list.eu/list/GQNVK",
        "source_url": "https://tranco-list.eu/download_daily/GQNVK",
        "threshold": 1000,
        "fetched_at": f"{utc_today().isoformat()}T00:00:00+00:00",
        "data_through": f"{days_ago(1)}T22:25:00+00:00",
        "data_through_source": "http_last_modified",
        "raw_row_count": 1_000_000,
        "kept_count": len(entries),
        "dropped_public_suffix_count": 42,
        "entries_file": ENTRIES_FILENAME,
        "entries_sha256": sha256(payload).hexdigest(),
        "attribution": "Tranco (https://tranco-list.eu/)",
    }
    manifest.update(manifest_overrides)
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def allowlist_of(tmp_path: Path, **kwargs: object) -> RankAllowlist:
    return RankAllowlist.load(write_snapshot(tmp_path, **kwargs), max_age_days=30)


# --- 載入是顯式動作 -------------------------------------------------------


def test_import_does_not_read_files() -> None:
    """沒跑過取得程式的環境也要跑得起 pytest —— CI 正是這樣的環境。"""
    import importlib

    module = importlib.import_module("scam_guard.allowlist")
    assert module.ENTRIES_FILENAME == ENTRIES_FILENAME


def test_missing_snapshot_names_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tools.fetch_tranco"):
        RankAllowlist.load(tmp_path, max_age_days=30)


def test_core_module_imports_no_network_library() -> None:
    """查詢層不上網。界線由 `pyproject.toml` 的 banned-api 守著，這裡再驗一次。"""
    source = Path("scam_guard/allowlist.py").read_text(encoding="utf-8")
    for library in ("urllib", "requests", "httpx", "socket", "http.client"):
        assert f"import {library}" not in source


def test_core_module_holds_no_external_format_knowledge() -> None:
    """下載網址與門檻常數都屬 `tools/`。"""
    source = Path("scam_guard/allowlist.py").read_text(encoding="utf-8")
    assert "tranco-list.eu/download" not in source
    assert "top-1m" not in source


# --- 新鮮度 ---------------------------------------------------------------


def test_max_age_days_is_required(tmp_path: Path) -> None:
    write_snapshot(tmp_path)
    with pytest.raises(TypeError):
        RankAllowlist.load(tmp_path)  # type: ignore[call-arg]


def test_stale_snapshot_raises_with_numbers_and_command(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, data_through=f"{days_ago(45)}T22:25:00+00:00")
    with pytest.raises(ValueError) as excinfo:
        RankAllowlist.load(directory, max_age_days=30)
    message = str(excinfo.value)
    assert "45" in message
    assert "30" in message
    assert days_ago(45) in message
    assert "tools.fetch_tranco" in message


def test_fresh_snapshot_loads(tmp_path: Path) -> None:
    assert allowlist_of(tmp_path).entry_count == 3


# --- 一致性檢查 -----------------------------------------------------------


def test_sha256_mismatch_raises_and_prints_both(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, entries_sha256="0" * 64)
    with pytest.raises(ValueError) as excinfo:
        RankAllowlist.load(directory, max_age_days=30)
    assert "sha256" in str(excinfo.value)
    assert "0" * 64 in str(excinfo.value)


def test_missing_manifest_field_lists_actual_fields(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["list_id"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        RankAllowlist.load(directory, max_age_days=30)
    assert "list_id" in str(excinfo.value)
    assert "permalink" in str(excinfo.value)


def test_entry_missing_rank_reports_line_and_fields(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, ({"domain": "google.com"},))
    with pytest.raises(ValueError) as excinfo:
        RankAllowlist.load(directory, max_age_days=30)
    assert "rank" in str(excinfo.value)
    assert "第 1 行" in str(excinfo.value)
    assert "domain" in str(excinfo.value)


def test_non_integer_rank_raises(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, ({"domain": "google.com", "rank": "1"},))
    with pytest.raises(ValueError, match="rank"):
        RankAllowlist.load(directory, max_age_days=30)


def test_zero_entries_raises(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, ())
    with pytest.raises(ValueError, match="總筆數為 0"):
        RankAllowlist.load(directory, max_age_days=30)


def test_rank_beyond_threshold_raises(tmp_path: Path) -> None:
    """只載入門檻以內的項目，是這個模組對記憶體的唯一承諾。"""
    directory = write_snapshot(tmp_path, ({"domain": "google.com", "rank": 5000},))
    with pytest.raises(ValueError) as excinfo:
        RankAllowlist.load(directory, max_age_days=30)
    assert "5000" in str(excinfo.value)
    assert "1000" in str(excinfo.value)


# --- 查詢 -----------------------------------------------------------------


def test_hit_returns_the_rank_not_a_boolean(tmp_path: Path) -> None:
    rank = allowlist_of(tmp_path).rank("weebly.com")
    assert rank == 365
    assert not isinstance(rank, bool)


def test_miss_returns_none(tmp_path: Path) -> None:
    assert allowlist_of(tmp_path).rank("evil.com") is None


def test_manifest_is_exposed_for_downstream(tmp_path: Path) -> None:
    """下游要 `list_id` 與 `threshold` 寫進依據文案與消融紀錄。"""
    manifest = allowlist_of(tmp_path).manifest
    assert manifest["list_id"] == "GQNVK"
    assert manifest["threshold"] == 1000
