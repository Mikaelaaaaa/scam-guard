"""`tools/fetch_tranco.py` 的解析、PSL 過濾與失敗行為。**不打真實網路。**

全部以構造的 CSV 與 zip 測試，測試中不對 `tranco-list.eu` 發出任何請求 ——
在無網路的環境執行 `pytest` 必須全數通過。
"""

import io
import json
import zipfile
from pathlib import Path

import pytest

from scam_guard.allowlist import ENTRIES_FILENAME, MANIFEST_FILENAME, RankAllowlist
from scam_guard.url import PublicSuffixList
from tests.test_blocklist_store import PSL_WITH_EXAMPLE
from tools.fetch_tranco import (
    CSV_MEMBER,
    build,
    extract_csv,
    filter_rows,
    last_modified_to_iso,
    parse_rows,
    write_snapshot,
)

LAST_MODIFIED = "Sun, 13 Sep 2026 22:25:00 GMT"

CSV = "1,google.com\n2,evil.com\n3,www.example.com\n4,wixsite.com\n5,phish.wixsite.com\n"


def psl_of() -> PublicSuffixList:
    return PublicSuffixList.parse(PSL_WITH_EXAMPLE)


def zip_of(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return buffer.getvalue()


# --- 解析 -----------------------------------------------------------------


def test_zip_member_is_extracted() -> None:
    assert extract_csv(zip_of({CSV_MEMBER: CSV})) == CSV


def test_missing_zip_member_lists_actual_names() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        extract_csv(zip_of({"something-else.csv": CSV}))
    message = str(excinfo.value)
    assert CSV_MEMBER in message
    assert "something-else.csv" in message


def test_rows_carry_rank_and_domain() -> None:
    rows = parse_rows(CSV)
    assert [(row.rank, row.domain) for row in rows][:2] == [(1, "google.com"), (2, "evil.com")]


def test_malformed_line_reports_line_number_and_text() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_rows("1,google.com\ngoogle.com\n")
    message = str(excinfo.value)
    assert "第 2 行" in message
    assert "google.com" in message


def test_non_contiguous_rank_reports_both_values() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_rows("1,google.com\n3,evil.com\n")
    message = str(excinfo.value)
    assert "第 2 行" in message
    assert "3" in message and "1" in message


def test_rank_not_starting_at_one_raises() -> None:
    with pytest.raises(ValueError, match="不由 1 起"):
        parse_rows("2,google.com\n3,evil.com\n")


def test_truncated_file_reports_both_counts() -> None:
    rows = parse_rows(CSV)
    with pytest.raises(ValueError) as excinfo:
        build(rows, psl_of(), 1000)
    message = str(excinfo.value)
    assert "5" in message and "1000" in message


def test_last_modified_becomes_iso() -> None:
    assert last_modified_to_iso(LAST_MODIFIED).startswith("2026-09-13T22:25:00")


def test_invalid_last_modified_reports_the_raw_value() -> None:
    with pytest.raises(ValueError, match="昨天"):
        last_modified_to_iso("昨天")


# --- PSL 過濾 -------------------------------------------------------------


def test_public_suffix_entries_are_dropped_and_counted() -> None:
    """`wixsite.com` 本身是 PRIVATE 後綴，不是一個網站，留著保護的是不存在的東西。

    `phish.wixsite.com` 則相反：它的可註冊網域就是它自己，是一個真的可以被
    查到的鍵，所以留下。這正是 PSL PRIVATE 區段的用意。
    """
    filtered = filter_rows(parse_rows(CSV), psl_of(), 5)
    assert [item.domain for item in filtered.kept] == [
        "google.com",
        "evil.com",
        "example.com",
        "phish.wixsite.com",
    ]
    assert filtered.dropped_public_suffix_count == 1


def test_www_prefix_is_normalized_before_the_psl_check() -> None:
    """順序不可顛倒：`www.gov.tw` 正規化後是 `gov.tw`，而那是一個 public suffix。

    先過濾再正規化會讓這一筆留在白名單裡。
    """
    filtered = filter_rows(parse_rows("1,www.gov.tw\n2,evil.com\n"), psl_of(), 2)
    assert [item.domain for item in filtered.kept] == ["evil.com"]
    assert filtered.dropped_public_suffix_count == 1


def test_normalization_collision_keeps_the_better_rank() -> None:
    filtered = filter_rows(parse_rows("1,example.com\n2,www.example.com\n"), psl_of(), 2)
    assert [(item.domain, item.rank) for item in filtered.kept] == [("example.com", 1)]
    assert filtered.dropped_duplicate_count == 1


def test_threshold_truncates_before_filtering() -> None:
    filtered = filter_rows(parse_rows(CSV), psl_of(), 2)
    assert [item.rank for item in filtered.kept] == [1, 2]


# --- 寫出 -----------------------------------------------------------------


def snapshot(tmp_path: Path, top: int = 5, **kwargs: object) -> dict[str, object]:
    rows = parse_rows(CSV)
    return write_snapshot(
        tmp_path,
        "GQNVK",
        top,
        LAST_MODIFIED,
        len(rows),
        filter_rows(rows, psl_of(), top),
        allow_shrink=bool(kwargs.get("allow_shrink", False)),
    )


def test_manifest_records_the_list_id_and_permalink(tmp_path: Path) -> None:
    manifest = snapshot(tmp_path)
    assert manifest["list_id"] == "GQNVK"
    assert manifest["permalink"].endswith("GQNVK")
    assert manifest["threshold"] == 5


def test_manifest_marks_the_weaker_freshness_source(tmp_path: Path) -> None:
    """`data_through` 取自 HTTP 標頭，是一個比黑名單弱的判準，必須標記。"""
    assert snapshot(tmp_path)["data_through_source"] == "http_last_modified"


def test_manifest_records_the_filter_statistics(tmp_path: Path) -> None:
    """`dropped_public_suffix_count` 是 PSL 過濾有沒有在跑的唯一可觀測訊號。"""
    manifest = snapshot(tmp_path)
    assert manifest["raw_row_count"] == 5
    assert manifest["kept_count"] == 4
    assert manifest["dropped_public_suffix_count"] == 1


def test_entries_carry_the_rank(tmp_path: Path) -> None:
    snapshot(tmp_path)
    records = [
        json.loads(line)
        for line in (tmp_path / ENTRIES_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert records[0] == {"domain": "google.com", "rank": 1}


def test_written_snapshot_is_loadable(tmp_path: Path) -> None:
    snapshot(tmp_path)
    allowlist = RankAllowlist.load(tmp_path, max_age_days=100_000)
    assert allowlist.rank("google.com") == 1


def test_shrink_is_refused_without_the_flag(tmp_path: Path) -> None:
    snapshot(tmp_path)
    rows = parse_rows("1,google.com\n")
    with pytest.raises(RuntimeError, match="拒絕覆寫"):
        write_snapshot(
            tmp_path,
            "GQNVK",
            1,
            LAST_MODIFIED,
            1,
            filter_rows(rows, psl_of(), 1),
            allow_shrink=False,
        )
    write_snapshot(
        tmp_path, "GQNVK", 1, LAST_MODIFIED, 1, filter_rows(rows, psl_of(), 1), allow_shrink=True
    )
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["kept_count"] == 1


def test_failed_build_leaves_the_existing_snapshot_untouched(tmp_path: Path) -> None:
    snapshot(tmp_path)
    before_entries = (tmp_path / ENTRIES_FILENAME).read_bytes()
    before_manifest = (tmp_path / MANIFEST_FILENAME).read_bytes()
    with pytest.raises(ValueError):
        build(parse_rows(CSV), psl_of(), 1000)
    assert (tmp_path / ENTRIES_FILENAME).read_bytes() == before_entries
    assert (tmp_path / MANIFEST_FILENAME).read_bytes() == before_manifest
    assert not list(tmp_path.glob("*.tmp"))
