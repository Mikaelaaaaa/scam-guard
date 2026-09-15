"""`tools/fetch_phishing_feed.py` 的旗標、解析與失敗行為。**不打真實網路。**

PhishTank 的註冊於 2026-09-14 實測為關閉，無金鑰下載回 403 —— 也就是說
這條路徑今天無法對真實資料驗證。這份測試以構造的 CSV 完整覆蓋解析與
失敗行為，但它證明不了「真的下載得到」，那要等註冊重開。
"""

import json
from pathlib import Path

import pytest

from scam_guard.blocklist import ENTRIES_FILENAME, MANIFEST_FILENAME
from tools.fetch_phishing_feed import (
    OPENPHISH,
    PHISHTANK,
    collect,
    main,
    parse_openphish,
    parse_phishtank,
    write_snapshot,
)

LAST_MODIFIED = "Sun, 14 Sep 2026 12:00:03 GMT"

CSV_HEADER = (
    "phish_id,url,phish_detail_url,submission_time,verified,verification_time,online,target\n"
)
CSV_PHISHTANK = CSV_HEADER + (
    "8888001,https://WWW.Example.COM/login,https://phishtank.org/x,"
    "2026-09-13T03:10:00+00:00,yes,2026-09-13T04:12:31+00:00,yes,PayPal\n"
    "8888002,http://evil.other.com/wp-content/x,https://phishtank.org/y,"
    "2026-09-13T05:00:00+00:00,yes,2026-09-13T06:00:00+00:00,yes,Netflix\n"
)

FEED_OPENPHISH = "https://wallet-restore.example/seed\nhttps://mysite.other.com/a\n"


# --- 旗標 -----------------------------------------------------------------


def test_no_source_does_nothing(tmp_path: Path, capsys) -> None:
    """預設不取得任何東西，而且這是刻意的 —— 兩個來源各有各的問題。"""
    assert main(["--out", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []
    printed = capsys.readouterr().out
    assert PHISHTANK in printed
    assert OPENPHISH in printed


def test_phishtank_without_a_key_raises_with_the_verification_result() -> None:
    """例外訊息本身就是查證結果，下一個人靠它省下一輪查證。"""
    with pytest.raises(ValueError) as excinfo:
        collect([PHISHTANK], None, False, 1.0)
    message = str(excinfo.value)
    assert "register.php" in message or "registration temporarily disabled" in message
    assert "403" in message
    assert "fallback" in message


def test_openphish_without_accepting_terms_quotes_both_clauses() -> None:
    with pytest.raises(ValueError) as excinfo:
        collect([OPENPHISH], None, False, 1.0)
    message = str(excinfo.value)
    assert "solely for your personal use" in message
    assert "distribute, display, disclose" in message


# --- PhishTank 的解析 -----------------------------------------------------


def test_host_is_normalized_and_url_is_kept_verbatim() -> None:
    parsed = parse_phishtank(CSV_PHISHTANK)
    assert parsed.entries[0].host == "example.com"
    assert parsed.entries[0].url == "https://WWW.Example.COM/login"


def test_target_is_kept_as_an_opaque_string() -> None:
    parsed = parse_phishtank(CSV_PHISHTANK)
    assert [entry.target for entry in parsed.entries] == ["PayPal", "Netflix"]


def test_verification_time_becomes_first_and_last_seen() -> None:
    (entry, _) = parse_phishtank(CSV_PHISHTANK).entries
    assert entry.first_seen == entry.last_seen == "2026-09-13T04:12:31+00:00"


def test_data_through_is_the_latest_verification_time() -> None:
    assert parse_phishtank(CSV_PHISHTANK).data_through == "2026-09-13T06:00:00+00:00"


def test_unverified_record_raises_with_the_phish_id_and_value() -> None:
    """不安靜過濾 —— 安靜過濾會讓我們永遠不知道前提已經不成立。"""
    broken = CSV_PHISHTANK.replace(
        "2026-09-13T03:10:00+00:00,yes,", "2026-09-13T03:10:00+00:00,no,"
    )
    with pytest.raises(ValueError) as excinfo:
        parse_phishtank(broken)
    message = str(excinfo.value)
    assert "8888001" in message
    assert "'no'" in message


def test_offline_record_raises() -> None:
    broken = CSV_PHISHTANK.replace(
        "2026-09-13T04:12:31+00:00,yes,PayPal", "2026-09-13T04:12:31+00:00,no,PayPal"
    )
    with pytest.raises(ValueError, match="8888001"):
        parse_phishtank(broken)


def test_renamed_field_reports_source_field_and_actual_header() -> None:
    renamed = CSV_PHISHTANK.replace("verification_time", "verified_at")
    with pytest.raises(ValueError) as excinfo:
        parse_phishtank(renamed)
    message = str(excinfo.value)
    assert PHISHTANK in message
    assert "verification_time" in message
    assert "verified_at" in message


def test_empty_phishtank_raises() -> None:
    with pytest.raises(ValueError, match="0 筆"):
        parse_phishtank(CSV_HEADER)


def test_url_without_a_host_reports_the_identifier() -> None:
    broken = CSV_PHISHTANK.replace("https://WWW.Example.COM/login", "https:///login")
    with pytest.raises(ValueError) as excinfo:
        parse_phishtank(broken)
    assert "8888001" in str(excinfo.value)


# --- OpenPhish 的解析 -----------------------------------------------------


def test_openphish_has_no_target_and_does_not_guess_one() -> None:
    """community 層沒有品牌資訊，而猜品牌是 `url_brand` 的工作。"""
    parsed = parse_openphish(FEED_OPENPHISH, LAST_MODIFIED)
    assert [entry.target for entry in parsed.entries] == [None, None]
    assert [entry.host for entry in parsed.entries] == [
        "wallet-restore.example",
        "mysite.other.com",
    ]


def test_openphish_data_through_comes_from_last_modified() -> None:
    parsed = parse_openphish(FEED_OPENPHISH, LAST_MODIFIED)
    assert parsed.data_through.startswith("2026-09-14T12:00:03")


def test_openphish_without_last_modified_raises() -> None:
    """該 feed 的內容沒有任何日期欄位，沒有這個標頭就沒有新鮮度判準。"""
    with pytest.raises(RuntimeError, match="Last-Modified"):
        parse_openphish(FEED_OPENPHISH, None)


def test_empty_openphish_raises() -> None:
    with pytest.raises(ValueError, match="0 筆"):
        parse_openphish("\n\n", LAST_MODIFIED)


# --- 寫出 -----------------------------------------------------------------


def test_feed_source_is_never_redistributable_nor_domain_matched(tmp_path: Path) -> None:
    manifest = write_snapshot(
        tmp_path,
        {OPENPHISH: parse_openphish(FEED_OPENPHISH, LAST_MODIFIED)},
        allow_shrink=False,
    )
    meta = manifest["sources"][OPENPHISH]
    assert meta["redistributable"] is False
    assert meta["domain_level_matching"] is False
    assert "third party" in meta["license_note"]


def test_phishtank_manifest_does_not_leak_the_app_key(tmp_path: Path) -> None:
    manifest = write_snapshot(
        tmp_path, {PHISHTANK: parse_phishtank(CSV_PHISHTANK)}, allow_shrink=False
    )
    meta = manifest["sources"][PHISHTANK]
    assert "{app_key}" in meta["source_url"]
    assert meta["domain_level_matching"] is False
    assert "Cisco" in meta["license_note"]


def test_snapshot_is_rebuilt_not_accumulated(tmp_path: Path) -> None:
    """PhishTank 已撤銷的項目只是從檔案消失，累加會讓撤銷永遠不生效。"""
    write_snapshot(tmp_path, {PHISHTANK: parse_phishtank(CSV_PHISHTANK)}, allow_shrink=False)
    shrunk = CSV_HEADER + (
        "8888002,http://evil.other.com/wp-content/x,https://phishtank.org/y,"
        "2026-09-13T05:00:00+00:00,yes,2026-09-13T06:00:00+00:00,yes,Netflix\n"
    )
    write_snapshot(tmp_path, {PHISHTANK: parse_phishtank(shrunk)}, allow_shrink=True)
    hosts = [
        json.loads(line)["host"]
        for line in (tmp_path / ENTRIES_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert hosts == ["evil.other.com"]


def test_other_sources_are_left_untouched(tmp_path: Path) -> None:
    write_snapshot(tmp_path, {PHISHTANK: parse_phishtank(CSV_PHISHTANK)}, allow_shrink=False)
    write_snapshot(
        tmp_path, {OPENPHISH: parse_openphish(FEED_OPENPHISH, LAST_MODIFIED)}, allow_shrink=False
    )
    records = [
        json.loads(line)
        for line in (tmp_path / ENTRIES_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert sorted({record["source"] for record in records}) == [OPENPHISH, PHISHTANK]
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert sorted(manifest["sources"]) == [OPENPHISH, PHISHTANK]


def test_shrink_is_refused_without_the_flag(tmp_path: Path) -> None:
    five_rows = CSV_HEADER + "".join(
        f"888800{index},http://h{index}.other.com/x,https://phishtank.org/{index},"
        f"2026-09-13T05:00:00+00:00,yes,2026-09-13T06:00:00+00:00,yes,Netflix\n"
        for index in range(5)
    )
    write_snapshot(tmp_path, {PHISHTANK: parse_phishtank(five_rows)}, allow_shrink=False)
    with pytest.raises(RuntimeError, match="拒絕覆寫"):
        write_snapshot(tmp_path, {PHISHTANK: parse_phishtank(CSV_PHISHTANK)}, allow_shrink=False)


def test_failed_parse_leaves_the_existing_snapshot_untouched(tmp_path: Path) -> None:
    write_snapshot(tmp_path, {PHISHTANK: parse_phishtank(CSV_PHISHTANK)}, allow_shrink=False)
    before_entries = (tmp_path / ENTRIES_FILENAME).read_bytes()
    before_manifest = (tmp_path / MANIFEST_FILENAME).read_bytes()
    with pytest.raises(ValueError):
        parse_phishtank(CSV_PHISHTANK.replace(",yes,2026-09-13T04:12:31+00:00,yes,", ",no,x,yes,"))
    assert (tmp_path / ENTRIES_FILENAME).read_bytes() == before_entries
    assert (tmp_path / MANIFEST_FILENAME).read_bytes() == before_manifest
    assert not list(tmp_path.glob("*.tmp"))
