"""`tools/fetch_blocklist.py` 的解析與失敗行為。**不打真實網路。**

全部以構造的 CSV / JSON 內容測試，測試中不對 data.gov.tw 發出任何請求 ——
在無網路的環境執行 `pytest` 必須全數通過。
"""

import json
from datetime import date
from pathlib import Path

import pytest

from scam_guard.blocklist import ENTRIES_FILENAME, MANIFEST_FILENAME
from tools.fetch_blocklist import (
    DEFAULT_RETIRED,
    collect,
    merge_entries,
    minguo_to_year_month,
    parse_160055,
    parse_165027,
    parse_176455,
    write_snapshot,
)

CSV_176455 = (
    "民國年月,網域,網站性質,法律依據,聲請單位\n"
    "11412,evil.com,金融保險,詐欺犯罪危害防制條例,刑事警察局詐欺犯罪防制中心\n"
    "11508,evil.com,金融保險,詐欺犯罪危害防制條例,刑事警察局詐欺犯罪防制中心\n"
    "11501,typo.example,金融保健,詐欺犯罪危害防制條例,刑事警察局詐欺犯罪防制中心\n"
)

CSV_160055 = (
    "網站名稱,網址,件數,統計起始日期,統計結束日期\n"
    "假投資甲,www.evil.com,1,2023/12/12,2023/12/18\n"
    "假投資乙,bet.example/path,2,2025/12/25,2025/12/31\n"
)

JSON_165027 = json.dumps(
    [
        {
            "編號": "1",
            "一頁式數位經濟詐騙網站": "連至某購物網站首頁",
            "偽冒網址": "https://onepage.example/ucy9mf2#/login",
            "網域名稱": "onepage.example",
            "詐騙網站創建日期": "2026-06-20T15:13:38",
            "接獲通報日期": "20260826",
            "停止解析日期": "20260826",
        }
    ],
    ensure_ascii=False,
)

TEXTS = {"176455": CSV_176455, "160055": CSV_160055, "165027": JSON_165027}


class DictLoader:
    """以構造內容回應的載入器；`fails_on` 指定的資料集會拋例外。"""

    def __init__(self, texts: dict[str, str], fails_on: str | None = None) -> None:
        self._texts = texts
        self._fails_on = fails_on
        self.calls: list[str] = []

    def __call__(self, dataset_id: str) -> str:
        self.calls.append(dataset_id)
        if dataset_id == self._fails_on:
            raise RuntimeError(f"下載資料集 {dataset_id} 失敗：回應狀態碼 503")
        return self._texts[dataset_id]


# --- 解析 -----------------------------------------------------------------


def test_176455_fields() -> None:
    parsed = parse_176455(CSV_176455)
    assert parsed.dataset_id == "176455"
    assert [record.host for record in parsed.records] == ["evil.com", "evil.com", "typo.example"]
    assert parsed.data_through == "2026-08-31"
    assert parsed.granularity == "month"


def test_160055_strips_www_via_shared_normalizer() -> None:
    """`www.evil.com`（160055）與 `evil.com`（176455）正規化後 host 相同。"""
    assert parse_160055(CSV_160055).records[0].host == "evil.com"
    assert parse_176455(CSV_176455).records[0].host == "evil.com"


def test_160055_data_through_is_the_latest_end_date() -> None:
    parsed = parse_160055(CSV_160055)
    assert parsed.data_through == "2025-12-31"
    assert parsed.granularity == "day"


def test_165027_keeps_site_created_on() -> None:
    parsed = parse_165027(JSON_165027)
    (record,) = parsed.records
    assert record.host == "onepage.example"
    assert record.url == "https://onepage.example/ucy9mf2#/login"
    assert record.site_created_on == "2026-06-20T15:13:38"
    assert parsed.data_through == "2026-08-26"


def test_nature_typos_are_kept_verbatim() -> None:
    """『金融保健』是實測存在的錯字（66 筆）。原樣保存，不失敗、不替換。"""
    natures = [record.nature for record in parse_176455(CSV_176455).records]
    assert "金融保健" in natures


def test_minguo_conversion() -> None:
    assert minguo_to_year_month("11412", "176455") == "2025-12"
    assert minguo_to_year_month("11508", "176455") == "2026-08"


def test_data_through_is_within_a_plausible_range() -> None:
    for parsed in (parse_176455(CSV_176455), parse_160055(CSV_160055), parse_165027(JSON_165027)):
        parsed_date = date.fromisoformat(parsed.data_through)
        assert date(2020, 1, 1) <= parsed_date <= date(2030, 1, 1), parsed.dataset_id


# --- 欄位驗證與失敗行為 ---------------------------------------------------


def test_renamed_field_reports_dataset_expected_and_actual_headers() -> None:
    """165027 的頁面文件寫『一頁式詐騙購物網站』，實際鍵是『一頁式數位經濟詐騙網站』。"""
    renamed = json.dumps([{"編號": "1", "一頁式詐騙購物網站": "x"}], ensure_ascii=False)
    with pytest.raises(ValueError) as excinfo:
        parse_165027(renamed)
    message = str(excinfo.value)
    assert "165027" in message
    assert "網域名稱" in message
    assert "一頁式詐騙購物網站" in message


def test_empty_dataset_raises() -> None:
    with pytest.raises(ValueError, match="0 筆"):
        parse_176455("民國年月,網域,網站性質,法律依據,聲請單位\n")


def test_empty_dataset_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        collect(DictLoader({**TEXTS, "176455": "民國年月,網域,網站性質,法律依據,聲請單位\n"}))
    assert list(tmp_path.iterdir()) == []


# --- 合併 -----------------------------------------------------------------


def test_same_host_in_one_source_is_merged() -> None:
    entries = merge_entries(parse_176455(CSV_176455))
    (merged,) = [entry for entry in entries if entry.host == "evil.com"]
    assert merged.first_seen == "2025-12"
    assert merged.last_seen == "2026-08"


def test_same_host_across_sources_stays_two_entries(tmp_path: Path) -> None:
    parsed = collect(DictLoader(TEXTS))
    write_snapshot(tmp_path, parsed, DEFAULT_RETIRED, allow_shrink=False)
    lines = (tmp_path / ENTRIES_FILENAME).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    sources = sorted(record["source"] for record in records if record["host"] == "evil.com")
    assert sources == ["160055", "176455"]


# --- 寫出 -----------------------------------------------------------------


def test_160055_is_retired_by_default(tmp_path: Path) -> None:
    parsed = collect(DictLoader(TEXTS))
    manifest = write_snapshot(tmp_path, parsed, DEFAULT_RETIRED, allow_shrink=False)
    assert manifest["sources"]["160055"]["retired"] is True
    assert manifest["sources"]["160055"]["retired_reason"]
    assert manifest["sources"]["176455"]["retired"] is False


def test_manifest_records_license_and_urls(tmp_path: Path) -> None:
    manifest = write_snapshot(
        tmp_path, collect(DictLoader(TEXTS)), DEFAULT_RETIRED, allow_shrink=False
    )
    assert manifest["license"] == "政府資料開放授權條款-第 1 版"
    for meta in manifest["sources"].values():
        assert meta["source_url"].startswith("https://")


def test_shrink_is_refused_without_the_flag(tmp_path: Path) -> None:
    write_snapshot(tmp_path, collect(DictLoader(TEXTS)), DEFAULT_RETIRED, allow_shrink=False)
    shrunk = dict(TEXTS)
    shrunk["176455"] = "民國年月,網域,網站性質,法律依據,聲請單位\n11508,only.example,釣魚網站,x,y\n"
    parsed = collect(DictLoader(shrunk))
    with pytest.raises(RuntimeError, match="拒絕覆寫"):
        write_snapshot(tmp_path, parsed, DEFAULT_RETIRED, allow_shrink=False)
    write_snapshot(tmp_path, parsed, DEFAULT_RETIRED, allow_shrink=True)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["sources"]["176455"]["record_count"] == 1


def test_failure_midway_leaves_the_existing_snapshot_untouched(tmp_path: Path) -> None:
    write_snapshot(tmp_path, collect(DictLoader(TEXTS)), DEFAULT_RETIRED, allow_shrink=False)
    before_entries = (tmp_path / ENTRIES_FILENAME).read_bytes()
    before_manifest = (tmp_path / MANIFEST_FILENAME).read_bytes()
    loader = DictLoader(TEXTS, fails_on="165027")
    with pytest.raises(RuntimeError, match="503"):
        collect(loader)
    assert (tmp_path / ENTRIES_FILENAME).read_bytes() == before_entries
    assert (tmp_path / MANIFEST_FILENAME).read_bytes() == before_manifest
    assert not list(tmp_path.glob("*.tmp"))
