"""`BlocklistStore` 的載入檢查、新鮮度判定與查詢。以**構造的快照**測試。

真實快照不進版控，每個環境的黑名單版本不同，測試結果因此無法互相比對 ——
全部單元測試使用構造的快照，真實資料只用於手動檢視與消融實驗。
"""

import json
import time
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from scam_guard.blocklist import ENTRIES_FILENAME, MANIFEST_FILENAME, BlocklistStore
from scam_guard.types import ScamType
from scam_guard.url import PublicSuffixList, utc_today
from tests.test_public_suffix import FIXTURE

PSL_WITH_EXAMPLE = FIXTURE.replace("com\ntw\n", "com\ntw\nexample\nnet\n")


def psl_of(text: str = PSL_WITH_EXAMPLE) -> PublicSuffixList:
    return PublicSuffixList.parse(text)


def days_ago(days: int) -> str:
    return (utc_today() - timedelta(days=days)).isoformat()


DEFAULT_ENTRIES = (
    {
        "host": "evil.com",
        "url": "evil.com/a",
        "source": "176455",
        "first_seen": "2025-12",
        "last_seen": "2026-08",
        "nature": "金融保險",
    },
    {
        "host": "evil.com",
        "url": "www.evil.com",
        "source": "160055",
        "first_seen": "2023-12-18",
        "last_seen": "2025-12-31",
    },
    {
        "host": "a1.other.com",
        "url": "a1.other.com",
        "source": "176455",
        "first_seen": "2026-08",
        "last_seen": "2026-08",
        "nature": "電子商務",
    },
    {
        "host": "onepage.example",
        "url": "https://onepage.example/x#/login",
        "source": "165027",
        "first_seen": "2026-08-26",
        "last_seen": "2026-08-26",
        "site_created_on": "2026-06-20T15:13:38",
    },
)


# 三份政府資料的門檻。160055 為 retired，不需要門檻也不得被要求。
GOVERNMENT_MAX_AGE = {"176455": 60, "165027": 60}

# `\u2028`（LINE SEPARATOR）以跳脫字元寫出，不寫成字面字元 ——
# 一個看不見的字元放在原始碼裡，被編輯器吃掉時不會有人發現。
LINE_SEPARATOR_URL = "https://trzsute.zapier.app/eng\u2028https://trexzor.example/usen"

# 給「這條測試要驗的不是新鮮度」的那些測試用，避免固定日期的構造資料隨時間過期。
RELAXED_MAX_AGE = {"176455": 100_000, "165027": 100_000}


def write_snapshot(
    directory: Path,
    entries: tuple[dict[str, object], ...] = DEFAULT_ENTRIES,
    **source_overrides: dict[str, object],
) -> Path:
    """寫出一份可載入的構造快照，供各測試改造。"""
    directory.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n").encode(
        "utf-8"
    )
    (directory / ENTRIES_FILENAME).write_bytes(payload)
    sources: dict[str, object] = {
        "176455": _source("165反詐騙諮詢專線_遭停止解析涉詐網站", days_ago(20), "month", False, ""),
        "160055": _source(
            "165反詐騙諮詢專線_假投資(博弈)網站", "2025-12-31", "day", True, "已被 176455 取代"
        ),
        "165027": _source(
            "數位發展部數位產業署聲請詐騙網域名稱停止解析網址清單", days_ago(20), "day", False, ""
        ),
    }
    for dataset_id, override in source_overrides.items():
        # 未登記的 dataset_id 視為新增一個來源（多來源快照的測試用），
        # 已登記的則是改造既有來源。
        base = (
            sources[dataset_id]
            if dataset_id in sources
            else _source(dataset_id, days_ago(1), "second", False, "")
        )
        sources[dataset_id] = {**base, **override}
    manifest = {
        "entries_file": ENTRIES_FILENAME,
        "entries_sha256": sha256(payload).hexdigest(),
        "fetched_at": f"{utc_today().isoformat()}T00:00:00+00:00",
        "sources": sources,
    }
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def _source(
    title: str, data_through: str, granularity: str, retired: bool, reason: str
) -> dict[str, object]:
    return {
        "title": title,
        "agency": "測試",
        "source_url": "https://example.invalid/download",
        "record_count": 1,
        "unique_host_count": 1,
        "data_through": data_through,
        "data_through_granularity": granularity,
        "data_through_source": "parsed_from_content",
        "retired": retired,
        "retired_reason": reason,
        "redistributable": True,
        "domain_level_matching": True,
        "license": "政府資料開放授權條款-第 1 版",
        "license_verified_on": "2026-09-14",
        "license_note": "",
    }


def store_of(
    tmp_path: Path,
    entries: tuple[dict[str, object], ...] = DEFAULT_ENTRIES,
    **kwargs: object,
) -> BlocklistStore:
    return BlocklistStore.load(
        write_snapshot(tmp_path, entries, **kwargs), psl_of(), max_age_days=GOVERNMENT_MAX_AGE
    )


# --- 載入是顯式動作 -------------------------------------------------------


def test_import_does_not_read_files() -> None:
    """沒跑過取得程式的環境也要跑得起 pytest —— CI 正是這樣的環境。"""
    import importlib

    module = importlib.import_module("scam_guard.blocklist")
    assert module.ENTRIES_FILENAME == ENTRIES_FILENAME


def test_max_age_days_is_required(tmp_path: Path) -> None:
    write_snapshot(tmp_path)
    with pytest.raises(TypeError):
        BlocklistStore.load(tmp_path, psl_of())  # type: ignore[call-arg]


# --- 新鮮度 ---------------------------------------------------------------


def test_stale_source_raises_and_names_it(tmp_path: Path) -> None:
    write_snapshot(tmp_path, **{"176455": {"data_through": days_ago(400)}})
    with pytest.raises(ValueError) as excinfo:
        BlocklistStore.load(tmp_path, psl_of(), max_age_days=GOVERNMENT_MAX_AGE)
    message = str(excinfo.value)
    assert "176455" in message
    assert days_ago(400) in message
    assert "60" in message


def test_fresh_download_with_stale_content_still_raises(tmp_path: Path) -> None:
    """`fetched_at` 是今天但 `data_through` 是九個月前 —— 仍然過期。

    只看 `fetched_at`，一份內容停滯九個月的清單看起來永遠新鮮。
    """
    write_snapshot(tmp_path, **{"165027": {"data_through": days_ago(270)}})
    with pytest.raises(ValueError, match="165027"):
        BlocklistStore.load(tmp_path, psl_of(), max_age_days=GOVERNMENT_MAX_AGE)


def test_retired_source_does_not_block_loading(tmp_path: Path) -> None:
    """160055 的 `data_through` 是 2025-12-31，再舊也不擋啟動，且資料仍可查。"""
    store = store_of(tmp_path)
    sources = {entry.source for entry in store.by_host("evil.com")}
    assert "160055" in sources


# --- 一致性檢查 -----------------------------------------------------------


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        BlocklistStore.load(directory, psl_of(), max_age_days=GOVERNMENT_MAX_AGE)


def test_missing_snapshot_names_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tools.fetch_blocklist"):
        BlocklistStore.load(tmp_path, psl_of(), max_age_days=GOVERNMENT_MAX_AGE)


def test_entry_missing_host_raises_with_line_number(tmp_path: Path) -> None:
    entries = ({"url": "x", "source": "176455", "first_seen": "a", "last_seen": "b"},)
    directory = write_snapshot(tmp_path, entries)
    with pytest.raises(ValueError) as excinfo:
        BlocklistStore.load(directory, psl_of(), max_age_days=GOVERNMENT_MAX_AGE)
    assert "host" in str(excinfo.value)
    assert "第 1 行" in str(excinfo.value)


def test_zero_entries_raises(tmp_path: Path) -> None:
    directory = write_snapshot(tmp_path, ())
    with pytest.raises(ValueError, match="總筆數為 0"):
        BlocklistStore.load(directory, psl_of(), max_age_days=GOVERNMENT_MAX_AGE)


# --- 查詢 -----------------------------------------------------------------


def test_by_host_exact_match_carries_source(tmp_path: Path) -> None:
    store = store_of(tmp_path)
    entries = store.by_host("a1.other.com")
    assert [entry.source for entry in entries] == ["176455"]
    assert entries[0].nature == "電子商務"


def test_by_host_miss_returns_empty(tmp_path: Path) -> None:
    assert store_of(tmp_path).by_host("clean.example") == ()


def test_by_registrable_domain_finds_sibling_hosts(tmp_path: Path) -> None:
    """清單中是 `a1.other.com`，查詢可註冊網域 `other.com` 仍找得到。"""
    entries = store_of(tmp_path).by_registrable_domain("other.com")
    assert [entry.host for entry in entries] == ["a1.other.com"]


def test_same_host_in_two_sources_returns_two_entries(tmp_path: Path) -> None:
    entries = store_of(tmp_path).by_host("evil.com")
    assert sorted(entry.source for entry in entries) == ["160055", "176455"]


def test_path_is_not_part_of_the_key(tmp_path: Path) -> None:
    """清單記錄的是 `evil.com/a`，查的是主機 `evil.com`，仍命中。"""
    store = store_of(tmp_path)
    assert store.by_host("evil.com")[0].url == "evil.com/a"


def test_entries_carry_no_verdict_fields(tmp_path: Path) -> None:
    """查詢層不判定類型或強度。"""
    (entry, *_) = store_of(tmp_path).by_host("evil.com")
    fields = set(type(entry).__dataclass_fields__)
    assert fields == {
        "host",
        "url",
        "source",
        "first_seen",
        "last_seen",
        "nature",
        "site_created_on",
        "target",
    }
    assert not any(isinstance(getattr(entry, name), ScamType) for name in fields)


def test_site_created_on_is_preserved(tmp_path: Path) -> None:
    """165027 帶著本機就有的網域年齡資料，那批主機不需要任何對外查詢。"""
    (entry,) = store_of(tmp_path).by_host("onepage.example")
    assert entry.site_created_on == "2026-06-20T15:13:38"


def test_registrable_domain_index_follows_the_supplied_psl(tmp_path: Path) -> None:
    """換一份把 `other.com` 列為 PRIVATE 後綴的 PSL，索引隨之改變。"""
    directory = write_snapshot(tmp_path)
    default_store = BlocklistStore.load(directory, psl_of(), max_age_days=GOVERNMENT_MAX_AGE)
    assert default_store.by_registrable_domain("other.com")

    private_psl = psl_of(PSL_WITH_EXAMPLE.replace("wixsite.com", "wixsite.com\nother.com"))
    reloaded = BlocklistStore.load(directory, private_psl, max_age_days=GOVERNMENT_MAX_AGE)
    assert reloaded.by_registrable_domain("other.com") == ()
    assert reloaded.by_registrable_domain("a1.other.com")


# --- 效能 -----------------------------------------------------------------


def test_lookup_is_a_hash_lookup(tmp_path: Path) -> None:
    """全量規模下一萬次查詢的平均耗時遠低於 1 毫秒。"""
    entries = tuple(
        {
            "host": f"h{index}.bulk.example",
            "url": f"h{index}.bulk.example",
            "source": "176455",
            "first_seen": "2026-08",
            "last_seen": "2026-08",
            "nature": "金融保險",
        }
        for index in range(130_000)
    )
    store = BlocklistStore.load(
        write_snapshot(tmp_path, entries), psl_of(), max_age_days=GOVERNMENT_MAX_AGE
    )
    assert store.entry_count == 130_000
    started = time.perf_counter()
    for index in range(10_000):
        store.by_host(f"h{index}.bulk.example")
    average_ms = (time.perf_counter() - started) * 1000 / 10_000
    assert average_ms < 0.1, average_ms


def test_url_containing_a_unicode_line_separator_is_one_record(tmp_path: Path) -> None:
    """JSON Lines 以 `\\n` 分隔，而 `str.splitlines()` 還會在 `U+2028` 切開。

    不是假想：PhishTank 2026-09-15 的 online-valid 裡有一筆
    （`phish_id=9410877`）的 URL 內含 `U+2028`。`json.dumps(ensure_ascii=False)`
    不跳脫它，用 `splitlines()` 讀會把那一行切成兩半，
    然後在一個與真正原因毫無關係的地方拋 `JSONDecodeError`。
    """
    entries = DEFAULT_ENTRIES + (
        {
            "host": "trzsute.zapier.app",
            "url": LINE_SEPARATOR_URL,
            "source": "176455",
            "first_seen": "2026-08",
            "last_seen": "2026-08",
            "nature": "金融保險",
        },
    )
    store = store_of(tmp_path, entries=entries)
    (entry,) = store.by_host("trzsute.zapier.app")
    assert entry.url == LINE_SEPARATOR_URL
