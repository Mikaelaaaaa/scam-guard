"""`PublicSuffixList` 的解析、比對、載入檢查與 TLD 集合。

比對實作以 Public Suffix List **官方公開的測試向量**逐項驗證
（`tests/psl_test_vectors.txt`，取自 publicsuffix/list 的 `tests/test_psl.txt`）。
自己實作 PSL 能站得住的唯一理由就是有現成的正確答案可對 ——
萬用與例外規則（`*.ck` / `!www.ck`）在真實資料裡罕見，寫錯不會有人發現。
"""

import json
import re
from hashlib import sha256
from pathlib import Path

import pytest

from scam_guard.url import (
    PSL_FILENAME,
    PSL_MANIFEST_FILENAME,
    PublicSuffixList,
    Section,
)

FIXTURE = """
// 註解行不成為規則。
// ===BEGIN ICANN DOMAINS===
com
tw
com.tw
gov.tw
uk
co.uk
cc
games
zip
*.ck
!www.ck

// ===END ICANN DOMAINS===
// ===BEGIN PRIVATE DOMAINS===
wixsite.com
// ===END PRIVATE DOMAINS===
"""

VECTOR_PATH = Path(__file__).with_name("psl_test_vectors.txt")
SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "psl"
VECTOR_PATTERN = re.compile(r"checkPublicSuffix\((.+?), (.+?)\);")


def fixture_psl() -> PublicSuffixList:
    return PublicSuffixList.parse(FIXTURE)


def write_snapshot(directory: Path, text: str, **manifest_overrides: object) -> Path:
    """在 `directory` 寫出一份可載入的快照，供載入檢查的測試改造。"""
    directory.mkdir(parents=True, exist_ok=True)
    raw = text.encode("utf-8")
    (directory / PSL_FILENAME).write_bytes(raw)
    manifest: dict[str, object] = {
        "source_url": "https://publicsuffix.org/list/public_suffix_list.dat",
        "fetched_at": "2026-09-14T00:00:00+00:00",
        "sha256": sha256(raw).hexdigest(),
        "rule_count": 11,
        "icann_rule_count": 8,
    }
    manifest.update(manifest_overrides)
    (directory / PSL_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def parse_vector(value: str) -> str | None:
    value = value.strip()
    if value == "null":
        return None
    return value.strip("'")


def to_punycode(host: str) -> str:
    """官方向量含 Unicode 主機，而規則與查詢鍵皆為 punycode，查詢前先轉換。"""
    labels = []
    for label in host.lower().split("."):
        labels.append(label if label.isascii() else label.encode("idna").decode("ascii"))
    return ".".join(labels)


# --- 解析 -----------------------------------------------------------------


def test_import_does_not_read_files() -> None:
    """import 不觸發任何 I/O —— 沒跑過取得程式的環境也要跑得起 pytest。"""
    import importlib

    module = importlib.import_module("scam_guard.url")
    assert module.PSL_FILENAME == PSL_FILENAME


def test_sections_are_recorded() -> None:
    psl = fixture_psl()
    assert psl._rules["com.tw"][1] is Section.ICANN
    assert psl._rules["wixsite.com"][1] is Section.PRIVATE


def test_comments_and_blank_lines_are_not_rules() -> None:
    psl = fixture_psl()
    assert "" not in psl._rules
    assert not any(rule.startswith("//") for rule in psl._rules)


def test_missing_icann_marker_raises() -> None:
    with pytest.raises(ValueError, match="ICANN"):
        PublicSuffixList.parse("com\ntw\n")


def test_zero_rules_raises() -> None:
    with pytest.raises(ValueError, match="規則數為 0"):
        PublicSuffixList.parse("// ===BEGIN ICANN DOMAINS===\n// ===END ICANN DOMAINS===\n")


# --- 可註冊網域 -----------------------------------------------------------


def test_taiwan_multilevel_suffix() -> None:
    assert fixture_psl().registrable_domain("www.esunbank.com.tw") == "esunbank.com.tw"


def test_government_domain_is_its_own_registrable_domain() -> None:
    assert fixture_psl().registrable_domain("post.gov.tw") == "post.gov.tw"


def test_uk_multilevel_suffix() -> None:
    assert fixture_psl().registrable_domain("www.evil.co.uk") == "evil.co.uk"


def test_private_section_hosting_platform() -> None:
    """只用 ICANN 段時這個值會是 `wixsite.com`，一筆黑名單就牽連全平台。"""
    assert fixture_psl().registrable_domain("phish123.wixsite.com") == "phish123.wixsite.com"


def test_suffix_itself_has_no_registrable_domain() -> None:
    assert fixture_psl().registrable_domain("com.tw") is None


def test_wildcard_and_exception_rules() -> None:
    psl = fixture_psl()
    # `*.ck` 使 `b.ck` 成為後綴，`a.b.ck` 的可註冊網域因此是它自己。
    assert psl.registrable_domain("a.b.ck") == "a.b.ck"
    # `!www.ck` 是例外，其公共後綴為 `ck`，故 `www.ck` 本身可註冊。
    assert psl.registrable_domain("www.ck") == "www.ck"
    assert psl.public_suffix("www.ck") == "ck"
    assert psl.public_suffix("b.ck") == "b.ck"


def test_ip_literal_has_no_registrable_domain() -> None:
    assert fixture_psl().registrable_domain("192.0.2.1") is None


# --- 官方測試向量 ---------------------------------------------------------


@pytest.mark.skipif(
    not (SNAPSHOT_DIR / PSL_FILENAME).is_file(),
    reason="需要真實 PSL 快照，請先執行 `python -m tools.fetch_psl`",
)
def test_official_vectors() -> None:
    psl = PublicSuffixList.parse((SNAPSHOT_DIR / PSL_FILENAME).read_text(encoding="utf-8"))
    checked = 0
    for line in VECTOR_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("//"):
            # 官方檔案以 `//` 註解掉幾列（`.local` 等非網際網路的 TLD），
            # 那幾列不是期望值。
            continue
        match = VECTOR_PATTERN.search(line)
        if match is None:
            continue
        host = parse_vector(match.group(1))
        expected = parse_vector(match.group(2))
        if host is None:
            continue
        checked += 1
        want = None if expected is None else to_punycode(expected)
        assert psl.registrable_domain(to_punycode(host)) == want, host
    assert checked > 50, f"官方向量只跑到 {checked} 列，檔案可能沒被讀到"


# --- 載入時的一致性檢查 ---------------------------------------------------


def test_max_age_days_is_required(tmp_path: Path) -> None:
    """不傳就 `TypeError` —— 系統 MUST NOT 自行假設一個門檻。"""
    write_snapshot(tmp_path, FIXTURE)
    with pytest.raises(TypeError):
        PublicSuffixList.load(tmp_path)  # type: ignore[call-arg]


def test_expired_snapshot_reports_dates(tmp_path: Path) -> None:
    write_snapshot(tmp_path, FIXTURE, fetched_at="2024-01-01T00:00:00+00:00")
    with pytest.raises(ValueError) as excinfo:
        PublicSuffixList.load(tmp_path, max_age_days=180)
    message = str(excinfo.value)
    assert "2024-01-01" in message
    assert "180" in message


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    write_snapshot(tmp_path, FIXTURE, sha256="0" * 64)
    with pytest.raises(ValueError) as excinfo:
        PublicSuffixList.load(tmp_path, max_age_days=100000)
    assert "sha256" in str(excinfo.value)


def test_missing_manifest_field_raises(tmp_path: Path) -> None:
    write_snapshot(tmp_path, FIXTURE)
    manifest_path = tmp_path / PSL_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["rule_count"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="rule_count"):
        PublicSuffixList.load(tmp_path, max_age_days=100000)


def test_missing_snapshot_names_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tools.fetch_psl"):
        PublicSuffixList.load(tmp_path, max_age_days=100000)


def test_snapshot_without_icann_marker_raises(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "com\ntw\n")
    with pytest.raises(ValueError, match="ICANN"):
        PublicSuffixList.load(tmp_path, max_age_days=100000)


def test_load_succeeds_with_fresh_snapshot(tmp_path: Path) -> None:
    write_snapshot(tmp_path, FIXTURE)
    psl = PublicSuffixList.load(tmp_path, max_age_days=100000)
    assert psl.registrable_domain("www.esunbank.com.tw") == "esunbank.com.tw"


# --- TLD 集合 -------------------------------------------------------------


@pytest.mark.skipif(
    not (SNAPSHOT_DIR / PSL_FILENAME).is_file(),
    reason="需要真實 PSL 快照，請先執行 `python -m tools.fetch_psl`",
)
def test_icann_tlds_from_real_snapshot() -> None:
    psl = PublicSuffixList.parse((SNAPSHOT_DIR / PSL_FILENAME).read_text(encoding="utf-8"))
    for tld in ("tw", "com", "cc", "zip"):
        assert tld in psl.icann_tlds
    for absent in ("usa", "local"):
        assert absent not in psl.icann_tlds


def test_icann_tlds_exclude_private_section() -> None:
    psl = fixture_psl()
    assert "com" in psl.icann_tlds
    assert "wixsite.com" not in psl.icann_tlds
