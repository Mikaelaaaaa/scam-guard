"""下載 Tranco 每日清單，寫出一份前 N 名的本機白名單快照。

執行：`python -m tools.fetch_tranco [--top 1000] [--out data/allowlist]`

**外部格式的知識全部在這個檔案裡** —— `tranco-list.eu` 的清單 ID 端點、
下載網址、zip 內的檔名、CSV 的欄位順序。`scam_guard/allowlist.py` 只認識
我們自己定義的 JSON Lines 格式，也不含任何排名門檻的常數。

**快照怎麼到得了部署環境，兩條路，都由此處被呼叫：**

1. **建置時取得** —— 在映像檔的建置階段執行本程式，快照成為映像檔的一部分。
   快照的新鮮度等於映像檔的新鮮度，而 `data_through` 的檢查會在重建太久之後
   讓載入失敗，這是對的。
2. **啟動時取得** —— 進入點在啟動服務之前執行本程式。每次冷啟動多花數秒
   下載約 9.3 MB 的 zip，但快照永遠是新的。

HuggingFace Spaces 免費層的檔案系統在休眠重啟後回到映像檔狀態，
所以第 1 條在那裡是實質上唯一會持久的一條。

兩者都不違反界線 —— 關鍵是**呼叫者是進入點或建置腳本，不是 `scam_guard/`**。

**不使用官方的 `tranco` PyPI 套件**，兩個理由：它在查詢時打網路（把 I/O 帶進
偵測核心，正是 `pyproject.toml` 的 banned-api 界線要防的事），且它會成為一個
新的執行期依賴（本專案已為此拒絕 PyYAML、拼音函式庫、`tldextract`）。
下載用 stdlib `urllib.request`，解壓用 stdlib `zipfile`。

**授權立場（2026-09-14 實查，2026-09-15 覆查）：** `tranco-list.eu/terms` 與
`/license` 皆回 404，網站只列出各**輸入來源**的授權，其中 Cloudflare Radar
為 CC BY-NC 4.0（禁止商業使用）。本程式因此只取前 N 個**網域名稱與名次**，
不取任何其他衍生資料，並把 attribution 字串寫進 manifest。
若日後需要更明確的授權立場，Tranco 首頁的「Configure a custom list」可以
排除特定提供者 —— 該端點與參數**未查證**。
"""

import argparse
import io
import json
import os
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path

from scam_guard.allowlist import ENTRIES_FILENAME, MANIFEST_FILENAME
from scam_guard.url import PublicSuffixList, normalize_host

LIST_ID_URL = "https://tranco-list.eu/top-1m-id"
DOWNLOAD_URL_TEMPLATE = "https://tranco-list.eu/download_daily/{list_id}"
PERMALINK_TEMPLATE = "https://tranco-list.eu/list/{list_id}"
CSV_MEMBER = "top-1m.csv"

DEFAULT_OUT = Path("data/allowlist")
DEFAULT_PSL = Path("data/psl")
TIMEOUT_SECONDS = 180.0

# 前 1,000 名。三條各自獨立的實測證據在此處全部為 0：與 165 黑名單的
# apex 衝突數、落在九個高風險 gTLD 的網域數、獨立釣魚樣本（OpenPhish
# 當日 300 筆）的 apex 命中數。到前 10,000 名出現第一個真正的衝突
# （`mql5.com`，排名 6,881，165 直接指控整個網站）。
#
# ⚠️ 這是一個**參數不是常數**：`add-ablation` MUST 掃描 1,000 / 5,000 /
# 10,000 並回報 FPR 與召回的變化。它寫進 manifest，不寫進 `scam_guard/`。
DEFAULT_TOP = 1000

# PSL 的後綴變動以月計（見 `scam_guard/url.py`），這裡取三個月。
# 這個值只影響取得程式願意用多舊的 PSL 做過濾，與查詢層無關。
DEFAULT_PSL_MAX_AGE_DAYS = 90

# 過濾後筆數低於既有快照一半時拒絕覆寫。沿用 `fetch_blocklist.py`。
# 在這裡比在黑名單更有意義：Tranco 的筆數固定為 N，過濾後只會因 PSL 版本
# 變動而小幅改變，掉一半一定是 PSL 或解析壞了。
SHRINK_RATIO = 0.5

ATTRIBUTION = (
    "Tranco (https://tranco-list.eu/)，其輸入來源為 Cisco Umbrella、Majestic (CC BY 3.0)、"
    "Farsight、Chrome User Experience Report (CC BY-SA 4.0)、Cloudflare Radar (CC BY-NC 4.0)。"
    "Tranco 本身未宣告整體授權（/terms 與 /license 實測皆 404）"
)


@dataclass(frozen=True)
class Row:
    """原始 CSV 的一列：名次與尚未正規化的網域。"""

    rank: int
    domain: str


@dataclass(frozen=True)
class Kept:
    """過濾後保留的一筆：正規化後的可註冊網域與其名次。"""

    domain: str
    rank: int


@dataclass(frozen=True)
class Download:
    """一次下載的結果。`last_modified` 是 `data_through` 的唯一來源。"""

    body: bytes
    last_modified: str


def fetch_list_id(timeout: float) -> str:
    """取得當日清單的 ID。

    **取不到就不寫快照。** 一份無法被引用的白名單，其實驗結果無法複驗，
    而 `add-ablation` 的紀錄依賴這個 ID。
    """
    request = urllib.request.Request(LIST_ID_URL, headers={"User-Agent": "scam-guard/fetch_tranco"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"取得 Tranco 清單 ID 失敗：{LIST_ID_URL} 回應狀態碼 {response.status}"
                )
            list_id = response.read().decode("utf-8").strip()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"取得 Tranco 清單 ID 失敗：{LIST_ID_URL} 回應狀態碼 {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"取得 Tranco 清單 ID 失敗：{LIST_ID_URL} 連線錯誤 {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"取得 Tranco 清單 ID 失敗：{LIST_ID_URL} 於 {timeout} 秒內未完成"
        ) from exc
    if not list_id:
        raise RuntimeError(f"取得 Tranco 清單 ID 失敗：{LIST_ID_URL} 回傳空字串")
    return list_id


def download(url: str, timeout: float) -> Download:
    """下載清單 zip 並取出 HTTP `Last-Modified`。

    `Last-Modified` 缺席時拋例外：Tranco 的檔案內沒有日期欄位，這個標頭是
    `data_through` 的唯一來源，沒有它就沒有新鮮度判準，而一份無法判定新鮮度的
    白名單會安靜地替一批可能已經易主的網域背書。
    """
    request = urllib.request.Request(url, headers={"User-Agent": "scam-guard/fetch_tranco"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"下載 Tranco 清單失敗：{url} 回應狀態碼 {response.status}")
            last_modified = response.headers["Last-Modified"]
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"下載 Tranco 清單失敗：{url} 回應狀態碼 {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下載 Tranco 清單失敗：{url} 連線錯誤 {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"下載 Tranco 清單失敗：{url} 於 {timeout} 秒內未完成") from exc
    if last_modified is None:
        raise RuntimeError(f"下載 Tranco 清單失敗：{url} 的回應不含 Last-Modified 標頭")
    return Download(body=body, last_modified=last_modified)


def extract_csv(raw: bytes) -> str:
    """自 zip 取出清單 CSV。找不到預期成員時印出 zip 內實際的檔名清單。"""
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Tranco 下載內容不是合法的 zip：{len(raw)} 位元組") from exc
    names = archive.namelist()
    if CSV_MEMBER not in names:
        raise RuntimeError(f"Tranco 的 zip 內找不到 {CSV_MEMBER}，實際的檔名清單為 {names}")
    with archive.open(CSV_MEMBER) as handle:
        return handle.read().decode("utf-8")


def parse_rows(text: str) -> tuple[Row, ...]:
    """逐行解析 `<整數>,<網域>`，並驗證名次由 1 起且連續。

    自己拆而不用 `csv`：失敗訊息要指出**行號與該行原文**，而格式承諾只有
    兩欄。名次不連續代表格式改版，那是一個必須有人看一眼的事實，
    不是一個可以跳過的列。
    """
    rows: list[Row] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        rank_text, separator, domain = line.partition(",")
        if not separator or not rank_text.isdigit() or not domain.strip():
            raise ValueError(f"Tranco 清單第 {line_number} 行不符 `<整數>,<網域>`：{line!r}")
        rank = int(rank_text)
        if rows and rank != rows[-1].rank + 1:
            raise ValueError(
                f"Tranco 清單的名次不連續：第 {line_number} 行為 {rank}，前一行為 {rows[-1].rank}"
            )
        if not rows and rank != 1:
            raise ValueError(f"Tranco 清單的名次不由 1 起：第 {line_number} 行為 {rank}")
        rows.append(Row(rank=rank, domain=domain.strip()))
    return tuple(rows)


@dataclass(frozen=True)
class Filtered:
    """過濾的結果與三個可觀測的統計。"""

    kept: tuple[Kept, ...]
    dropped_public_suffix_count: int
    dropped_duplicate_count: int


def filter_rows(rows: tuple[Row, ...], psl: PublicSuffixList, top: int) -> Filtered:
    """取前 `top` 名，先正規化主機再以 PSL 判定，丟棄本身即為 public suffix 者。

    **順序不可顛倒。** 實測前 100,000 名裡有 32 個以 `www.` 開頭的項目
    （例如 `www.gov.uk`），正規化後是 `gov.uk`，而那是一個 public suffix；
    先過濾再正規化會讓這些項目留在白名單裡。

    丟棄的理由不只是乾淨：`phish.wixsite.com` 的可註冊網域是它自己
    （PSL PRIVATE 有 `wixsite.com`），永遠不會等於 `wixsite.com`，
    留著只會讓人以為白名單保護了 Wix，而實際上保護的是不存在的東西。

    正規化與可註冊網域一律呼叫 `scam_guard.url`，不在此自行實作 ——
    兩邊各算一次，不一致時的症狀是「明明在清單裡卻查不到」，沒有錯誤訊息。
    """
    kept: list[Kept] = []
    seen: set[str] = set()
    dropped_public_suffix = 0
    dropped_duplicate = 0
    for row in rows[:top]:
        host = normalize_host(row.domain)
        if psl.registrable_domain(host) != host:
            dropped_public_suffix += 1
            continue
        if host in seen:
            # 正規化會讓 `www.foo.com` 與 `foo.com` 撞在一起。保留名次較前的
            # 那一筆（列已依名次遞增），並把撞掉的筆數記進 manifest ——
            # 安靜地少幾筆與安靜地多幾筆一樣糟。
            dropped_duplicate += 1
            continue
        seen.add(host)
        kept.append(Kept(domain=host, rank=row.rank))
    return Filtered(
        kept=tuple(kept),
        dropped_public_suffix_count=dropped_public_suffix,
        dropped_duplicate_count=dropped_duplicate,
    )


def previous_kept_count(out_dir: Path) -> int | None:
    """既有快照的筆數。無既有 manifest 時回傳 `None`。"""
    manifest_path = out_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "kept_count" not in manifest:
        raise ValueError(f"既有 manifest 缺少 kept_count 欄位：{manifest_path}")
    return manifest["kept_count"]


def last_modified_to_iso(value: str) -> str:
    """HTTP `Last-Modified` 轉 ISO 8601。不合法時拋例外並指出原字串。"""
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Last-Modified 不是合法的 HTTP 日期：{value!r}") from exc


def build_manifest(
    list_id: str,
    top: int,
    last_modified: str,
    raw_row_count: int,
    filtered: Filtered,
    payload: bytes,
) -> dict[str, object]:
    """manifest 的全部欄位。

    `dropped_public_suffix_count` 不是裝飾品：它是「PSL 過濾有沒有真的在跑」
    的唯一可觀測訊號。這個數字突然變成 0 代表 PSL 沒載入或版本壞了，
    而白名單會安靜地多收數十個平台後綴。

    `data_through_source` 記為 `"http_last_modified"`，與黑名單的
    `"parsed_from_content"` 區分開 —— 這是一個**比較弱**的判準，
    兩者不可混淆。
    """
    return {
        "list_id": list_id,
        "permalink": PERMALINK_TEMPLATE.format(list_id=list_id),
        "source_url": DOWNLOAD_URL_TEMPLATE.format(list_id=list_id),
        "threshold": top,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through": last_modified_to_iso(last_modified),
        "data_through_source": "http_last_modified",
        "raw_row_count": raw_row_count,
        "kept_count": len(filtered.kept),
        "dropped_public_suffix_count": filtered.dropped_public_suffix_count,
        "dropped_duplicate_count": filtered.dropped_duplicate_count,
        "entries_file": ENTRIES_FILENAME,
        "entries_sha256": sha256(payload).hexdigest(),
        "attribution": ATTRIBUTION,
    }


def entries_payload(kept: tuple[Kept, ...]) -> bytes:
    """快照的位元組內容。存名次而不是集合 —— 依據文案要寫得出確切名次。"""
    lines = [
        json.dumps({"domain": item.domain, "rank": item.rank}, ensure_ascii=False) for item in kept
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_snapshot(
    out_dir: Path,
    list_id: str,
    top: int,
    last_modified: str,
    raw_row_count: int,
    filtered: Filtered,
    *,
    allow_shrink: bool,
) -> dict[str, object]:
    """寫出 `entries.jsonl` 與 `manifest.json`，先寫暫存檔再原子改名。

    沒有原子改名這一條，一次失敗的更新會同時毀掉舊快照。
    """
    before = previous_kept_count(out_dir)
    new = len(filtered.kept)
    previous = before if before is not None else "（無既有快照）"
    print(f"過濾後筆數：舊 {previous} → 新 {new}")
    if before is not None and new < before * SHRINK_RATIO and not allow_shrink:
        raise RuntimeError(
            f"白名單筆數由 {before} 掉到 {new}，低於一半，拒絕覆寫。"
            f"確認 PSL 與資料來源無誤後加上 --allow-shrink"
        )
    payload = entries_payload(filtered.kept)
    manifest = build_manifest(list_id, top, last_modified, raw_row_count, filtered, payload)

    out_dir.mkdir(parents=True, exist_ok=True)
    entries_path = out_dir / ENTRIES_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME
    entries_tmp = entries_path.with_suffix(entries_path.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    entries_tmp.write_bytes(payload)
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(entries_tmp, entries_path)
    os.replace(manifest_tmp, manifest_path)
    return manifest


def build(rows: tuple[Row, ...], psl: PublicSuffixList, top: int) -> Filtered:
    """驗證筆數並過濾。原始行數少於門檻是一個看起來成功的失敗。"""
    if len(rows) < top:
        raise ValueError(f"Tranco 清單的原始行數為 {len(rows)}，少於門檻 {top}，拒絕寫出快照")
    return filter_rows(rows, psl, top)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下載 Tranco 清單並寫出本機白名單快照")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="快照輸出目錄")
    parser.add_argument("--psl", type=Path, default=DEFAULT_PSL, help="PSL 快照目錄")
    parser.add_argument(
        "--psl-max-age-days",
        type=int,
        default=DEFAULT_PSL_MAX_AGE_DAYS,
        help="PSL 快照可接受的最大天數",
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP, help="保留的名次上限，寫入 manifest 的 threshold"
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="下載逾時秒數")
    parser.add_argument(
        "--allow-shrink", action="store_true", help="允許筆數低於既有快照一半時覆寫"
    )
    args = parser.parse_args(argv)

    psl = PublicSuffixList.load(args.psl, max_age_days=args.psl_max_age_days)
    list_id = fetch_list_id(args.timeout)
    downloaded = download(DOWNLOAD_URL_TEMPLATE.format(list_id=list_id), args.timeout)
    rows = parse_rows(extract_csv(downloaded.body))
    filtered = build(rows, psl, args.top)
    manifest = write_snapshot(
        args.out,
        list_id,
        args.top,
        downloaded.last_modified,
        len(rows),
        filtered,
        allow_shrink=args.allow_shrink,
    )
    print(f"清單 ID：{manifest['list_id']}（{manifest['permalink']}）")
    print(f"原始行數：{manifest['raw_row_count']}、門檻：{manifest['threshold']}")
    print(
        f"丟棄 public suffix {manifest['dropped_public_suffix_count']} 筆、"
        f"正規化後重複 {manifest['dropped_duplicate_count']} 筆"
    )
    print(f"data_through={manifest['data_through']}（{manifest['data_through_source']}）")
    print(f"快照寫出於：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
