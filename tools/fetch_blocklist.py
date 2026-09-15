"""下載 165 涉詐網址的三個 data.gov.tw 資料集，合併成一份本機快照。

執行：`python -m tools.fetch_blocklist [--out data/blocklist] [--allow-shrink]`

**外部格式的知識全部在這個檔案裡** —— 下載網址、中文欄位名稱、UTF-8 BOM、
民國紀年。`scam_guard/blocklist.py` 只認識我們自己定義的 JSON Lines 格式。

**快照怎麼到得了部署環境，兩條路，都由此處被呼叫：**

1. **建置時取得** —— 在映像檔的建置階段執行本程式，快照成為映像檔的一部分。
   快照的新鮮度等於映像檔的新鮮度，而 `data_through` 的檢查會在重建太久之後
   讓它啟動失敗，這是對的。
2. **啟動時取得** —— 進入點在啟動服務之前執行本程式。每次冷啟動多花數十秒
   下載約 12 MB（三份原始檔合計），但快照永遠是新的。

HuggingFace Spaces 免費層的檔案系統在休眠重啟後回到映像檔狀態，
所以第 1 條在那裡是實質上唯一會持久的一條。

兩者都不違反界線 —— 關鍵是**呼叫者是進入點或建置腳本，不是 `scam_guard/`**。
`scam_guard/blocklist.py` MUST NOT 自行下載。

**三個資料集說的不是同一件事**（2026-09-14 實測）：

| | 176455 | 160055 | 165027 |
|---|---|---|---|
| 內容 | 遭停止解析涉詐網站 | 假投資(博弈)網站 | 一頁式數位經濟詐騙網站 |
| 筆數 | 83,323 | 45,258 | 1,612 |
| 唯一主機 | 75,642 | 30,564 | 1,574 |
| 涵蓋 | 民國 11412–11508 | 2022-01-02 – 2025-12-31 | 2023-06-29 – 2026-08-26 |

合併成一個無差別的集合會讓類型資訊消失，所以每筆紀錄攜帶 `source`。
"""

import argparse
import csv
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

from scam_guard.blocklist import ENTRIES_FILENAME, MANIFEST_FILENAME, Entry
from scam_guard.url import normalize_host

DEFAULT_OUT = Path("data/blocklist")
TIMEOUT_SECONDS = 180.0
LICENSE = "政府資料開放授權條款-第 1 版"
# 授權逐 source 記錄而非全域一個欄位：加入國際 feed 之後每個來源的條款不同，
# 一個全域欄位會變成一句假話。
LICENSE_VERIFIED_ON = "2026-09-14"

# 筆數低於既有快照一半時拒絕覆寫。
# ⚠️ 「一半」沒有依據，但它的錯誤成本很低 —— 判錯只是多打一個旗標。
SHRINK_RATIO = 0.5


@dataclass(frozen=True)
class DatasetSpec:
    """一個資料集的外部知識：正式名稱、提供機關、下載網址、格式。"""

    dataset_id: str
    title: str
    agency: str
    url: str
    fmt: str


DATASETS: dict[str, DatasetSpec] = {
    "176455": DatasetSpec(
        dataset_id="176455",
        title="165反詐騙諮詢專線_遭停止解析涉詐網站",
        agency="內政部警政署",
        url=(
            "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/"
            "29E8E643-88ED-4952-B21E-BD42A3B7108C/resource/"
            "D7D372F9-4823-47A9-B345-9F9BEF51FB61/download"
        ),
        fmt="CSV",
    ),
    "160055": DatasetSpec(
        dataset_id="160055",
        title="165反詐騙諮詢專線_假投資(博弈)網站",
        agency="內政部警政署",
        url=(
            "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/"
            "033197D4-70F4-45EB-9FB8-6D83532B999A/resource/"
            "A00B1802-6A4A-42B4-B842-B66A2D937DAE/download"
        ),
        fmt="CSV",
    ),
    "165027": DatasetSpec(
        dataset_id="165027",
        title="數位發展部數位產業署聲請詐騙網域名稱停止解析網址清單",
        agency="數位發展部數位產業署",
        url="https://www-api.moda.gov.tw/OpenData/Files/16352",
        fmt="JSON",
    ),
}

# 欄位名稱是 data.gov.tw 的財產，只出現在這裡。
FIELDS_176455 = ("民國年月", "網域", "網站性質")
FIELDS_160055 = ("網站名稱", "網址", "統計起始日期", "統計結束日期")
FIELDS_165027 = ("編號", "偽冒網址", "網域名稱", "詐騙網站創建日期", "接獲通報日期")

# `retired` 由**執行取得程式的人**依資料集描述判斷後寫入，不由程式從筆數推論。
# 「這個資料集不會再更新了」是一個需要讀公告才知道的事實。
# 160055 的資料集描述明文寫著它已被 176455 取代（「範圍非限縮於原有之假投資
# 或假博奕」），且檔案內最新的一筆停在 2025-12-31。
DEFAULT_RETIRED: dict[str, str] = {
    "160055": "資料集描述明文指出已被 176455 取代（範圍非限縮於原有之假投資或假博奕），"
    "檔案內最新一筆為 2025-12-31",
}


@dataclass(frozen=True)
class RawRecord:
    """單一資料列正規化後的中間形式，尚未依主機合併。"""

    host: str
    url: str
    observed_on: str
    nature: str | None
    site_created_on: str | None


@dataclass(frozen=True)
class ParsedSource:
    """一個資料集解析後的全部結果。"""

    dataset_id: str
    records: tuple[RawRecord, ...]
    data_through: str
    granularity: str


def download(dataset_id: str, timeout: float) -> str:
    """下載一個資料集。逾時或非 200 時拋例外並指出資料集編號與狀態碼。"""
    spec = DATASETS[dataset_id]
    request = urllib.request.Request(spec.url, headers={"User-Agent": "scam-guard/fetch_blocklist"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"下載資料集 {dataset_id} 失敗：回應狀態碼 {response.status}")
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"下載資料集 {dataset_id} 失敗：回應狀態碼 {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下載資料集 {dataset_id} 失敗：連線錯誤 {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"下載資料集 {dataset_id} 失敗：{timeout} 秒內未完成") from exc
    # 三份檔案皆為 UTF-8 with BOM。編碼也是外部格式知識，只出現在這裡。
    return raw.decode("utf-8-sig")


def require_fields(dataset_id: str, expected: tuple[str, ...], actual: list[str]) -> None:
    """欄位缺失或改名時拋例外，並印出資料集編號、預期欄位與**實際標頭**。

    這不是假想：165027 的 data.gov.tw 頁面把欄位寫成「一頁式詐騙購物網站」，
    而實際 JSON 的鍵是「一頁式數位經濟詐騙網站」。一個 `row.get(欄位, "")`
    會讓 1,612 筆安靜地變成 1,612 筆垃圾。
    """
    missing = [field for field in expected if field not in actual]
    if missing:
        raise ValueError(
            f"資料集 {dataset_id} 缺少欄位 {missing}；預期 {list(expected)}、實際標頭為 {actual}"
        )


def minguo_to_year_month(value: str, dataset_id: str) -> str:
    """民國年月（`11508`）轉西元年月（`2026-08`）。粒度為月。

    ⚠️ 轉錯 1911 年不會有人發現 —— `data_through` 會差 1911 年，
    新鮮度檢查因此永遠通過或永遠失敗。以兩個固定值測試。
    """
    text = value.strip()
    if len(text) != 5 or not text.isdigit():
        raise ValueError(f"資料集 {dataset_id} 的『民國年月』格式不符：{value!r}，預期五位數字")
    year = int(text[:3]) + 1911
    month = int(text[3:])
    if not 1 <= month <= 12:
        raise ValueError(f"資料集 {dataset_id} 的『民國年月』月份不合法：{value!r}")
    return f"{year:04d}-{month:02d}"


def slash_date_to_iso(value: str, dataset_id: str, field: str) -> str:
    """`2025/12/31` 轉 `2025-12-31`。"""
    text = value.strip()
    try:
        return datetime.strptime(text, "%Y/%m/%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(
            f"資料集 {dataset_id} 的『{field}』不是合法日期：{value!r}，預期 YYYY/MM/DD"
        ) from exc


def compact_date_to_iso(value: str, dataset_id: str, field: str) -> str:
    """`20260826` 轉 `2026-08-26`。"""
    text = value.strip()
    try:
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(
            f"資料集 {dataset_id} 的『{field}』不是合法日期：{value!r}，預期 YYYYMMDD"
        ) from exc


def parse_176455(text: str) -> ParsedSource:
    """遭停止解析涉詐網站。`data_through` 取『民國年月』的最大值，粒度為月。"""
    reader = csv.DictReader(io.StringIO(text))
    require_fields("176455", FIELDS_176455, list(reader.fieldnames or []))
    records: list[RawRecord] = []
    months: list[str] = []
    for row in reader:
        host = normalize_host(row["網域"])
        month = minguo_to_year_month(row["民國年月"], "176455")
        months.append(month)
        records.append(
            RawRecord(
                host=host,
                url=row["網域"].strip(),
                observed_on=month,
                # 網站性質原樣保存。它是被冒用的**產業**不是案類，且是帶錯字的
                # 自由文字（「金融保健」66 筆、「金融保線」2 筆），
                # 寫成 dict 映射會讓那 68 筆讓整份取得失敗，
                # 寫成 dict.get(nature, DEFAULT) 則是本專案明文禁止的 fallback。
                nature=row["網站性質"].strip(),
                site_created_on=None,
            )
        )
    if not records:
        raise ValueError("資料集 176455 解析結果為 0 筆，拒絕寫出快照")
    latest = max(months)
    return ParsedSource(
        dataset_id="176455",
        records=tuple(records),
        data_through=_last_day_of_month(latest),
        granularity="month",
    )


def parse_160055(text: str) -> ParsedSource:
    """假投資(博弈)網站。`data_through` 取『統計結束日期』的最大值。"""
    reader = csv.DictReader(io.StringIO(text))
    require_fields("160055", FIELDS_160055, list(reader.fieldnames or []))
    records: list[RawRecord] = []
    ends: list[str] = []
    for row in reader:
        # 實測 20,987 / 45,258 筆帶 `www.` 前綴，而 176455 的網域不帶。
        # 正規化一律呼叫 `scam_guard.url`，不在此自行實作 ——
        # 兩邊各算一次，不一致時的症狀是「黑名單裡明明有卻查不到」。
        host = normalize_host(row["網址"].split("/", 1)[0])
        end = slash_date_to_iso(row["統計結束日期"], "160055", "統計結束日期")
        ends.append(end)
        records.append(
            RawRecord(
                host=host,
                url=row["網址"].strip(),
                observed_on=end,
                nature=None,
                site_created_on=None,
            )
        )
    if not records:
        raise ValueError("資料集 160055 解析結果為 0 筆，拒絕寫出快照")
    return ParsedSource(
        dataset_id="160055",
        records=tuple(records),
        data_through=max(ends),
        granularity="day",
    )


def parse_165027(text: str) -> ParsedSource:
    """一頁式數位經濟詐騙網站。`data_through` 取『接獲通報日期』的最大值。

    保留『詐騙網站創建日期』為 `site_created_on` —— 那是**本機就有的網域年齡
    資料**，這批主機不需要 RDAP。
    """
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"資料集 165027 的內容不是 JSON 陣列，實為 {type(rows).__name__}")
    if not rows:
        raise ValueError("資料集 165027 解析結果為 0 筆，拒絕寫出快照")
    require_fields("165027", FIELDS_165027, list(rows[0]))
    records: list[RawRecord] = []
    reported: list[str] = []
    for row in rows:
        require_fields("165027", FIELDS_165027, list(row))
        host = normalize_host(row["網域名稱"])
        report_date = compact_date_to_iso(row["接獲通報日期"], "165027", "接獲通報日期")
        reported.append(report_date)
        created = row["詐騙網站創建日期"].strip()
        records.append(
            RawRecord(
                host=host,
                url=row["偽冒網址"].strip(),
                observed_on=report_date,
                nature=None,
                site_created_on=created or None,
            )
        )
    return ParsedSource(
        dataset_id="165027",
        records=tuple(records),
        data_through=max(reported),
        granularity="day",
    )


PARSERS: dict[str, Callable[[str], ParsedSource]] = {
    "176455": parse_176455,
    "160055": parse_160055,
    "165027": parse_165027,
}


def parse_dataset(dataset_id: str, text: str) -> ParsedSource:
    return PARSERS[dataset_id](text)


def collect(loader: Callable[[str], str]) -> dict[str, ParsedSource]:
    """取得並解析全部資料集。**任一個失敗即整體失敗，不寫出任何檔案。**"""
    return {dataset_id: parse_dataset(dataset_id, loader(dataset_id)) for dataset_id in DATASETS}


def merge_entries(parsed: ParsedSource) -> list[Entry]:
    """同一 source 內同一主機的多筆合併為一筆，計算 `first_seen` 與 `last_seen`。

    不同 source 的同一主機**不**合併 —— 三個資料集說的不是同一件事，
    合併會讓類型資訊消失。
    """
    by_host: dict[str, list[RawRecord]] = {}
    order: list[str] = []
    for record in parsed.records:
        if record.host not in by_host:
            by_host[record.host] = []
            order.append(record.host)
        by_host[record.host].append(record)
    entries: list[Entry] = []
    for host in order:
        group = by_host[host]
        dates = sorted(record.observed_on for record in group)
        natures = [record.nature for record in group if record.nature]
        created = [record.site_created_on for record in group if record.site_created_on]
        entries.append(
            Entry(
                host=host,
                url=group[0].url,
                source=parsed.dataset_id,
                first_seen=dates[0],
                last_seen=dates[-1],
                nature=natures[0] if natures else None,
                site_created_on=min(created) if created else None,
            )
        )
    return entries


def previous_counts(out_dir: Path) -> dict[str, int]:
    """既有快照各 source 的筆數。無既有 manifest 時為空字典。"""
    manifest_path = out_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sources" not in manifest:
        raise ValueError(f"既有 manifest 缺少 sources 欄位：{manifest_path}")
    return {dataset_id: meta["record_count"] for dataset_id, meta in manifest["sources"].items()}


def write_snapshot(
    out_dir: Path,
    parsed_by_id: dict[str, ParsedSource],
    retired: dict[str, str],
    *,
    allow_shrink: bool,
) -> dict[str, object]:
    """寫出 `entries.jsonl` 與 `manifest.json`，先寫暫存檔再原子改名。

    沒有原子改名這一條，一次失敗的更新會同時毀掉舊資料，
    而黑名單是系統裡唯一的硬證據來源。
    """
    before = previous_counts(out_dir)
    entries_by_source = {
        dataset_id: merge_entries(parsed) for dataset_id, parsed in parsed_by_id.items()
    }
    for dataset_id, parsed in parsed_by_id.items():
        old = before.get(dataset_id)
        new = len(parsed.records)
        previous = old if old is not None else "（無既有快照）"
        print(f"資料集 {dataset_id} 筆數：舊 {previous} → 新 {new}")
        if old is not None and new < old * SHRINK_RATIO and not allow_shrink:
            raise RuntimeError(
                f"資料集 {dataset_id} 的筆數由 {old} 掉到 {new}，低於一半，拒絕覆寫。"
                f"確認資料來源無誤後加上 --allow-shrink"
            )

    lines = []
    for dataset_id in parsed_by_id:
        for entry in entries_by_source[dataset_id]:
            record: dict[str, object] = {
                "host": entry.host,
                "url": entry.url,
                "source": entry.source,
                "first_seen": entry.first_seen,
                "last_seen": entry.last_seen,
            }
            if entry.nature is not None:
                record["nature"] = entry.nature
            if entry.site_created_on is not None:
                record["site_created_on"] = entry.site_created_on
            lines.append(json.dumps(record, ensure_ascii=False))
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    manifest: dict[str, object] = {
        "entries_file": ENTRIES_FILENAME,
        "entries_sha256": sha256(payload).hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            dataset_id: {
                "title": DATASETS[dataset_id].title,
                "agency": DATASETS[dataset_id].agency,
                "source_url": DATASETS[dataset_id].url,
                "format": DATASETS[dataset_id].fmt,
                "record_count": len(parsed.records),
                "unique_host_count": len(entries_by_source[dataset_id]),
                "data_through": parsed.data_through,
                "data_through_granularity": parsed.granularity,
                # 日期由**檔案內容**掃出（民國年月、統計結束日期、接獲通報日期），
                # 不是 HTTP 標頭。這比取 `Last-Modified` 強，兩者不可混為一談。
                "data_through_source": "parsed_from_content",
                "retired": dataset_id in retired,
                "retired_reason": retired.get(dataset_id, ""),
                # 三份皆為政府資料開放授權條款，可對第三方顯示、且參與可註冊
                # 網域層比對（165 的資料形態是一個詐騙者窮舉自己網域下的子網域，
                # 實測 `word1018.shop` 有 3,006 個被通報的子網域）。
                "redistributable": True,
                "domain_level_matching": True,
                "license": LICENSE,
                "license_verified_on": LICENSE_VERIFIED_ON,
                "license_note": "",
            }
            for dataset_id, parsed in parsed_by_id.items()
        },
    }

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


class TimedLoader:
    """把逾時秒數綁在 `download` 上的載入器。不用 closure。"""

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    def __call__(self, dataset_id: str) -> str:
        return download(dataset_id, self._timeout)


def parse_retired_argument(values: list[str]) -> dict[str, str]:
    """`--retired 160055=理由` 的解析。未指定時沿用 `DEFAULT_RETIRED`。"""
    retired = dict(DEFAULT_RETIRED)
    for value in values:
        dataset_id, separator, reason = value.partition("=")
        if not separator or not reason.strip():
            raise ValueError(f"--retired 的格式為 資料集編號=理由，實為 {value!r}")
        if dataset_id not in DATASETS:
            raise ValueError(f"--retired 指定了未知的資料集編號：{dataset_id!r}")
        retired[dataset_id] = reason.strip()
    return retired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下載 165 涉詐網址黑名單並寫出本機快照")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="快照輸出目錄")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="下載逾時秒數")
    parser.add_argument(
        "--allow-shrink", action="store_true", help="允許某資料集筆數低於既有快照一半時覆寫"
    )
    parser.add_argument(
        "--retired",
        action="append",
        default=[],
        metavar="ID=理由",
        help="標記某資料集已停止更新（預設已標記 160055）",
    )
    parser.add_argument(
        "--active",
        action="append",
        default=[],
        metavar="ID",
        help="取消某資料集的停止更新標記",
    )
    args = parser.parse_args(argv)

    retired = parse_retired_argument(args.retired)
    for dataset_id in args.active:
        retired.pop(dataset_id, None)

    parsed_by_id = collect(TimedLoader(args.timeout))
    manifest = write_snapshot(args.out, parsed_by_id, retired, allow_shrink=args.allow_shrink)
    sources = manifest["sources"]
    total = sum(meta["record_count"] for meta in sources.values())
    unique = sum(meta["unique_host_count"] for meta in sources.values())
    print(f"合計 {total} 筆、{unique} 個（source, 主機）組合，快照寫出於：{args.out}")
    for dataset_id, meta in sources.items():
        flag = "（已停止更新）" if meta["retired"] else ""
        granularity = meta["data_through_granularity"]
        print(f"  {dataset_id} data_through={meta['data_through']} ({granularity}){flag}")
    return 0


def _last_day_of_month(year_month: str) -> str:
    """`2026-08` → `2026-08-31`。176455 只有月粒度，記成該月最後一天。

    不假裝有日粒度 —— manifest 另記 `data_through_granularity`。
    """
    year, month = (int(part) for part in year_month.split("-"))
    first_of_next = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return (first_of_next - timedelta(days=1)).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
