"""取得國際釣魚 feed，合併進**既有的**黑名單快照。

執行（不帶來源時什麼都不做）：

    python -m tools.fetch_phishing_feed
    python -m tools.fetch_phishing_feed --source phishtank --phishtank-app-key <KEY>
    python -m tools.fetch_phishing_feed --source openphish --accept-openphish-terms

**兩個來源預設皆不啟用，而且這是刻意的。** 下面兩段查證結果推翻了
「接上兩個公開 feed」這個原本的假設，它們寫在這裡而不是只寫在 design 裡，
是為了讓下一個人在動手之前就看到。

## PhishTank（2026-09-14 以 curl 實查）

| 項目 | 查證結果 |
|---|---|
| 營運者 | Cisco Talos Intelligence Group |
| 費用 | 免費 |
| 更新頻率 | **每小時**（`developer_info.php`：「updated hourly」） |
| 自動抓取 | **需要 application key**，key 由免費註冊取得 |
| 註冊狀態 | `register.php` 回 200，內容為 **「New user registration temporarily disabled.」** |
| 無金鑰下載 | `data/online-valid.json.gz` 實測回 **HTTP 403**，而文件寫的是限流 |
| 規模 | `stats.php`：線上且已驗證 75,856 筆；歷來驗證為真 4,296,942 筆 |
| 授權 | 封存版 ToU 寫「commercial use without charge」；頁首指向**現行的 Cisco EULA**，**未查證** |

也就是說：**現在拿不到金鑰。**

⚠️ **2026-09-15 覆測時 403 沒有重現**：無金鑰的
`online-valid.json.gz` 與 `online-valid.csv.bz2` 都回 200
（301 → 302 → 簽章的 `cdn.phishtank.com`）。403 看來是當時的暫時性攔截
而不是一個穩定的拒絕。**本程式仍然要求金鑰**：官方文件對自動抓取的要求
就是金鑰，而一條今天剛好通的網址不是一個可以依賴的介面。

那份無金鑰的 CSV 已用來驗證解析假設（見 design 的覆驗附記）：
標頭恰好是下方 `PHISHTANK_FIELDS` 那八個欄位、75,959 筆、
`verified` 與 `online` 的值集合都是 `{"yes"}`、
`target` 有 88% 是字串 `"Other"`。

⚠️ 帶金鑰的下載網址取自官方文件的 JSON 範例
（`data.phishtank.com/data/<app key>/online-valid.json.bz2`），
此處用的是同一個模式的 CSV 形式，**帶金鑰的那條路徑未實測**。

## OpenPhish community feed（2026-09-14 實查）

| 項目 | 查證結果 |
|---|---|
| 端點 | `openphish.com/feed.txt` → 302 → GitHub raw |
| 是否需註冊 | 不需要 |
| 規模 | 300 個 URL、224 個唯一主機、208 個唯一可註冊網域 |
| 更新頻率 | **12 小時**（GitHub commit 時間實測為每日 00:00 與 12:00 UTC） |
| 內容 | **只有 URL**，community 層無品牌、IP、產業等欄位 |
| 儲存庫授權 | GitHub API 回 `license: null`，即未宣告 |

其 Terms of Use 有三段直接打到本專案：

> The Services are provided solely for your personal use.

> You agree not to use any part of the Services for any commercial purposes,
> including ... **detection** ...

> ... you agree not to ... **distribute, display, disclose** ... all or any
> portion of the information obtained through the Services available to any
> third party.

本系統的產出是**給使用者看的**判斷依據，而一句「此網址在 OpenPhish 上」
就是第三段所禁止的那件事。

**結論：OpenPhish 的快照 MUST NOT 用於任何面向使用者的部署。**
這不是一個可以靠自律維持的區別，所以 manifest 的 `redistributable` 寫死為
`false` 且**不提供覆寫參數**，由 `BlocklistStore.load()` 在載入時擋下。

## 順手查證的第三個來源，結論是不適用

abuse.ch 的 URLhaus（`urlhaus.abuse.ch/downloads/csv_online/`）實測回 200、
3,324,103 位元組、不需金鑰，看起來理想。但內容是**惡意軟體散布 URL**
（`threat=malware_download`，多為 IP:port 的 Mirai / Mozi 載荷），
與本系統的威脅模型（使用者在 LINE 收到的釣魚連結）不重合。
記在這裡是為了讓下一個人不必再查一次。

## 快照整份重建，不累加

PhishTank 有誤判申訴機制，但**沒有任何欄位標示某筆曾被撤銷**——
已撤銷的項目只是從 `online-valid` 消失。所以本程式一律以整份下載的結果
覆寫該來源的全部紀錄，**不提供任何合併或累積的選項**：
累加會讓撤銷永遠不生效，而那份聯集會單調成長且越來越髒。
"""

import argparse
import bz2
import csv
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path

from scam_guard.blocklist import ENTRIES_FILENAME, MANIFEST_FILENAME, Entry
from scam_guard.url import normalize_host

DEFAULT_OUT = Path("data/blocklist")
TIMEOUT_SECONDS = 180.0
SHRINK_RATIO = 0.5

PHISHTANK = "phishtank"
OPENPHISH = "openphish"

PHISHTANK_URL_TEMPLATE = "http://data.phishtank.com/data/{app_key}/online-valid.csv.bz2"
PHISHTANK_FIELDS = (
    "phish_id",
    "url",
    "submission_time",
    "verified",
    "verification_time",
    "online",
    "target",
)
PHISHTANK_EXPECTED_VALUE = "yes"

OPENPHISH_URL = "https://openphish.com/feed.txt"

# 未提供 application key 時的例外訊息。寫成常數是為了讓測試逐句斷言它 ——
# 這段文字是本 change 最重要的產出之一，下一個人靠它省下一輪查證。
PHISHTANK_KEY_REQUIRED = (
    "啟用 phishtank 來源需要 --phishtank-app-key。"
    "金鑰由免費註冊取得，但 2026-09-14 與 2026-09-15 兩次實測 "
    "phishtank.org/register.php 皆顯示「New user registration temporarily "
    "disabled.」，即目前無法取得金鑰；無金鑰下載 "
    "data.phishtank.com/data/online-valid.json.gz 在 2026-09-14 實測回 HTTP 403，"
    "但 2026-09-15 覆測回 200（重導至簽章的 cdn.phishtank.com）—— "
    "也就是說它通不通每天不一樣。"
    "**不改用無金鑰網址重試**：官方文件對自動抓取的要求就是金鑰，"
    "而一條今天剛好通的網址不是一個可以依賴的介面，"
    "依賴它正是本專案禁止的那種臆測性 fallback"
)

OPENPHISH_TERMS_REQUIRED = (
    "啟用 openphish 來源需要 --accept-openphish-terms。其 Terms of Use 明文寫："
    "「The Services are provided solely for your personal use.」與"
    "「you agree not to ... distribute, display, disclose ... all or any portion "
    "of the information obtained through the Services available to any third party.」"
    "本系統向使用者顯示「此網址在 OpenPhish 上」即落在後者，"
    "因此其快照 MUST NOT 用於任何面向使用者的部署"
)

OPENPHISH_LICENSE = "未宣告授權（GitHub API 回 license: null），適用其 Terms of Use"
OPENPHISH_LICENSE_NOTE = (
    "you agree not to ... distribute, display, disclose ... all or any portion "
    "of the information obtained through the Services available to any third party"
)
PHISHTANK_LICENSE = "PhishTank Archived Terms of Use（commercial use without charge）"
PHISHTANK_LICENSE_NOTE = (
    "terms.php 頁首指向現行的 Cisco End User License Agreement，該條款未查證；"
    "上述引文出自標示為 Archived 的舊條款"
)
LICENSE_VERIFIED_ON = "2026-09-14"


@dataclass(frozen=True)
class FeedSpec:
    """一個來源的外部知識：正式名稱、營運者、授權與比對限制。"""

    source_id: str
    title: str
    agency: str
    license: str
    license_note: str
    redistributable: bool
    data_through_source: str
    granularity: str


FEEDS: dict[str, FeedSpec] = {
    PHISHTANK: FeedSpec(
        source_id=PHISHTANK,
        title="PhishTank online-valid",
        agency="Cisco Talos Intelligence Group",
        license=PHISHTANK_LICENSE,
        license_note=PHISHTANK_LICENSE_NOTE,
        redistributable=True,
        data_through_source="parsed_from_content",
        granularity="second",
    ),
    OPENPHISH: FeedSpec(
        source_id=OPENPHISH,
        title="OpenPhish community feed",
        agency="OpenPhish",
        license=OPENPHISH_LICENSE,
        license_note=OPENPHISH_LICENSE_NOTE,
        redistributable=False,
        data_through_source="http_last_modified",
        granularity="second",
    ),
}


@dataclass(frozen=True)
class Parsed:
    """一個來源解析後的全部結果。"""

    source_id: str
    entries: tuple[Entry, ...]
    data_through: str


@dataclass(frozen=True)
class Download:
    body: bytes
    last_modified: str | None


def download(url: str, source_id: str, timeout: float) -> Download:
    """下載一個來源。逾時或非 200 時拋例外並指出來源名稱與狀態碼。"""
    request = urllib.request.Request(url, headers={"User-Agent": "scam-guard/fetch_phishing_feed"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"下載來源 {source_id} 失敗：回應狀態碼 {response.status}")
            return Download(body=response.read(), last_modified=response.headers["Last-Modified"])
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"下載來源 {source_id} 失敗：回應狀態碼 {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下載來源 {source_id} 失敗：連線錯誤 {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"下載來源 {source_id} 失敗：{timeout} 秒內未完成") from exc


def host_of(url: str, identifier: str, source_id: str) -> str:
    """自 URL 解出主機並以 `scam_guard.url` 正規化。

    正規化一律呼叫核心，不在此自行實作 —— 兩邊各算一次，不一致時的症狀是
    「明明在清單裡卻查不到」，而且沒有任何錯誤訊息。
    """
    rest = url.split("://", 1)[1] if "://" in url else url
    authority = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    hostport = authority.rpartition("@")[2]
    host = hostport.rsplit(":", 1)[0] if hostport.count(":") == 1 else hostport
    if not host:
        raise ValueError(f"來源 {source_id} 的第 {identifier} 筆解不出主機：{url!r}")
    try:
        return normalize_host(host)
    except ValueError as exc:
        raise ValueError(f"來源 {source_id} 的第 {identifier} 筆解不出主機：{url!r}") from exc


def parse_phishtank(text: str) -> Parsed:
    """解析 PhishTank 的 CSV。

    **逐筆驗證 `verified` 與 `online` 皆為 `yes`。** 官方文件說公開下載檔
    只含已驗證且線上的資料，而那是我們所有強度判斷的前提；哪天 Talos 改了
    檔案內容，`verified=no` 會安靜地混進來。不符即拋例外並指出 `phish_id`
    與實際值 —— **不安靜地過濾掉它就算了**，那會讓我們永遠不知道前提已不成立。
    """
    reader = csv.DictReader(io.StringIO(text))
    header = list(reader.fieldnames or [])
    missing = [field for field in PHISHTANK_FIELDS if field not in header]
    if missing:
        raise ValueError(
            f"來源 {PHISHTANK} 缺少欄位 {missing}；"
            f"預期 {list(PHISHTANK_FIELDS)}、實際標頭為 {header}"
        )
    entries: list[Entry] = []
    times: list[str] = []
    for row in reader:
        for field in ("verified", "online"):
            if row[field] != PHISHTANK_EXPECTED_VALUE:
                raise ValueError(
                    f"來源 {PHISHTANK} 的 phish_id={row['phish_id']} 其 {field} 為 "
                    f"{row[field]!r}，預期 {PHISHTANK_EXPECTED_VALUE!r}。"
                    f"公開下載檔只含已驗證且線上的資料是我們所有強度判斷的前提"
                )
        verification_time = row["verification_time"].strip()
        times.append(verification_time)
        entries.append(
            Entry(
                host=host_of(row["url"], row["phish_id"], PHISHTANK),
                url=row["url"].strip(),
                source=PHISHTANK,
                first_seen=verification_time,
                last_seen=verification_time,
                # 被冒用的品牌原樣保存為不透明字串，與 176455 的 `nature` 平行。
                # 它 MUST NOT 被映射為 `ScamType`，也 MUST NOT 自動灌進
                # `brands.json` —— 那會讓一個擁有者不明的網域被列成官方網域。
                target=row["target"].strip() or None,
            )
        )
    if not entries:
        raise ValueError(f"來源 {PHISHTANK} 解析結果為 0 筆，拒絕寫出快照")
    return Parsed(source_id=PHISHTANK, entries=tuple(entries), data_through=max(times))


def parse_openphish(text: str, last_modified: str | None) -> Parsed:
    """解析 OpenPhish 的純文字 feed（每行一個 URL）。

    community 層**沒有任何日期欄位**（官網表格明示無 30 Days Archive 等），
    所以 `data_through` 只能取 HTTP `Last-Modified`，並在 manifest 標記
    `data_through_source` 與政府資料的「由內容掃出」區分開。
    缺這個標頭就沒有新鮮度判準，因此拋例外。
    """
    if last_modified is None:
        raise RuntimeError(
            f"下載來源 {OPENPHISH} 失敗：回應不含 Last-Modified 標頭，"
            f"而該 feed 的內容沒有任何日期欄位，沒有它就無法判定新鮮度"
        )
    data_through = parsedate_to_datetime(last_modified).isoformat()
    entries: list[Entry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        url = line.strip()
        if not url:
            continue
        entries.append(
            Entry(
                host=host_of(url, str(line_number), OPENPHISH),
                url=url,
                source=OPENPHISH,
                first_seen=data_through,
                last_seen=data_through,
                # community 層沒有品牌資訊，且 **MUST NOT 由 URL 猜測** ——
                # 猜品牌是 `url_brand` 的工作，而它有官方網域清單可以對。
                target=None,
            )
        )
    if not entries:
        raise ValueError(f"來源 {OPENPHISH} 解析結果為 0 筆，拒絕寫出快照")
    return Parsed(source_id=OPENPHISH, entries=tuple(entries), data_through=data_through)


def fetch_phishtank(app_key: str, timeout: float) -> Parsed:
    downloaded = download(PHISHTANK_URL_TEMPLATE.format(app_key=app_key), PHISHTANK, timeout)
    return parse_phishtank(bz2.decompress(downloaded.body).decode("utf-8"))


def fetch_openphish(timeout: float) -> Parsed:
    downloaded = download(OPENPHISH_URL, OPENPHISH, timeout)
    return parse_openphish(downloaded.body.decode("utf-8"), downloaded.last_modified)


def read_existing(out_dir: Path) -> tuple[list[dict], dict]:
    """既有快照的紀錄與 manifest。沒有既有快照時回傳空的兩者。"""
    manifest_path = out_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return [], {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries_path = out_dir / manifest["entries_file"]
    records = [
        json.loads(line)
        for line in entries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, manifest


def source_meta(spec: FeedSpec, parsed: Parsed, unique_hosts: int) -> dict[str, object]:
    """該來源在 manifest 裡的那一段。

    `domain_level_matching` **一律為 false**，且不提供參數覆寫：國際 feed 的
    形態是大量寄生在共用平台上（實測當日 300 筆的 OpenPhish 樣本裡
    `godaddysites.com` 一個網域就佔 12 筆，而 PSL 沒收它），讓它們參與
    可註冊網域層比對，等於讓該平台每一個合法使用者命中黑名單。
    """
    return {
        "title": spec.title,
        "agency": spec.agency,
        # PhishTank 記的是**帶佔位符的網址模板**而不是實際下載的網址 ——
        # 後者含 application key，而 manifest 是一個會被貼進 issue 的檔案。
        "source_url": OPENPHISH_URL if spec.source_id == OPENPHISH else PHISHTANK_URL_TEMPLATE,
        "format": "TXT" if spec.source_id == OPENPHISH else "CSV",
        "record_count": len(parsed.entries),
        "unique_host_count": unique_hosts,
        "data_through": parsed.data_through,
        "data_through_granularity": spec.granularity,
        "data_through_source": spec.data_through_source,
        "retired": False,
        "retired_reason": "",
        "redistributable": spec.redistributable,
        "domain_level_matching": False,
        "license": spec.license,
        "license_verified_on": LICENSE_VERIFIED_ON,
        "license_note": spec.license_note,
    }


def record_of(entry: Entry) -> dict[str, object]:
    record: dict[str, object] = {
        "host": entry.host,
        "url": entry.url,
        "source": entry.source,
        "first_seen": entry.first_seen,
        "last_seen": entry.last_seen,
    }
    if entry.target is not None:
        record["target"] = entry.target
    return record


def write_snapshot(
    out_dir: Path,
    parsed_by_id: dict[str, Parsed],
    *,
    allow_shrink: bool,
) -> dict[str, object]:
    """把各來源的紀錄**整份覆寫**進既有快照，先寫暫存檔再原子改名。

    「整份覆寫」不是實作偷懶：PhishTank 已撤銷的項目只是從 `online-valid`
    消失，沒有任何欄位標示撤銷，所以任何形式的累加都會讓撤銷永遠不生效。
    """
    existing_records, existing_manifest = read_existing(out_dir)
    sources: dict[str, object] = dict(existing_manifest["sources"]) if existing_manifest else {}

    for source_id, parsed in parsed_by_id.items():
        old = sources[source_id]["record_count"] if source_id in sources else None
        new = len(parsed.entries)
        previous = old if old is not None else "（無既有紀錄）"
        print(f"來源 {source_id} 筆數：舊 {previous} → 新 {new}")
        if old is not None and new < old * SHRINK_RATIO and not allow_shrink:
            raise RuntimeError(
                f"來源 {source_id} 的筆數由 {old} 掉到 {new}，低於一半，拒絕覆寫。"
                f"確認資料來源無誤後加上 --allow-shrink"
            )

    kept = [record for record in existing_records if record["source"] not in parsed_by_id]
    lines = [json.dumps(record, ensure_ascii=False) for record in kept]
    for source_id, parsed in parsed_by_id.items():
        spec = FEEDS[source_id]
        hosts = {entry.host for entry in parsed.entries}
        sources[source_id] = source_meta(spec, parsed, len(hosts))
        lines.extend(json.dumps(record_of(entry), ensure_ascii=False) for entry in parsed.entries)
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    manifest: dict[str, object] = {
        "entries_file": ENTRIES_FILENAME,
        "entries_sha256": sha256(payload).hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
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


def collect(sources: list[str], app_key: str | None, accepted_terms: bool, timeout: float) -> dict:
    """取得指定的來源。旗標不足時**在任何網路請求之前**拋例外。"""
    if PHISHTANK in sources and not app_key:
        raise ValueError(PHISHTANK_KEY_REQUIRED)
    if OPENPHISH in sources and not accepted_terms:
        raise ValueError(OPENPHISH_TERMS_REQUIRED)
    parsed: dict[str, Parsed] = {}
    for source_id in sources:
        if source_id == PHISHTANK:
            parsed[source_id] = fetch_phishtank(app_key, timeout)
        else:
            parsed[source_id] = fetch_openphish(timeout)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="取得國際釣魚 feed 並合併進既有黑名單快照")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="快照目錄")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        choices=sorted(FEEDS),
        help="要啟用的來源，可重複。**不指定時什麼都不做**",
    )
    parser.add_argument(
        "--phishtank-app-key",
        default=None,
        help=(
            "PhishTank 的 application key。金鑰由免費註冊取得，"
            "但 2026-09-14 實測註冊已關閉，且無金鑰下載回 HTTP 403"
        ),
    )
    parser.add_argument(
        "--accept-openphish-terms",
        action="store_true",
        help=(
            "確認已讀過 OpenPhish 的 Terms of Use："
            "「The Services are provided solely for your personal use.」"
            "與「you agree not to ... distribute, display, disclose ... all or any "
            "portion of the information obtained through the Services available to "
            "any third party.」其快照 MUST NOT 用於任何面向使用者的部署"
        ),
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="下載逾時秒數")
    parser.add_argument(
        "--allow-shrink", action="store_true", help="允許某來源筆數低於既有一半時覆寫"
    )
    args = parser.parse_args(argv)

    if not args.source:
        print("未指定來源，不做任何事。可用的來源：")
        for source_id, spec in FEEDS.items():
            flag = (
                "需 --phishtank-app-key"
                if source_id == PHISHTANK
                else "需 --accept-openphish-terms"
            )
            print(f"  {source_id}：{spec.title}（{spec.agency}），{flag}")
        return 0

    parsed_by_id = collect(
        args.source, args.phishtank_app_key, args.accept_openphish_terms, args.timeout
    )
    manifest = write_snapshot(args.out, parsed_by_id, allow_shrink=args.allow_shrink)
    for source_id in parsed_by_id:
        meta = manifest["sources"][source_id]
        print(
            f"  {source_id} data_through={meta['data_through']}"
            f"（{meta['data_through_source']}）、"
            f"可對外顯示={meta['redistributable']}、"
            f"參與網域層比對={meta['domain_level_matching']}"
        )
    print(f"快照寫出於：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
