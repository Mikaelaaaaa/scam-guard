"""下載 IANA 的 RDAP bootstrap 註冊表快照 —— 網路 I/O 只在此處。

執行：`python -m tools.fetch_rdap_bootstrap [--out data/rdap]`

產物是 `dns.json` 與 `manifest.json` 兩個檔案，由 `net.rdap.RdapBootstrap.load()`
讀取。`data/` 已被 `.gitignore` 排除，快照不進版控。

**為什麼 bootstrap 要預先抓，而年齡不能預先抓。** bootstrap 是一份有限且已知的
清單（1,438 個 TLD 的層級），與 PSL、165 黑名單同性質，`tools/` 抓得動；
而要查年齡的**網域**在訊息送進來之前不存在，沒有東西可以預先抓 ——
那正是 `net/` 必須存在的理由。

`--out` 目錄同時也是 `net.rdap_cache` 預設放快取的地方，兩者不衝突：
本程式只寫 `dns.json` 與 `manifest.json`。

覆蓋率統計需要「總 TLD 數」，來源是 IANA 的已委派 TLD 清單
（`tlds-alpha-by-domain.txt`）—— 同一個權威機關的另一份檔案，
所以本程式抓兩個檔案，但只把 bootstrap 寫成快照。
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from net.rdap import (
    BOOTSTRAP_FILENAME,
    BOOTSTRAP_MANIFEST_FILENAME,
    BOOTSTRAP_SOURCE_URL,
    DEFAULT_BOOTSTRAP_DIR,
    RdapBootstrap,
)

TLD_LIST_URL = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
TIMEOUT_SECONDS = 20.0
USER_AGENT = "scam-guard/fetch_rdap_bootstrap"


def download(url: str, timeout: float) -> bytes:
    """下載一份檔案。逾時或非 200 時拋例外並指出狀態碼。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            if status != 200:
                raise RuntimeError(f"下載失敗：{url} 回應狀態碼 {status}")
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"下載失敗：{url} 回應狀態碼 {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下載失敗：{url} 連線錯誤 {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"下載失敗：{url} 於 {timeout} 秒內未完成") from exc


def parse_tld_list(raw: bytes) -> frozenset[str]:
    """解析 IANA 的已委派 TLD 清單。首行是版本註解，其餘每行一個 TLD。"""
    tlds = {
        line.strip().lower()
        for line in raw.decode("utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    if not tlds:
        raise ValueError(f"TLD 清單解析結果為 0 筆：{TLD_LIST_URL}")
    return frozenset(tlds)


def build_manifest(raw: bytes, delegated: frozenset[str]) -> dict[str, object]:
    """驗證內容並產出 manifest。解析失敗時拋例外，不寫出任何檔案。"""
    payload = json.loads(raw.decode("utf-8"))
    bootstrap = RdapBootstrap.parse(payload)
    covered = sum(1 for tld in delegated if bootstrap.endpoint_for(tld) is not None)
    return {
        "source_url": BOOTSTRAP_SOURCE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256(raw).hexdigest(),
        "publication": payload.get("publication"),
        "https_tld_count": bootstrap.tld_count,
        "delegated_tld_count": len(delegated),
        "covered_delegated_tld_count": covered,
        "tld_list_url": TLD_LIST_URL,
    }


def write_snapshot(out_dir: Path, raw: bytes, manifest: dict[str, object]) -> None:
    """先寫暫存檔再原子改名 —— 一次失敗的更新不得同時毀掉舊快照。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = out_dir / BOOTSTRAP_FILENAME
    manifest_path = out_dir / BOOTSTRAP_MANIFEST_FILENAME
    snapshot_tmp = snapshot.with_suffix(snapshot.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    snapshot_tmp.write_bytes(raw)
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(snapshot_tmp, snapshot)
    os.replace(manifest_tmp, manifest_path)


def report_coverage(bootstrap: RdapBootstrap, delegated: frozenset[str]) -> None:
    """印出覆蓋率，並**逐項點名台灣相關的 TLD** —— 這個訊號在 `.tw` 上有沒有用，
    決定它值不值得預設啟用，不能只看總數。"""
    missing = sorted(tld for tld in delegated if bootstrap.endpoint_for(tld) is None)
    two_letter = [tld for tld in missing if len(tld) == 2]
    covered = len(delegated) - len(missing)
    print(f"已委派 TLD 數：{len(delegated)}")
    print(f"登記了 https RDAP 服務的 TLD 數：{covered}")
    print(f"未登記：{len(missing)}，其中兩字母 ccTLD {len(two_letter)} 個")
    print(f"未登記的兩字母 ccTLD（前 20）：{' '.join(two_letter[:20])}")
    for tld in ("tw", "jp", "cn", "de", "com"):
        endpoint = bootstrap.endpoint_for(tld)
        print(f"  .{tld} → {endpoint if endpoint is not None else '未登記 RDAP 服務'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下載 IANA RDAP bootstrap 快照")
    parser.add_argument("--out", type=Path, default=DEFAULT_BOOTSTRAP_DIR, help="快照輸出目錄")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="連線與讀取逾時秒數")
    args = parser.parse_args(argv)

    raw = download(BOOTSTRAP_SOURCE_URL, args.timeout)
    delegated = parse_tld_list(download(TLD_LIST_URL, args.timeout))
    manifest = build_manifest(raw, delegated)
    write_snapshot(args.out, raw, manifest)
    report_coverage(RdapBootstrap.parse(json.loads(raw.decode("utf-8"))), delegated)
    print(f"bootstrap 發布時間：{manifest['publication']}")
    print(f"快照寫出於：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
