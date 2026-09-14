"""下載 Public Suffix List 快照 —— 網路 I/O 只在此處。

執行：`python -m tools.fetch_psl [--out data/psl]`

產物是 `public_suffix_list.dat` 與 `manifest.json` 兩個檔案，
由 `scam_guard.url.PublicSuffixList.load()` 讀取。`data/` 已被 `.gitignore`
排除，快照不進版控。

**快照怎麼到得了部署環境，兩條路，都由此處被呼叫：**

1. **建置時取得** —— 在映像檔的建置階段執行本程式，快照成為映像檔的一部分。
   快照的新鮮度等於映像檔的新鮮度，重建太久之後 `load()` 的
   `max_age_days` 檢查會讓它啟動失敗，這是對的。
2. **啟動時取得** —— 進入點在啟動服務之前執行本程式。每次冷啟動多花數秒
   下載約 250 KB，但快照永遠是新的。

HuggingFace Spaces 免費層的檔案系統在休眠重啟後回到映像檔狀態，
所以第 1 條在那裡是實質上唯一會持久的一條。

兩者都不違反界線 —— 關鍵是**呼叫者是進入點或建置腳本，不是 `scam_guard/`**。
`scam_guard/url.py` MUST NOT 自行下載。
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from scam_guard.url import (
    ICANN_BEGIN,
    PSL_FILENAME,
    PSL_MANIFEST_FILENAME,
    PublicSuffixList,
)

SOURCE_URL = "https://publicsuffix.org/list/public_suffix_list.dat"
DEFAULT_OUT = Path("data/psl")
CONNECT_TIMEOUT_SECONDS = 20.0


def download(url: str, timeout: float) -> bytes:
    """下載一份快照。逾時或非 200 時拋例外並指出狀態碼。"""
    request = urllib.request.Request(url, headers={"User-Agent": "scam-guard/fetch_psl"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            if status != 200:
                raise RuntimeError(f"下載 PSL 失敗：{url} 回應狀態碼 {status}")
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"下載 PSL 失敗：{url} 回應狀態碼 {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下載 PSL 失敗：{url} 連線錯誤 {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"下載 PSL 失敗：{url} 於 {timeout} 秒內未完成") from exc


def build_manifest(raw: bytes) -> dict[str, object]:
    """驗證內容並產出 manifest。缺 ICANN 標記或規則數為 0 時拋例外，不寫出任何檔案。"""
    text = raw.decode("utf-8")
    if ICANN_BEGIN not in text:
        raise ValueError(f"PSL 內容缺少 {ICANN_BEGIN} 標記，拒絕寫出快照")
    psl = PublicSuffixList.parse(text)
    if psl.rule_count == 0:
        raise ValueError("PSL 內容的規則數為 0，拒絕寫出快照")
    return {
        "source_url": SOURCE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256(raw).hexdigest(),
        "rule_count": psl.rule_count,
        "icann_rule_count": len(psl.icann_tlds),
        "license": "Mozilla Public License 2.0",
    }


def write_snapshot(out_dir: Path, raw: bytes, manifest: dict[str, object]) -> None:
    """先寫暫存檔再原子改名 —— 一次失敗的更新不得同時毀掉舊快照。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = out_dir / PSL_FILENAME
    manifest_path = out_dir / PSL_MANIFEST_FILENAME
    snapshot_tmp = snapshot.with_suffix(snapshot.suffix + ".tmp")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    snapshot_tmp.write_bytes(raw)
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(snapshot_tmp, snapshot)
    os.replace(manifest_tmp, manifest_path)


def previous_rule_count(out_dir: Path) -> int | None:
    """既有快照的規則數，供人工比對。無既有 manifest 時回傳 `None`。"""
    manifest_path = out_dir / PSL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "rule_count" not in manifest:
        raise ValueError(f"既有 manifest 缺少 rule_count 欄位：{manifest_path}")
    return manifest["rule_count"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下載 Public Suffix List 快照")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="快照輸出目錄")
    parser.add_argument(
        "--timeout", type=float, default=CONNECT_TIMEOUT_SECONDS, help="連線與讀取逾時秒數"
    )
    args = parser.parse_args(argv)

    before = previous_rule_count(args.out)
    raw = download(SOURCE_URL, args.timeout)
    manifest = build_manifest(raw)
    write_snapshot(args.out, raw, manifest)
    previous = before if before is not None else "（無既有快照）"
    print(f"規則數：舊 {previous} → 新 {manifest['rule_count']}")
    print(f"ICANN 第一層標籤數：{manifest['icann_rule_count']}")
    print(f"快照寫出於：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
