"""清除 RDAP 快取中已過期的紀錄。

執行：`python -m tools.prune_rdap_cache [--path data/rdap/cache.sqlite3]`

**為什麼需要一個指令，而不是讓快取自己清。** 過期紀錄在下一次查詢同一個網域時
會被覆寫，所以正確性上不需要清除；需要清除的理由是**那個檔案本身就是一份
「這台機器查過哪些網域」的記錄**。留著不動的話，一個七天前就沒用的網域名稱
會在磁碟上待到下一次剛好有人再貼同一個連結為止。

這件事屬於維運，不屬於請求路徑 —— 所以它在 `tools/` 而不是在 `net/` 裡
用一個背景執行緒做。本程式不發出任何網路請求。
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from net.rdap_cache import DEFAULT_CACHE_PATH, RdapCache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清除 RDAP 快取中已過期的紀錄")
    parser.add_argument("--path", type=Path, default=DEFAULT_CACHE_PATH, help="快取檔路徑")
    args = parser.parse_args(argv)

    if not args.path.is_file():
        raise FileNotFoundError(f"RDAP 快取檔不存在：{args.path}")

    cache = RdapCache(args.path)
    before = cache.count()
    removed = cache.prune(now=datetime.now(timezone.utc))
    print(f"快取筆數：{before} → {before - removed}（刪除 {removed} 筆已過期紀錄）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
