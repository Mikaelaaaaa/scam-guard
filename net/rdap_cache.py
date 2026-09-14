"""RDAP 查詢結果的本機快取 —— 在 `net/`，不在 `scam_guard/`。

7 天快取需要**寫檔**，而偵測核心至今沒有寫過任何檔案。「不做 I/O」若只解釋成
「不上網、但可以寫檔」，界線就又鬆了一格。把快取放在 `lookup` 之後，
核心的呼叫語意完全不變：它問一個網域的年齡，拿到一個答案，
不知道答案來自網路還是 SQLite。

**用 `sqlite3` 而非 JSON 檔，理由具體。** Gradio 以執行緒池處理請求（模型與
介面同一個程序），`detect-api` 之後還會有多 worker 行程的版本。兩個執行緒或
兩個行程對同一個 JSON 檔做讀-改-寫會互相覆蓋，而且**不會有任何錯誤訊息** ——
只會讓快取命中率莫名其妙地低。`sqlite3` 在 stdlib 裡，兩種情形都不需要新依賴。

**`UNAVAILABLE` 的快取語意是「暫不重試」，不是「這個網域的年齡未知」。**
差別在讀取端：命中短快取時回傳的仍是 `UNAVAILABLE`，永遠不會因為快取命中
就變成別的 outcome。完全不快取它會有具體的壞處 —— 一則訊息含五個同網域的
URL、或使用者連續轉傳三則同樣的詐騙簡訊，就會對一個正在限流的端點連打數次，
把限流狀態延長。

**只存四個欄位，不存 RDAP 的原始回應。** 原始回應含註冊人、聯絡方式、註冊商
—— 第三方個資，存下來就進入 `pii-redact` 的管轄範圍，而我們需要的只有一個日期。
不取、不存，是最省事的合規方式。

⚠️ **這個檔案本身就是一份「使用者查過哪些網域」的記錄。** `data/` 已被
`.gitignore` 排除，但那只防進版控，不防外洩。過期資料在下一次查詢同網域時
被覆寫，不會自動清除 —— 主動清除用 `python -m tools.prune_rdap_cache`。

⚠️ **7 天在實際部署上拿不到 7 天。** 執行環境是 HuggingFace Spaces 免費層，
檔案系統在休眠重啟後回到映像檔狀態，快取存活期因此等於 Space 的連續運作時間。
後果是對外查詢次數比設計假設的多，也就是**外流面比預期大**。這不是技術問題，
是要算進「要不要啟用這個檢查」的決定裡的事實。
"""

import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from scam_guard.domain_age import AgeOutcome, DomainAge

DEFAULT_CACHE_PATH = Path("data/rdap/cache.sqlite3")
DEFAULT_KNOWN_TTL = timedelta(days=7)
DEFAULT_UNAVAILABLE_TTL = timedelta(minutes=15)
"""⚠️ 分鐘級，且**沒有實驗依據** —— RDAP 的速率限制普遍不公布具體數字
（查得到的只有 rdap.org 自己公布的 10 req / 10s），所以這個值只能保守。
與 30 天門檻同性質：是參數不是常數，寫成建構參數供 `add-ablation` 掃描。"""

SQLITE_TIMEOUT_SECONDS = 5.0

TABLE = "domain_age"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    domain TEXT PRIMARY KEY,
    outcome TEXT NOT NULL,
    registered_on TEXT,
    fetched_at TEXT NOT NULL
)
"""


class RdapCache:
    """網域 → 查詢結果的本機快取。TTL 依 outcome 而不同。

    `now` 是每個方法的必填僅限關鍵字參數，不由本類別讀時鐘：過期行為是這個
    類別的全部邏輯，而一個自己讀時鐘的類別要測它就只能改系統時間或打補丁。
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_CACHE_PATH,
        *,
        known_ttl: timedelta = DEFAULT_KNOWN_TTL,
        unavailable_ttl: timedelta = DEFAULT_UNAVAILABLE_TTL,
    ) -> None:
        self._path = Path(path)
        self._known_ttl = known_ttl
        self._unavailable_ttl = unavailable_ttl
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # 每次操作開一條連線而非共用一條：`sqlite3` 的連線預設不可跨執行緒，
        # 而 Gradio 以執行緒池處理請求。開檔的成本遠小於一次 RDAP 往返。
        return sqlite3.connect(self._path, timeout=SQLITE_TIMEOUT_SECONDS)

    def ttl_for(self, outcome: AgeOutcome) -> timedelta:
        """該 outcome 的存活時間。`UNAVAILABLE` MUST NOT 與其他兩者同 TTL。"""
        if outcome is AgeOutcome.UNAVAILABLE:
            return self._unavailable_ttl
        return self._known_ttl

    def get(self, domain: str, *, now: datetime) -> DomainAge | None:
        """未過期的快取結果，沒有或已過期時回傳 `None`。

        命中時**原樣回傳存進去的那個 outcome**，不做任何轉換 ——
        快取是一層存取最佳化，不是一個會改變答案的地方。
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT outcome, registered_on, fetched_at FROM {TABLE} WHERE domain = ?",
                (domain,),
            ).fetchone()
        if row is None:
            return None
        # 未知的 outcome 字串在此拋 `ValueError`。快取檔被改壞時應該大聲壞掉，
        # 而不是安靜地當成「沒有快取」再去打一次網路。
        outcome = AgeOutcome(row[0])
        fetched_at = datetime.fromisoformat(row[2])
        if _as_utc(now) - fetched_at >= self.ttl_for(outcome):
            return None
        registered_on = date.fromisoformat(row[1]) if row[1] is not None else None
        return DomainAge(domain=domain, outcome=outcome, registered_on=registered_on)

    def put(self, age: DomainAge, *, now: datetime) -> None:
        """寫入或覆寫一筆結果。"""
        registered_on = age.registered_on.isoformat() if age.registered_on is not None else None
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT INTO {TABLE} (domain, outcome, registered_on, fetched_at) "
                f"VALUES (?, ?, ?, ?) "
                f"ON CONFLICT(domain) DO UPDATE SET "
                f"outcome = excluded.outcome, "
                f"registered_on = excluded.registered_on, "
                f"fetched_at = excluded.fetched_at",
                (age.domain, age.outcome.value, registered_on, _as_utc(now).isoformat()),
            )

    def prune(self, *, now: datetime) -> int:
        """刪除已過期的紀錄，回傳刪除筆數。

        兩個 TTL 分兩道 DELETE。`fetched_at` 一律以 UTC 寫入，偏移量相同，
        所以 ISO 8601 字串的字典序比較與時間先後一致。
        """
        moment = _as_utc(now)
        short_cutoff = (moment - self._unavailable_ttl).isoformat()
        long_cutoff = (moment - self._known_ttl).isoformat()
        with closing(self._connect()) as conn, conn:
            short = conn.execute(
                f"DELETE FROM {TABLE} WHERE outcome = ? AND fetched_at <= ?",
                (AgeOutcome.UNAVAILABLE.value, short_cutoff),
            ).rowcount
            long = conn.execute(
                f"DELETE FROM {TABLE} WHERE outcome != ? AND fetched_at <= ?",
                (AgeOutcome.UNAVAILABLE.value, long_cutoff),
            ).rowcount
        return short + long

    def count(self) -> int:
        """快取中的總筆數，供 `tools/` 的清除指令列印前後對照。"""
        with closing(self._connect()) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]


def _as_utc(moment: datetime) -> datetime:
    """把時間轉為 UTC。無時區資訊時拋 `ValueError` —— 不猜它是哪個時區。"""
    if moment.tzinfo is None:
        raise ValueError(f"快取的時間參數必須帶時區資訊，實為 naive datetime：{moment!r}")
    return moment.astimezone(timezone.utc)
