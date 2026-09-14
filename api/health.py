"""`GET /health` —— 回答「這個行程現在還能不能給出有意義的判定」。

不回答「行程還活著嗎」：對一個偵測服務，那個問題沒有資訊量。一個載入不到
registry 或權重表的行程在啟動時就崩潰了，根本走不到能回應這個端點的那一步，
所以**能回應 `GET /health` 的行程，定義上就是一個至少能跑的行程** ——
差別只在於資料新不新鮮。狀態因此只有 `"ok"` 與 `"degraded"` 兩態，
沒有 `"unhealthy"`。

**狀態碼恆為 200，`degraded` 也是。** `degraded` 是一個「該去看
`tools/fetch_blocklist.py` 有沒有重跑」的維運事實，不是「這個端點壞了」；
用 503 表達會讓監控系統把一個資料新鮮度的提醒誤判成服務中斷。

**黑名單新鮮度每次呼叫時重算，不信賴啟動時做過的那一次。**
`BlocklistStore.load()` 在啟動時呼叫過一次新鮮度檢查，超過門檻直接 `raise` ——
那保證一個「一啟動就已經過期」的黑名單不會讓行程活起來。但 `blocklist.py` 的
docstring 明寫「一致性檢查全部在此完成，MUST NOT 於每次查詢時執行」，
意思是那個判斷**只在載入當下發生一次**。於是一個在啟動當天通過檢查的黑名單
（`data_through` 距今 3 天、門檻 7 天）會在行程**不重啟的情況下**於第 5 天
悄悄過期，而沒有任何機制會在那一刻說話 —— 系統繼續正常回應、繼續在黑名單
訊號上保持沉默，而沉默在這個系統裡與「沒有命中」完全同一個樣子。

**重算的兩行邏輯是重寫的，不是重用 `blocklist._check_freshness()`。**
理由不是嫌它私有，是職責不同：那個函式的失敗模式是「`raise` 讓載入失敗」，
這裡要的是「回報天數與是否超標，回應仍是 200」——一個是異常，一個是資料。
把它改成公開函式會讓本層依賴一個目前被設計成內部細節的簽章。本模組只讀
`BlocklistStore.manifest`（既有公開屬性，`url_check` 已經在用）並重用
`scam_guard.url.parse_iso_date` 與 `utc_today`（既有公開函式），
**不修改 `scam_guard/` 一行**。

**不檢查語言模型是否載入。** `detect-api` 的 `depends-on` 只有 `[scoring]`，
不含 `llm-layer`；本層的 registry 裡沒有任何一個 LLM 檢查可以回報它載入了沒有，
現在寫這一項是在為一個不存在的東西寫健康檢查。若日後 `llm-layer` 落地且被這個
組裝層掛載，健康檢查 MUST 補上這一項 —— 但不預先開一個永遠回傳固定值的欄位。
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.url import parse_iso_date, utc_today
from scam_guard.weights import WeightTable

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"


class UnregisteredCheck(BaseModel):
    """一個已實作但組裝層選擇不註冊的檢查，以及不註冊的理由。"""

    name: str
    reason: str


class SourceFreshness(BaseModel):
    """黑名單單一 source 的即時新鮮度。"""

    source: str
    data_through: str
    age_days: int
    max_age_days: int
    retired: bool = Field(
        description=(
            "此資料集是否已停用。停用者不再更新，其天數 MUST NOT 影響整體狀態 ——"
            "黑名單過期意味著召回下降而不是答案變錯，一個 2024 年被通報的假投資"
            "網站今天仍然是一個假投資網站。"
        )
    )
    stale: bool = Field(description="是否超過門檻。`retired` 者恆為 `false`。")


class BlocklistHealth(BaseModel):
    """黑名單的整體回報。未被組裝層載入時 `loaded` 為 `false` 且 `sources` 為空。"""

    loaded: bool
    sources: list[SourceFreshness]


class WeightsHealth(BaseModel):
    """權重表「有沒有載入」的證據。不重算任何門檻值。

    ⚠️ `path` 是伺服器上的絕對路徑，會揭露部署目錄結構。留著它是因為
    「載入的是哪一份表」在多個部署共用一份映像時是唯一能分辨的證據，
    而這個端點依 `api/app.py` 記下的部署層要求本來就 MUST NOT 在沒有網路層
    存取控制的情況下公開暴露。這兩件事必須一起成立，只成立一件就該拿掉 `path`。
    """

    path: str
    signals: int


class HealthResponse(BaseModel):
    status: str = Field(description='`"ok"` 或 `"degraded"`，無第三態。')
    checks_registered: int = Field(
        description=(
            "當前 registry 中已啟用的檢查數。這個數字沒有「正確」的門檻"
            "（那取決於部署環境有哪些資料檔案可用），暴露它是為了讓監控系統"
            "至少能發現「這次部署掉到只剩 2 個檢查」這種相對異常。"
        )
    )
    checks_unregistered: list[UnregisteredCheck]
    blocklist: BlocklistHealth
    weights: WeightsHealth


def source_freshness(
    dataset_id: str, meta: dict[str, object], max_age_days: int
) -> SourceFreshness:
    """單一 source 的天數與是否超標。缺欄位即拋例外並指名欄位。"""
    for field_name in ("retired", "data_through"):
        if field_name not in meta:
            raise ValueError(
                f"黑名單 manifest 的 sources[{dataset_id!r}] 缺少欄位 {field_name!r}，"
                f"實際欄位為 {sorted(meta)}"
            )
    retired = bool(meta["retired"])
    data_through = parse_iso_date(meta["data_through"], f"sources[{dataset_id!r}].data_through")
    age_days = (utc_today() - data_through).days
    return SourceFreshness(
        source=dataset_id,
        data_through=data_through.isoformat(),
        age_days=age_days,
        max_age_days=max_age_days,
        retired=retired,
        stale=(not retired) and age_days > max_age_days,
    )


def blocklist_health(store: BlocklistStore | None, max_age_days: int) -> BlocklistHealth:
    """逐 source 重算新鮮度。`store` 為 `None` 代表組裝層沒有載入黑名單。"""
    if store is None:
        return BlocklistHealth(loaded=False, sources=[])
    sources = store.manifest["sources"]
    if not isinstance(sources, dict):
        raise ValueError(f"黑名單 manifest 的 sources 必須為物件，實為 {sources!r}")
    freshness: list[SourceFreshness] = []
    for dataset_id, meta in sorted(sources.items()):
        if not isinstance(meta, dict):
            raise ValueError(
                f"黑名單 manifest 的 sources[{dataset_id!r}] 必須為物件，實為 {type(meta).__name__}"
            )
        freshness.append(source_freshness(dataset_id, meta, max_age_days))
    return BlocklistHealth(loaded=True, sources=freshness)


def build_health(
    registry: CheckRegistry,
    unregistered: Sequence[tuple[str, str]],
    store: BlocklistStore | None,
    max_age_days: int,
    table: WeightTable,
) -> HealthResponse:
    """組出一次健康檢查回應。不發出任何網路請求，也不呼叫 `detect()`。"""
    blocklist = blocklist_health(store, max_age_days)
    degraded = any(source.stale for source in blocklist.sources)
    return HealthResponse(
        status=STATUS_DEGRADED if degraded else STATUS_OK,
        checks_registered=len(registry.enabled()),
        checks_unregistered=[
            UnregisteredCheck(name=name, reason=reason) for name, reason in unregistered
        ],
        blocklist=blocklist,
        weights=WeightsHealth(path=str(table.path), signals=len(table.signals)),
    )
