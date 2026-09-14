"""RDAP 網域查詢 —— 本專案唯一在請求路徑上發出的對外請求。

**為什麼是 RDAP 而不是展開短網址（兩者都是網路請求，為何一個做一個不做）：**
對象不同。RDAP 送到的是**註冊局**，一個與詐騙者無關的第三方，它學到的是
「有人查了這個網域」；展開短網址送到的是**詐騙者控制的伺服器**，它學到的是
我們的 IP、這個連結仍在流通、以及可以據此對後續訪客改變內容。

**為什麼是 RDAP 而不是 WHOIS：** RDAP 是 WHOIS 的後繼協定，回應是結構化 JSON；
WHOIS 是自由文字，每個註冊局格式不同，解析它等於為每個 TLD 寫一個 parser。

**誠實揭露送出了什麼：** 每次查詢送出的恰好是一個可註冊網域（punycode 形式），
其餘只有協定必需的欄位與一個固定的 User-Agent。**不送**完整 URL、path、query、
主機的子網域、訊息內容、使用者識別。註冊局（以及路徑上的觀察者，透過 TLS SNI
與 DNS 查詢）會知道「某個 IP 在某個時間查了這個網域」；它學不到是誰收到訊息、
訊息內容是什麼、我們最後判成什麼。這比展開短網址少非常多，但它**不是零**。

不送子網域不是保守，是協定決定的：RDAP 的網域查詢對象是**註冊**的網域。
`login-esunbank.evil.com` 送 `evil.com` 就夠，送子網域反而多洩漏一件事 ——
那個子網域是詐騙者為這波攻擊取的，它本身就是攻擊活動的資訊。

**不重試、不退避。** 使用者在等，一次請求裡重試只是把延遲加倍。429 歸
`UNAVAILABLE`，由短快取抑制後續重試。若日後有批次評估需求，那是 `tools/`
的工作，可以慢慢跑，不走這條路徑。

⚠️ **IANA 的 RDAP bootstrap 不涵蓋全部 TLD。** 2026-09-09 發布的快照實測：
1,438 個已委派 TLD 中只有 1,200 個登記了 RDAP 服務，缺的 238 個裡有
178 個是兩字母 ccTLD —— `.jp`、`.cn`、`.de`、`.io`、`.co` 都不在。
本模組再排除只登記 `http://` 端點的 `.kg` 與 `.mg`，實際可用的是 **1,198**，
未涵蓋 240（其中兩字母 ccTLD 180）。對這些 TLD 本檢查恆為 `NO_DATA`。

⚠️⚠️ **`.tw` 在 bootstrap 裡，但本客戶端拿不到它的資料。**
TWNIC 的端點是 `https://ccrdap.twnic.tw/tw/`，而且它**確實提供** `registration`
事件（以 HTTP/2 查 `esunbank.com.tw` 得到 `1997-05-01T03:57:36Z`）。
問題在協定版本：**TWNIC 對 HTTP/1.1 回 426 Upgrade Required**，
而 stdlib 的 `http.client` 只會說 HTTP/1.1。實測 22 個主要 TLD 端點中
只有 `.tw` 這麼做，但 `.tw` 正是台灣詐騙網域最相關的那一個。

後果要直說：**在不引入 HTTP/2 客戶端的前提下，這個訊號在 `.tw` 上是空的。**
要補上它就得加一個第三方依賴（`httpx[http2]` 之流），而本專案目前執行期
依賴為零、執行環境是 2 vCPU 的免費層 —— 那是一個比本 change 更大的決定，
不在這裡單方面做。此事實已寫入 `design.md` 的 Risks，並且是評估
「這個檢查值不值得預設啟用」時最重要的一個輸入。
"""

import http.client
import json
import socket
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from net.rdap_cache import RdapCache
from scam_guard.domain_age import AgeOutcome, DomainAge

BOOTSTRAP_SOURCE_URL = "https://data.iana.org/rdap/dns.json"
BOOTSTRAP_FILENAME = "dns.json"
BOOTSTRAP_MANIFEST_FILENAME = "manifest.json"
BOOTSTRAP_FETCH_COMMAND = "python -m tools.fetch_rdap_bootstrap"
DEFAULT_BOOTSTRAP_DIR = Path("data/rdap")

USER_AGENT = "scam-guard/rdap"
"""固定字串，不含版本以外的任何可變內容 —— 標頭不是夾帶識別資訊的地方。"""

REGISTRATION_EVENT = "registration"

DEFAULT_CONNECT_TIMEOUT_SECONDS = 3.0
DEFAULT_READ_TIMEOUT_SECONDS = 5.0
"""⚠️ 兩個值都沒有實驗依據，只有一個明確的約束：執行環境是 2 vCPU，
同一個程序裡還有 Gemma 3 1B 在推論，而 RDAP 往返是**序列**發生在推論之前的。
所以上限設得比一般預設緊，單一端點合計不超過數秒。"""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """一次 HTTP 回應。只保留判斷需要的兩項。"""

    status: int
    body: bytes


class RdapTransport(Protocol):
    """發出一次 GET 的能力。抽出來是為了讓 `RdapLookup` 的全部判斷邏輯
    （狀態碼分類、例外分類、事件解析、快取）在**沒有網路**的情況下可測 ——
    `rdap-lookup` 的 spec 要求測試不得對外發出請求。"""

    def __call__(self, url: str) -> HttpResponse: ...


class HttpsTransport:
    """以 `http.client` 發出一次 HTTPS GET，**連線與讀取的逾時分開設定**。

    不用 `urllib.request` 的唯一理由就是這個：它只有單一 `timeout` 參數，
    連線慢與回應慢分不開，而兩者該容忍的時間長度不一樣。

    只支援 `https`。bootstrap 裡有兩個 TLD（`.kg`、`.mg`）只登記了 `http://`
    端點，對它們發出明文查詢會把網域暴露給路徑上的每一個觀察者 ——
    那正是本模組費力避免的事，所以它們由 `RdapBootstrap` 直接排除。
    """

    def __init__(
        self,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    ) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout

    def __call__(self, url: str) -> HttpResponse:
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise ValueError(f"RDAP 查詢只走 https，實為 {parts.scheme!r}：{url}")
        if not parts.hostname:
            raise ValueError(f"RDAP 端點沒有主機：{url}")
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        conn = http.client.HTTPSConnection(
            parts.hostname, parts.port, timeout=self._connect_timeout
        )
        try:
            conn.connect()
            conn.sock.settimeout(self._read_timeout)
            conn.request(
                "GET",
                path,
                headers={"Accept": "application/rdap+json", "User-Agent": USER_AGENT},
            )
            response = conn.getresponse()
            return HttpResponse(status=response.status, body=response.read())
        finally:
            conn.close()


class RdapBootstrap:
    """TLD → RDAP 服務端點的對照，來自 IANA 的 bootstrap 註冊表。

    **端點只從快照解析，不在查詢時下載，也不猜。** 猜端點（例如
    `rdap.nic.<tld>`）會把查詢送到一個我們沒有理由信任的主機上 ——
    那比沒有答案糟。
    """

    def __init__(self, endpoints: dict[str, str]) -> None:
        self._endpoints = endpoints

    @property
    def tld_count(self) -> int:
        return len(self._endpoints)

    @classmethod
    def parse(cls, payload: object) -> "RdapBootstrap":
        """解析 `dns.json` 的內容。結構不符時拋 `ValueError`。

        `services` 的每一項形如 `[[tld, ...], [url, ...]]`。同一個 TLD 只取
        第一個 `https://` 端點 —— 目前的快照裡沒有任何一項登記多個端點，
        取第一個與取任何一個沒有差別；非 https 的項目直接不收錄，
        於是那些 TLD 走「不在 bootstrap 中」那條路徑，回 `NO_DATA`。
        """
        if not isinstance(payload, dict):
            raise ValueError(f"RDAP bootstrap 的內容不是 JSON 物件，實為 {type(payload).__name__}")
        services = payload.get("services")
        if not isinstance(services, list) or not services:
            raise ValueError("RDAP bootstrap 缺少非空的 services 陣列，拒絕使用")
        endpoints: dict[str, str] = {}
        for index, service in enumerate(services):
            if not isinstance(service, list) or len(service) < 2:
                raise ValueError(f"RDAP bootstrap 的 services 第 {index} 項格式不符：{service!r}")
            tlds, urls = service[0], service[1]
            if not isinstance(tlds, list) or not isinstance(urls, list):
                raise ValueError(f"RDAP bootstrap 的 services 第 {index} 項格式不符：{service!r}")
            https = [url for url in urls if isinstance(url, str) and url.startswith("https://")]
            if not https:
                continue
            for tld in tlds:
                if not isinstance(tld, str) or not tld:
                    raise ValueError(f"RDAP bootstrap 的 services 第 {index} 項含非法 TLD：{tld!r}")
                endpoints.setdefault(tld.lower(), https[0])
        if not endpoints:
            raise ValueError("RDAP bootstrap 解析後沒有任何 https 端點，拒絕使用")
        return cls(endpoints)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BOOTSTRAP_DIR) -> "RdapBootstrap":
        """自本機快照目錄載入。快照缺失時拋 `FileNotFoundError` 並指出取得指令。"""
        snapshot = Path(path) / BOOTSTRAP_FILENAME
        if not snapshot.is_file():
            raise FileNotFoundError(
                f"RDAP bootstrap 快照不存在：{snapshot}。請先執行 `{BOOTSTRAP_FETCH_COMMAND}` 取得"
            )
        return cls.parse(json.loads(snapshot.read_text(encoding="utf-8")))

    def endpoint_for(self, tld: str) -> str | None:
        """該 TLD 的 RDAP 服務端點。未登記時回傳 `None`。"""
        return self._endpoints.get(tld.lower())


class RdapLookup:
    """`DomainAgeLookup` 的 RDAP 實作。組裝層建立它並注入 `DomainAgeCheck`。

    例外的捕捉全部發生在這裡且**逐一列舉型別**，每一個都對應一個已知的失敗模式：

        TimeoutError              連線或讀取逾時
        socket.gaierror           端點主機的 DNS 解析失敗
        ConnectionError           連線被拒、被重設、中斷
        ssl.SSLError              TLS 交握或憑證驗證失敗
        http.client.HTTPException 回應不是合法的 HTTP
        json.JSONDecodeError      回應不是合法的 JSON
        UnicodeDecodeError        回應不是合法的 UTF-8

    不在清單上的例外**照常往上拋**。那代表出現了我們沒想到的失敗，
    應該大聲壞掉，而不是被寫成「這個網域查不到」—— 後者會讓一個程式錯誤
    永遠偽裝成一個關於網域的事實。
    """

    def __init__(
        self,
        bootstrap: RdapBootstrap,
        cache: RdapCache,
        transport: RdapTransport,
    ) -> None:
        self._bootstrap = bootstrap
        self._cache = cache
        self._transport = transport

    def __call__(self, domain: str) -> DomainAge:
        now = datetime.now(timezone.utc)
        cached = self._cache.get(domain, now=now)
        if cached is not None:
            return cached
        age = self._fetch(domain)
        self._cache.put(age, now=now)
        return age

    def _fetch(self, domain: str) -> DomainAge:
        endpoint = self._bootstrap.endpoint_for(domain.rsplit(".", 1)[-1])
        if endpoint is None:
            # 沒有登記 RDAP 服務是關於**這個 TLD** 的事實，重試一萬次也一樣，
            # 所以是 NO_DATA 而不是 UNAVAILABLE。並且不發出任何請求。
            return DomainAge(domain=domain, outcome=AgeOutcome.NO_DATA)
        url = f"{endpoint.rstrip('/')}/domain/{domain}"
        try:
            response = self._transport(url)
        except (
            TimeoutError,
            socket.gaierror,
            ConnectionError,
            ssl.SSLError,
            http.client.HTTPException,
        ):
            return DomainAge(domain=domain, outcome=AgeOutcome.UNAVAILABLE)
        if response.status in (404, 426):
            # 404 —— 註冊局明確回答了「查無此網域」，那是關於該網域的事實，
            # 不是關於本次請求的事實，所以不該被短快取後一再重問。
            #
            # 426 Upgrade Required —— 註冊局拒絕 HTTP/1.1。**TWNIC（`.tw`）
            # 就是這一種**，見模組 docstring。它歸 NO_DATA 而不是 UNAVAILABLE，
            # 理由與 404 同一條：本客戶端只會說 HTTP/1.1，重試一萬次都會拿到
            # 同一個 426。歸成 UNAVAILABLE 會讓系統每隔幾分鐘就對 TWNIC
            # 重打一次一個注定失敗的請求 —— 那正是 NO_DATA 這個分類存在的理由。
            return DomainAge(domain=domain, outcome=AgeOutcome.NO_DATA)
        if response.status != 200:
            # 429、5xx 與其餘一切（含未追隨的 3xx）都歸 UNAVAILABLE：
            # 它們的共同點是「註冊局沒有給我們關於這個網域的答案」，
            # 而那是關於這次請求的事實。實測 22 個端點皆直接回
            # 200 / 404 / 429 / 426，沒有一個回 3xx，所以本客戶端不追隨轉址。
            return DomainAge(domain=domain, outcome=AgeOutcome.UNAVAILABLE)
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return DomainAge(domain=domain, outcome=AgeOutcome.UNAVAILABLE)
        if not isinstance(payload, dict):
            return DomainAge(domain=domain, outcome=AgeOutcome.UNAVAILABLE)
        return _registration_date(domain, payload)


def _registration_date(domain: str, payload: dict) -> DomainAge:
    """自 RDAP 回應取註冊日期。取不到時 `NO_DATA`，**不以其他事件推算**。

    `last changed` MUST NOT 被拿來代替：它是網域資料最後一次異動的時間，
    一個 1997 年註冊的網域上週改了 DNS 就會顯示上週。
    `last update of RDAP database` 更不行 —— 它是**註冊局資料庫**的更新時間，
    每次查都會變（實測 `esunbank.com.tw` 與 `example.com` 的該欄位都是
    「查詢當天」），拿它推算年齡會讓每一個網域看起來都是今天註冊的。

    **只讀 `events`，不讀 `entities`。** 註冊人、聯絡方式、註冊商都在
    `entities` 裡，那是第三方個資；取得它們不會增加成本，但存放它們會。
    """
    events = payload.get("events")
    if not isinstance(events, list):
        return DomainAge(domain=domain, outcome=AgeOutcome.NO_DATA)
    for event in events:
        if not isinstance(event, dict) or event.get("eventAction") != REGISTRATION_EVENT:
            continue
        raw = event.get("eventDate")
        if not isinstance(raw, str):
            return DomainAge(domain=domain, outcome=AgeOutcome.NO_DATA)
        parsed = _parse_event_date(raw)
        if parsed is None:
            return DomainAge(domain=domain, outcome=AgeOutcome.NO_DATA)
        return DomainAge(domain=domain, outcome=AgeOutcome.KNOWN, registered_on=parsed)
    return DomainAge(domain=domain, outcome=AgeOutcome.NO_DATA)


def _parse_event_date(raw: str) -> date | None:
    """把 RDAP 的 `eventDate` 解析為 UTC 日期。不合法時回傳 `None`。

    **不使用當下時間或任何預設值** —— 解析不出來就是不知道，而「不知道」
    在上層有一個專門的表達方式（`NO_DATA`）。

    RFC 3339 要求帶偏移量，但實測有註冊局送出不帶偏移量的字串；那種情況
    直接取其日期部分，誤差上限一天，不猜它是哪個時區。
    """
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.date()
    return moment.astimezone(timezone.utc).date()
