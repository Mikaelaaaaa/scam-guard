"""GitHub Pages 靜態 demo 的 Python 側 —— 在瀏覽器（Pyodide）裡執行。

**為什麼這裡有第二個介面層，而不是直接跑 `app.py`。**
原訂的載體是 Gradio-Lite（`@gradio/lite`，Pyodide 上的 Gradio），那條路今天走不通：
該套件在 npm 上最後一次發佈是 2025-09-10（同一天其餘 `@gradio/*` 都還在更新），
它內建的 Gradio 5.45.0 對上今天的 PyPI 會在啟動階段連續踩到三個相依性漂移的錯
（`huggingface-hub` 版本衝突、`filelock` 新版的 `os.link(follow_symlinks=...)`
探測撞上 Gradio-Lite 自己的 `os.link` mock、`anyio` / `typing-inspection` 找不到
純 Python wheel）。上游官方的 hello world 範例在瀏覽器裡同樣起不來，所以那不是
本專案的問題，也不是版本挑錯了 —— 那條路目前是壞的。

於是本頁不經過 Gradio：Pyodide 直接安裝 `scam_guard` 的 wheel（它的執行期依賴
是 `[]`，micropip 不需要向 PyPI 解析任何東西），介面由 `index.html` 與本檔組成。

**與 `app.py` 的關係：兩者是兩個載體，但畫面上的標記只有一份。**
判定卡、偵測細節、氣泡、逸出、狀態分類與樣式表全部來自 `demo_ui`，本檔不自己
組裝任何一塊。這是先前記在這裡的那筆債的還款 —— 兩份標記各改一次，第二次就是
它的利息。`demo_ui` 不 import gradio，那正是它能在這一側被使用的前提。

**兩個模式都在這裡了，而那是因為模型層來了。** 這裡原本寫著「詐騙對練不搬上來，
因為模型層在瀏覽器裡根本不存在（`llama-cpp-python` 沒有 Pyodide 版本）」——
那句話對 llama.cpp 仍然成立，但模型層現在走的是 transformers.js + ONNX Runtime
Web，而它在瀏覽器裡跑得起來。對練模式的展示重點（判定卡早於受害方台詞完成更新，
也就是規則層與模型層的速度差）因此有了對照組，累積命中排行也跟著搬上來。

**本檔只做組裝。** 兩趟 `detect()`、重放式 runtime 與一輪的狀態機在
`browser_llm.py`，標記在 `demo_ui.py`，本檔提供的是註冊表、權重表、上限、
未註冊清單與兩個模式的標籤 —— 以及那幾個 JavaScript 真的會呼叫的函式。

**線上版與本機版仍然存在的差異，逐條列在這裡。**
`url_blocklist` 的缺席已經補上（黑名單與白名單兩份快照由建置腳本打包），
剩下的五條是已知且刻意留下的，而它們留在這裡是因為一份寫在 design 裡的清單
沒有人會在改動這個檔案時讀到：

1. **LLM 措辭潤飾層的 llama.cpp 那一半在瀏覽器裡不存在**（`llama-cpp-python`
   沒有 Pyodide 版本）。瀏覽器側走的是 transformers.js + ONNX Runtime Web，
   兩者是不同的推論引擎，同一個 prompt 的輸出不保證逐字相同。
2. **`domain_age` 兩邊都不註冊**，理由相同：本頁不對外連線。它在
   `UNREGISTERED_CHECKS` 裡看得見。
3. **線上與本機的 Tranco `list_id` 必然不同**（Tranco 每日換一份清單），
   所以白名單降級的依據文案會寫出不同的清單 ID 與名次。這不是行為差異，
   是可複現性的差異 —— 兩邊各自的 `list_id` 都記在自己的 manifest 裡。
4. **線上快照最多比本機新抓的舊 7 天**：`.github/workflows/pages.yml` 的排程
   是每週一次，而本機開發者是當下執行取得程式。
5. **`add-phishtank-feed` 的兩個國際釣魚 feed 兩邊都不在。** 線上這一側是由
   `require_redistributable=True` 保證的，不是由設定保證的 —— 那兩個 feed 的
   授權不允許對第三方顯示，`BlocklistStore.load()` 會在載入時擋下它們。
"""

import json
import re

import browser_llm
import demo_ui
from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.llm.check import TIMEOUT_SENTINEL
from scam_guard.normalize import DEFAULT_LIMITS
from scam_guard.pii import find_pii
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import Message, Request
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import load_tables, register_url_checks
from scam_guard.weights import load_weights

LIMITS = DEFAULT_LIMITS
"""上限的單一來源。**同一個變數**同時傳給 `detect()` 與 `build_document()` ——
兩處用不同的 `Limits` 會產生不同的丟棄則數，於是同一個座標在兩邊指向不同的句子。"""

PSL_DIR = "/psl"
"""PSL 快照在 Pyodide 檔案系統中的位置，由 `index.html` 在啟動時寫入。"""

PSL_MAX_AGE_DAYS = 90
"""PSL 快照的新鮮度上限。`PublicSuffixList.load()` 的必填參數，沒有預設值。

靜態站台沒有「啟動」這件事：快照的取得時間就是**最後一次部署的時間**，所以這個
數字實際上是在說「這個站台超過 90 天沒有重新部署就不該再運作」。
`.github/workflows/pages.yml` 以每週一次的排程重新部署來滿足它。

超過上限時本檔在 import 階段就拋例外，頁面顯示那個例外的原文（含 `fetched_at`
與實際天數），**不會少一個訊號繼續跑**。一個安靜降級的頁面不會告訴任何人任何事。
"""

BLOCKLIST_DIR = "/blocklist"
"""165 涉詐網址黑名單快照在 Pyodide 檔案系統中的位置，由 `index.html` 在啟動時寫入。"""

ALLOWLIST_DIR = "/allowlist"
"""Tranco 排名白名單快照在 Pyodide 檔案系統中的位置，同樣由 `index.html` 寫入。"""

BLOCKLIST_MAX_AGE_DAYS = {"176455": 60, "165027": 60}
"""黑名單的新鮮度上限，逐 source。數字取自 `api/app.py` 的 `BLOCKLIST_MAX_AGE_DAYS`
——同一份資料在兩個部署上不該有兩個門檻。

160055 不在表中：它標記為 `retired`（資料集描述明文指出已被 176455 取代），
`_check_freshness` 會跳過它，而傳一個它不需要的門檻進去只會讓人以為它還會更新。

判準是資料內容的 `data_through`，不是下載時間 —— 160055 的詮釋資料在 2026-07-29
還更新過，而檔內最新的一筆停在 2025-12-31。

靜態站台沒有「啟動」這件事，所以這個數字實際上是在說「這個站台超過 60 天沒有
重新部署，這一項檢查就不該再運作」。而一份**當天建置**的快照本來就已經用掉
15 至 20 天的額度（176455 的粒度是月），實際餘裕約 40 至 45 天。
"""

ALLOWLIST_MAX_AGE_DAYS = 30
"""白名單的新鮮度上限。數字取自 `RankAllowlist.load()` 的 docstring：Tranco 明文
說明每日 0:00 UTC 更新，一份 30 天沒更新的白名單意味著 `tranco-list.eu` 停止服務了。

**它比黑名單的 60 天緊，所以線上這一項的實際壽命由它決定。** 白名單一到期，
下面的成對規則會把黑名單一起帶走。
"""

DOMAIN_AGE_UNREGISTERED: demo_ui.UnregisteredCheck = (
    "domain_age",
    "網域年齡查詢",
    "需要向網域註冊局查詢，本頁不對外連線",
)
"""唯一一筆與載入結果無關的未註冊紀錄：本頁不對外連線是這個載體的性質。"""


def load_snapshots(
    psl: PublicSuffixList, blocklist_dir: str, allowlist_dir: str
) -> tuple[BlocklistStore | None, RankAllowlist | None, str]:
    """載入黑名單與白名單兩份快照，回傳 `(store, allowlist, 未註冊理由)`。

    **兩份快照成對：任一載入失敗即兩份都不用**，回傳 `(None, None, 理由)`。
    理由來自 `add-tranco-allowlist` 的一筆實測偽陽性：160055 收錄了
    `play.google.com`，於是一則直接連向它的正常訊息會取得 `hard=True` 並短路掉
    整個 LLM 層。白名單修掉那一筆，而白名單**不是 `Check`** ——
    `UNREGISTERED_CHECKS` 是「已知存在但未註冊的檢查」清單，沒有它的位置。
    所以一個缺白名單的 `url_blocklist` 會帶著已知偽陽性上線，而畫面上無處申報。

    **只捕捉 `(FileNotFoundError, ValueError)`，且不改寫訊息。**
    這不是臆測性 fallback，是 `BlocklistStore.load()` 自己指定的降級方式
    （它的 docstring：「要降級的話，正確做法是組裝層捕捉這個例外後**不註冊**
    黑名單檢查」）。兩個型別各自明確：`FileNotFoundError` 是快照檔不存在、
    `ValueError` 是 sha256 不符／manifest 欄位缺漏／授權不允許／過期。
    四種 `ValueError` 的處置相同（這份快照不能用），而它們各自的訊息已經指名了
    是哪一種 —— 所以原文上畫面即可，**不以字串比對去分辨過期與損毀**。

    **MUST NOT 回傳一個空的 `BlocklistStore`。** 「這個網域不在名單上」與
    「這裡沒有名單」在 `Verdict.checks` 裡長得一模一樣，而前者是一次成功的推論、
    後者是一次缺席。

    `require_redistributable` 用預設的 `True`，**不顯式傳 `False`**：本站台的
    產出就是給第三方看的判斷依據，而 GitHub Pages 是這件事最純粹的形式。
    """
    try:
        store = BlocklistStore.load(blocklist_dir, psl, max_age_days=BLOCKLIST_MAX_AGE_DAYS)
    except (FileNotFoundError, ValueError) as error:
        return None, None, f"165 涉詐網址名單快照無法載入：{error}"
    try:
        allowlist = RankAllowlist.load(allowlist_dir, max_age_days=ALLOWLIST_MAX_AGE_DAYS)
    except (FileNotFoundError, ValueError) as error:
        return (
            None,
            None,
            f"Tranco 排名白名單快照無法載入，涉詐網址名單因此一併不啟用（"
            f"沒有白名單時它會重現一筆已知的硬證據偽陽性，而畫面上無處申報）：{error}",
        )
    return store, allowlist, ""


def unregistered_checks(blocklist_reason: str) -> tuple[demo_ui.UnregisteredCheck, ...]:
    """本部署**明確知道其存在、但選擇不註冊**的檢查：識別字、中文名與一行理由。

    只收錄已實作、且有規格定義「未註冊」狀態的檢查。少一個訊號要在畫面上看得見，
    否則「沒有訊號」與「沒有資料」在結果裡長得一模一樣。

    `url_blocklist` 那一筆**由當次載入的結果決定**，理由是該次例外的訊息原文，
    不是一段寫死的字串。這裡原本寫著「名單有十萬筆，在瀏覽器裡載入太重」——
    那個理由實測是假的（gzip 後 1,268,967 B、Pyodide 內常駐約 28 MB，而同一個
    瀏覽器的 wasm32 上限實測為 4 GiB），而一個寫著假理由的揭露比不揭露更糟。

    ⚠️ **這裡的內容是清單中與模型無關的部分，不是清單的全部。**
    語意判讀那一筆的有無取決於這一次判讀發生了什麼（模型沒載入？載入中？
    載入失敗？跑了但作廢？跑完而且成功？），所以它由 `browser_llm` 逐次接上去 ——
    畫面上的清單因此不再是一個靜態的 tuple。上面那句「少一個訊號要在畫面上看得見」
    正是它要動起來的理由：模型跑完且成功時，那一筆 MUST 消失。
    """
    if not blocklist_reason:
        return (DOMAIN_AGE_UNREGISTERED,)
    return (
        DOMAIN_AGE_UNREGISTERED,
        ("url_blocklist", "涉詐網址名單比對", blocklist_reason),
    )


SENDER_LABEL = "你貼上的訊息"
REPLY_LABEL = "對方"
PRACTICE_SENDER_LABEL = "你（扮演詐騙方）"
PRACTICE_REPLY_LABEL = "對方"

BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
"""多則轉傳的分隔符。**以空行分隔，不解析時間戳行與暱稱行** —— 那個格式隨
LINE 版本與語言設定改變，猜錯的後果是把一則訊息切成六則、每則半句話，座標系跟著錯。"""


def build_registry(
    psl: PublicSuffixList, store: BlocklistStore | None, allowlist: RankAllowlist | None
) -> CheckRegistry:
    """建立本介面唯一的檢查註冊表。每落地一項檢查，此處多一行。

    URL 層的檢查在這裡註冊得起來，是因為建置腳本把三份快照一起部署了。
    `store` 為 `None` 時 `register_url_checks()` 不註冊 `url_blocklist` ——
    那是該函式本來就定義好的行為（它的 `store` 與 `allowlist` 兩個選配參數
    正是為了這件事存在的注入點），不是這裡的特例處理。

    **`scam_guard/` 一行都不改。** 另建一個滿足相同介面的類別會複製
    `BlocklistStore` 的三條規則（只有標記 `domain_level_matching` 的 source 進
    網域層索引、可註冊網域在載入時以傳入的 PSL 現算、每筆的 `source` 必須登記於
    manifest），而複製得到程式碼、複製不到它們的理由。
    """
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    register_url_checks(registry, psl, load_tables(), store=store, allowlist=allowlist)
    return registry


PSL = PublicSuffixList.load(PSL_DIR, max_age_days=PSL_MAX_AGE_DAYS)
"""PSL 只有一份，黑名單的載入與 URL 檢查共用它。

黑名單的網域層索引在載入時以**這一個**實例現算可註冊網域，所以兩處用不同的
PSL 版本會讓索引的鍵與查詢的鍵對不起來，而沒有任何地方會報告這件事。

PSL 載入失敗時本檔在 import 階段就拋例外、整頁停止 —— 與黑名單不同，
因為 PSL 沒有「未註冊」這個狀態：少了它，URL 層的檢查全部一起消失，
而沒有任何規格描述那個頁面長什麼樣。
"""

STORE, ALLOWLIST, BLOCKLIST_UNREGISTERED_REASON = load_snapshots(PSL, BLOCKLIST_DIR, ALLOWLIST_DIR)

REGISTRY: CheckRegistry = build_registry(PSL, STORE, ALLOWLIST)

UNREGISTERED_CHECKS: tuple[demo_ui.UnregisteredCheck, ...] = unregistered_checks(
    BLOCKLIST_UNREGISTERED_REASON
)

TABLE = load_weights()
"""權重表的單一來源。`detect()` 的必填參數，且在 import 時就載入並驗證整張表 ——
缺漏會在頁面啟動時炸，不是在第一次命中時。"""


def recognize_pii(sentence: str) -> list[demo_ui.PiiSpan]:
    """把 `scam_guard.pii` 的輸出轉成標記層要的 `(start, end, type)`。

    標記層只認 tuple，不認 `scam_guard.pii.PiiSpan` —— 它同樣接受 `app.py` 那側
    注入的辨識器，而那個注入點的契約（`add-pii-recognizers`）本來就是三元組。
    """
    return [(span.start, span.end, span.entity_type) for span in find_pii(sentence)]


def build_request(text: str) -> Request:
    """把輸入框的內容組成 `Request`。以**空行**切分為多則，單則即單則。

    `sender` 與 `sent_at` 一律為 `None`，因為系統真的不知道。
    """
    kept = [chunk.strip() for chunk in BLANK_LINE.split(text) if chunk.strip()]
    if not kept:
        raise ValueError("輸入為空：請貼上你收到的訊息內容")
    if len(kept) == 1:
        return Request.from_text(kept[0])
    return Request(messages=[Message(text=chunk) for chunk in kept])


TWO_PASS = browser_llm.TwoPass(registry=REGISTRY, table=TABLE, limits=LIMITS)
"""兩趟流程的持有者。`REGISTRY` 只有 `Stage.LOCAL` 的檢查，
所以第一趟在結構上不可能等一個模型（`TwoPass.__init__` 擋住其餘）。"""

INQUIRY = browser_llm.InquiryMode(
    TWO_PASS,
    browser_llm.Presentation(
        table=TABLE,
        unregistered=UNREGISTERED_CHECKS,
        sender_label=SENDER_LABEL,
        reply_label=REPLY_LABEL,
        recognizer=recognize_pii,
    ),
)

PRACTICE = browser_llm.PracticeMode(
    TWO_PASS,
    browser_llm.Presentation(
        table=TABLE,
        unregistered=UNREGISTERED_CHECKS,
        sender_label=PRACTICE_SENDER_LABEL,
        reply_label=PRACTICE_REPLY_LABEL,
        recognizer=recognize_pii,
    ),
)
"""對練模式的跨輪狀態就是這個實例：`messages` / `replies` / `spoken` 都在它裡面。

兩個模式共用**同一個** `TwoPass`，所以也共用同一個一次性 token 的槽位 ——
切換模式會讓另一個模式等待中的那次判讀作廢，而那正是想要的：
一次只有一個判讀在飛。
"""


def _state(name: str) -> browser_llm.LlmState:
    """把 JavaScript 傳來的狀態字串變成 `LlmState`。不認得的值讓 `ValueError` 傳播。

    不給預設值：一個把未知狀態當成「沒有載入」的預設，會讓一個壞掉的呼叫端
    安靜地永遠拿到純規則版，而畫面上看起來一切正常。
    """
    return browser_llm.LlmState(name)


def _json(payload: dict[str, object]) -> str:
    """回傳 JSON 字串而不是一整塊 HTML：每一塊在畫面上各有自己的位置與捲動行為，
    由 `index.html` 決定；把版面塞進這裡會讓兩邊都要知道對方的結構。"""
    return json.dumps(payload, ensure_ascii=False)


def constants() -> str:
    """JavaScript 那一側要用、而唯一來源在 Python 這一側的值。

    助理回合的預填前綴由 `scam_guard.llm.schema.FIELD_NAMES` 產生 ——
    **JavaScript MUST NOT 自己寫欄位名**：手寫的欄位名與 schema 不同步時
    沒有任何機制會報告，而那一側連 lint 都看不到這個關聯。
    """
    return _json(
        {
            "prefill": browser_llm.ASSISTANT_PREFILL,
            "deadline_s": browser_llm.DEADLINE_S,
            "timeout_sentinel": TIMEOUT_SENTINEL,
        }
    )


def inquiry_first(text: str, llm_state: str) -> str:
    """模式二「這是詐騙嗎」的第一趟。判定卡在這裡就已經完整。"""
    return _json(INQUIRY.first(build_request(text), _state(llm_state)))


def inquiry_second(token: str, raw: str) -> str:
    """模式二的第二趟。**簽章裡沒有訊息文字** —— 它從來沒有離開過 Python 側。"""
    return _json(INQUIRY.second(token, raw))


def inquiry_without_model(token: str, llm_state: str) -> str:
    """模式二在沒有生成發生時收尾（模型未載入／載入中／載入失敗）。"""
    return _json(INQUIRY.without_model(token, _state(llm_state)))


def practice_first(text: str, llm_state: str) -> str:
    """模式一「詐騙對練」的第一趟：判定卡與排行更新，**還沒有受害方台詞**。"""
    return _json(PRACTICE.first(text, _state(llm_state)))


def practice_second(token: str, raw: str) -> str:
    """模式一的第二趟：判定卡與排行再更新，然後才產出受害方台詞。"""
    return _json(PRACTICE.second(token, raw))


def practice_without_model(token: str, llm_state: str) -> str:
    """模式一在沒有生成發生時收尾。台詞與潤飾驗證器仍然吃同一個 `Verdict`。"""
    return _json(PRACTICE.without_model(token, _state(llm_state)))


def practice_polish_feed(chunk: str) -> str:
    """餵入潤飾生成的一塊。`ok` 為 `false` 時 JavaScript MUST 立刻中止生成。"""
    return _json(PRACTICE.polish_feed(chunk))


def practice_polish_end() -> str:
    """潤飾生成正常結束。"""
    return _json(PRACTICE.polish_end())


def styles() -> str:
    """共用樣式表。頁面在就緒後把它寫進一個 `<style>`。

    樣式表只有一份，跟著標記走 —— `index.html` 自己只定義主題變數的調色盤，
    不重複定義任何由 `demo_ui` 產生的 class。
    """
    return demo_ui.CSS


def check_count() -> int:
    """已註冊的檢查數。頁面在就緒時顯示它，數字由註冊表產生而不是寫死在 HTML。"""
    return len(REGISTRY.enabled())
