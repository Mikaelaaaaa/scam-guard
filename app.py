"""Gradio 介面 —— 偵測核心的第一個看得見的消費者。

本檔是 Gradio 相關程式碼的**唯一**容身處，位置由兩個既有約束指定：
`pyproject.toml` 的 `banned-api` 對 `gradio` 的訊息明寫「Gradio 相關程式碼請放在
`app.py`」、`per-file-ignores` 只豁免 `"app.py"`；HuggingFace Spaces 亦固定執行
repo 根目錄的 `app.py`。

三層結構，由內而外：

    scam_guard.detect()      偵測核心（本檔不修改它一行）
        ↓
    demo_ui                  標記層（純 Python，無 gradio，兩份介面層共用）
        ↓
    ├─ 模式二「這是詐騙嗎」：直接顯示，MUST NOT 經過模型
    └─ 模式一「詐騙對練」：可選的措辭潤飾層，受三條前綴可判定的條件約束

交棒事項：文案與標記現在的中繼站是 `demo_ui.py`（`docs/pages_app.py` 也 import
它）。終點不變 —— `add-verdict-render` 在 `scam_guard/` 內持有文案層，
`Verdict.evidence` 與 `Verdict.actions` 本來就是它產的；`demo_ui` 留下的是標記，
那一層不該進核心。

更新順序是 requirement 而非實作細節：判定卡與排行 MUST 在潤飾層產生任何字元之前
完成更新，受害方的回應 MUST 以 streaming 呈現。因此送出的 handler 是 generator
function，第一次 `yield` 已含完整的判定卡。
"""

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr

import demo_ui
from scam_guard.check import CheckRegistry
from scam_guard.normalize import DEFAULT_LIMITS, Limits, build_document
from scam_guard.pii import find_pii
from scam_guard.pipeline import detect
from scam_guard.redact import RedactedText
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.types import Message, Request, Verdict
from scam_guard.weights import load_weights

# ---------------------------------------------------------------------------
# 組裝層：註冊、上限、可選依賴的注入點
# ---------------------------------------------------------------------------

LIMITS: Limits = DEFAULT_LIMITS
"""上限的單一來源。**同一個變數**同時傳給 `detect()` 與 `build_document()`。

兩處用不同的 `Limits` 會產生不同的丟棄則數，於是同一個座標 `(3, 0)` 在兩邊
指向不同的句子 —— `detect()` 的 docstring 為此警告過「座標系必須有唯一的產生者」。
"""

SENDER_THEM = "them"
"""模式一的發送者標示。`project.md` 的輸入契約範例寫的就是 `"from": "them"`。

這是**偵測語義**（被檢查的那一方），與版面的左右無關 —— 見 `demo_ui.CSS` 裡
氣泡對齊那一段。
"""

SAMPLES_PATH = Path(__file__).with_name("demo_samples.json")

UNREGISTERED_CHECKS: tuple[demo_ui.UnregisteredCheck, ...] = (
    ("domain_age", "網域年齡查詢", "需要向網域註冊局查詢，本服務不對外連線"),
)
"""組裝層**明確知道其存在、但選擇不註冊**的檢查：識別字、中文名與一行理由。

只收錄已實作且有 spec 定義「未註冊」狀態的檢查。尚未實作的檢查不列入 ——
把它們寫進清單等於臆測未來的名稱，而名稱一旦不符，畫面上會永遠掛著一行
指向不存在的東西的「未註冊」。

清單本身要留著：少一個訊號要在畫面上看得見，否則「沒有訊號」與「沒有資料」
長得一模一樣。理由縮成一行 —— 使用者關心的是「它有沒有把我的網址送出去」，
不是共用出口 IP 的論證，那段論證在 `README.md`。
"""

PiiRecognizer = demo_ui.PiiRecognizer
PiiSpan = demo_ui.PiiSpan
Polisher = Callable[[Sequence[str]], Iterator[str]]
TranscriptLogger = Callable[[RedactedText], None]
"""記錄器的簽章。**參數型別就是許可** —— `RedactedText` 是系統裡唯一可以寫進
log 的東西（`scam_guard/redact.py`：「外層拿得到這個型別的實例，就等於拿到許可」），
於是「把原文寫進 log」在型別上不可表達，不必靠一句叮嚀。"""


def recognize_pii(sentence: str) -> list[demo_ui.PiiSpan]:
    """把 `scam_guard.pii` 的輸出轉成標記層要的 `(start, end, type)`。

    標記層只認 tuple，不認 `scam_guard.pii.PiiSpan` —— 它同樣接受
    `docs/pages_app.py` 那側掛上的辨識器，而那個注入點的契約
    （`add-pii-recognizers`）本來就是三元組。

    **與 `docs/pages_app.py` 的同名函式逐字相同，而且刻意不搬進 `demo_ui`。**
    搬進去會讓標記層直接相依 `scam_guard.pii`，於是「有沒有開啟個資標註」
    從**部署的選擇**變成**標記層的預設值**，而下面那個 `None` 的狀態就再也
    表達不出來了。兩行重複換的是一個不可能被誤設成「總是開啟」的結構。
    """
    return [(span.start, span.end, span.entity_type) for span in find_pii(sentence)]


PII_RECOGNIZER: PiiRecognizer | None = recognize_pii
"""個資辨識器。四條辨識器是純標準庫、就在同一個 wheel 裡，所以這一側掛得上 ——
`docs/pages_app.py` 早就掛著同一個 `find_pii`，兩份介面層對同一份能力
不該給出兩種答案。

未掛載（`None`）時畫面顯示「沒有開啟」，MUST NOT 顯示「沒有找到」：
前者是沒有人在看，後者是看過了沒有。"""

POLISHER: Polisher | None = None
"""模式一的措辭潤飾層。輸入只有受害方那一句，**簽章中沒有訊息原文**。"""

TRANSCRIPT_LOGGER: TranscriptLogger | None = None
"""**兩個模式共用**的記錄器。以掛載與否表達而非布林開關 —— 一個預設為 False
的布林是一個可以被貼進設定檔、看起來像是有人想過的值；而「沒有掛上」是不可能
被誤解的狀態。**公開部署預設不掛。** 掛上時 UI 才顯示「本模式的輸入會被記錄」，
且隱私說明跟著改口（見 `privacy_note()`）。"""


def log_transcript(verdict: Verdict) -> None:
    """兩個模式的**唯一**記錄點。未掛記錄器時什麼都不做。

    寫進去的是 `Verdict.redacted`，而且只有它。記錄發生在 `detect()` **之後**，
    副作用是被 `Limits` 丟棄的最舊訊息不進 log —— 那是正確的：
    沒有遮蔽投影的文字沒有合法的記錄形式。

    ⚠️ 遮蔽的保證範圍是四個辨識類型，**不等於「不含個資」**：姓名、地址、
    銀行帳號、護照號碼仍然原樣留在裡面。文案 MUST NOT 把它說成已去識別化。
    """
    if TRANSCRIPT_LOGGER is not None:
        TRANSCRIPT_LOGGER(verdict.redacted)


def build_registry() -> CheckRegistry:
    """建立本介面唯一的檢查註冊表。每落地一項檢查，此處多一行。

    **URL 層的五個檢查今日不註冊**：`register_url_checks()` 需要一份
    `PublicSuffixList`，由 `tools/fetch_psl.py` 落到被版控排除的 `data/`，
    而 HF Spaces 上 clone 出來的 repo 裡沒有那個目錄。權重表已經落地
    （`weights.toml` 進了版控），所以擋住的只剩 PSL 快照這一項 ——
    部署環境備妥該檔之後，此處多兩行。
    """
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    return registry


REGISTRY: CheckRegistry = build_registry()

TABLE = load_weights()
"""權重表的單一來源。`detect()` 的必填參數 —— 一個有預設表的 `detect()`
會讓呼叫端在沒有表的情況下跑出一個看起來正常的結果。

模組層載入而非每次請求載入：`load_weights()` 會讀檔並驗證整張表，
放進請求路徑等於每則訊息都重讀一次 TOML。它同時在 import 時就驗證
`REGISTRY` 的每個檢查都登記在表中 —— 缺漏會在啟動時炸，不是在第一次命中時。
"""


# ---------------------------------------------------------------------------
# 範例庫
# ---------------------------------------------------------------------------

SHORT_LABEL_MAX = 6
"""`short_label` 的長度上限。標籤是一顆按鈕上的字，八顆要能換行排下。"""


@dataclass(frozen=True)
class Sample:
    """一則進版控的範例訊息。`source_uri` 為 CC BY-SA 4.0 的姓名標示要求。

    `short_label` 是畫面上那顆按鈕的字，**不從 `label` 推導**：八筆的 `label`
    是「類型：說明」的形式，在 `：` 切開會得到兩筆都叫「假借補助金」的標籤，
    而兩個一模一樣的按鈕比一個長按鈕更糟。
    """

    label: str
    short_label: str
    text: str
    source_uri: str


def load_samples(path: Path = SAMPLES_PATH) -> list[Sample]:
    """載入範例庫。

    **缺檔即 `raise`，不降級。** 這個檔案進版控，缺少它代表安裝壞了，不是一個
    正常狀態。不讀 `data/`、不內建預設清單、不回傳空清單 —— 「有 `data/` 就讀
    `data/`、沒有就用內嵌」會讓 HF Spaces 永遠走其中一條、本機永遠走另一條，
    於是兩條路徑的差異不會被任何人發現。

    `short_label` 的三個條件（存在、不超過上限、跨全部範例唯一）皆於此處驗證並
    指名該筆：重複的短標籤會產生兩顆一模一樣的按鈕，而那是一個沒有任何地方會
    報告的錯。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"範例庫檔案不存在：{path.name}（預期位於 repo 根目錄 {path}）。"
            f"此檔進版控，缺少它代表安裝不完整。"
        )
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if "samples" not in loaded:
        raise ValueError(f"範例庫檔案缺少 samples 欄位：{path.name}")
    samples: list[Sample] = []
    seen: set[str] = set()
    for entry in loaded["samples"]:
        for field_name in ("label", "short_label", "text", "source_uri"):
            if field_name not in entry:
                raise ValueError(
                    f"範例庫的一筆資料缺少 {field_name} 欄位：{path.name}，該筆為 {entry!r}"
                )
        short_label = entry["short_label"]
        if len(short_label) > SHORT_LABEL_MAX:
            raise ValueError(
                f"範例庫的 short_label 超過 {SHORT_LABEL_MAX} 個字：{path.name}，"
                f"該值為 {short_label!r}（{len(short_label)} 個字）"
            )
        if short_label in seen:
            raise ValueError(f"範例庫的 short_label 重複：{path.name}，該值為 {short_label!r}")
        seen.add(short_label)
        samples.append(
            Sample(
                label=entry["label"],
                short_label=short_label,
                text=entry["text"],
                source_uri=entry["source_uri"],
            )
        )
    return samples


SAMPLES: list[Sample] = load_samples()


def sample_text(short_label: str) -> str:
    """以短標籤取範例的內文。未知的短標籤 `raise` 並指名該值。

    不回傳空字串、不退回第一筆 —— 標籤與範例庫出自同一個 `SAMPLES`，對不上
    代表其中一邊被改壞了。
    """
    for sample in SAMPLES:
        if sample.short_label == short_label:
            return sample.text
    raise ValueError(f"範例庫中沒有這個短標籤：{short_label!r}")


# ---------------------------------------------------------------------------
# 兩個模式各自的 Request 組法
# ---------------------------------------------------------------------------

BLANK_LINE = re.compile(r"\n[^\S\n]*\n")


def practice_messages(previous: Sequence[Message], text: str) -> list[Message]:
    """模式一：把新的一則接到對話後面。

    `sender` 全部填 `"them"`（人扮演詐騙方）；`sent_at` 填**實際送出時間**，
    不填 `None` —— `None` 的語意是「不知道」，而模式一知道。為了讓下游好看而
    謊報「不知道」，與「禁止臆測性 fallback」是同一條原則的反面。

    受害方的台詞 MUST NOT 出現在回傳值中：它是 `Verdict` 的呈現，回灌之後
    下一輪 `detect()` 會讀到自己上一輪的輸出並對它產生訊號。
    """
    return [*previous, Message(text=text, sender=SENDER_THEM, sent_at=datetime.now(tz=UTC))]


def build_inquiry_request(text: str) -> Request:
    """模式二：以**空行**切分為多則。單則即 `Request.from_text()`。

    不解析 `2026/09/14 10:00 小明` 這種 LINE 轉傳的時間戳行 —— 那個格式隨
    LINE 版本與語言設定改變，猜錯的後果是把一則訊息切成六則，每一則只有半句話，
    座標系跟著錯。解析屬 `add-line-adapter`。

    `sender` 與 `sent_at` 一律為 `None`，因為系統真的不知道。
    """
    chunks = [chunk.strip() for chunk in BLANK_LINE.split(text)]
    kept = [chunk for chunk in chunks if chunk]
    if not kept:
        raise gr.Error("輸入為空：請貼上收到的訊息內容")
    if len(kept) == 1:
        return Request.from_text(kept[0])
    return Request(messages=[Message(text=chunk) for chunk in kept])


# ---------------------------------------------------------------------------
# Event handlers —— 全部定義於模組層，狀態以顯式參數傳入與傳出
# ---------------------------------------------------------------------------

PRACTICE_SENDER = "你（扮演詐騙方）"
PRACTICE_REPLY = "對方"
INQUIRY_SENDER = "你貼上的訊息"


def practice_submit(
    text: str,
    messages: list[Message],
    replies: list[str],
    spoken: list[str],
) -> Iterator[tuple[list[Message], list[str], list[str], str, str, str, str, str]]:
    """模式一的送出處理，**generator function**。

    第一次 `yield` 帶完整的判定卡、排行與受害方那一句；其後每次 `yield` 追加
    潤飾層的新片段。判定卡 MUST 在模型產生任何字元之前完成更新 —— 若實作成
    「等模型跑完再一起更新」，畫面仍然正確，只是慢，而**沒有任何測試會報告
    展示效果消失**。因此測試驗證的是第一次產出的內容，不是最終畫面。

    `spoken` 是跨輪的狀態，與 `messages` / `replies` 同一個機制：它記住哪幾條
    依據已經被說過，使受害方每輪至多說一條**新**的。
    """
    if not text.strip():
        raise gr.Error("輸入為空：請輸入一則詐騙方會說的話")

    updated = practice_messages(messages, text)
    request = Request(messages=updated)
    verdict = detect(request, REGISTRY, TABLE, limits=LIMITS)
    document = build_document(request.messages, LIMITS)
    log_transcript(verdict)

    baseline, updated_spoken = demo_ui.victim_reply(verdict, spoken)
    updated_replies = [*replies, baseline]
    card = demo_ui.render_verdict_card(
        verdict, document, TABLE, UNREGISTERED_CHECKS, PII_RECOGNIZER
    )
    ranking = demo_ui.render_ranking(verdict.checks, len(updated))
    conversation = demo_ui.render_conversation(
        updated, updated_replies, document, PRACTICE_SENDER, PRACTICE_REPLY, PII_RECOGNIZER
    )
    status = demo_ui.POLISH_NOT_INJECTED if POLISHER is None else demo_ui.POLISH_STREAMING
    yield updated, updated_replies, updated_spoken, conversation, card, ranking, status, ""

    if POLISHER is None:
        return

    validator = demo_ui.PolishValidator([baseline], verdict)
    for chunk in POLISHER([baseline]):
        if not validator.feed(chunk):
            updated_replies[-1] = baseline
            yield (
                updated,
                updated_replies,
                updated_spoken,
                demo_ui.render_conversation(
                    updated,
                    updated_replies,
                    document,
                    PRACTICE_SENDER,
                    PRACTICE_REPLY,
                    PII_RECOGNIZER,
                ),
                card,
                ranking,
                demo_ui.POLISH_DISCARDED,
                "",
            )
            return
        updated_replies[-1] = validator.text
        yield (
            updated,
            updated_replies,
            updated_spoken,
            demo_ui.render_conversation(
                updated, updated_replies, document, PRACTICE_SENDER, PRACTICE_REPLY, PII_RECOGNIZER
            ),
            card,
            ranking,
            demo_ui.POLISH_STREAMING,
            "",
        )
    yield (
        updated,
        updated_replies,
        updated_spoken,
        demo_ui.render_conversation(
            updated, updated_replies, document, PRACTICE_SENDER, PRACTICE_REPLY, PII_RECOGNIZER
        ),
        card,
        ranking,
        demo_ui.POLISH_ACCEPTED,
        "",
    )


def practice_from_sample(
    short_label: str,
    messages: list[Message],
    replies: list[str],
    spoken: list[str],
) -> Iterator[tuple[list[Message], list[str], list[str], str, str, str, str, str]]:
    """點一顆範例標籤就送出那一則，一步完成。

    標籤的身分走**資料**進 handler（`gr.State` 常數），不走捕獲：
    以 lambda 或巢狀 `def` 捕獲迴圈變數，八顆按鈕會全部送出最後一筆，
    而那個錯在每顆按鈕上看起來都「有反應」。
    """
    yield from practice_submit(sample_text(short_label), messages, replies, spoken)


def inquiry_submit(text: str) -> tuple[str, str]:
    """模式二的送出處理。

    **不呼叫潤飾層，即使已注入。** 這是產品路徑，使用者在問一個關於自己安危的
    問題，模型在這條路徑上連措辭都不經手 —— 這是「LLM 不能有最終話語權」最直接
    的實作。而且模式二未來要給 LINE 用，那裡沒有串流可以展示速度差。

    **記錄與模式一走同一個點**（`log_transcript()`），寫入的只有
    `Verdict.redacted` 的 `sentences` / `coords` / `counts`，
    MUST NOT 寫入 `Request`、`Message.text` 或 `Verdict.evidence`。
    公開部署沒有掛記錄器，那時這一行什麼都不做。
    """
    request = build_inquiry_request(text)
    verdict = detect(request, REGISTRY, TABLE, limits=LIMITS)
    document = build_document(request.messages, LIMITS)
    log_transcript(verdict)
    return (
        demo_ui.render_verdict_card(verdict, document, TABLE, UNREGISTERED_CHECKS, PII_RECOGNIZER),
        demo_ui.render_conversation(
            request.messages, (), document, INQUIRY_SENDER, PRACTICE_REPLY, PII_RECOGNIZER
        ),
    )


def inquiry_from_sample(short_label: str) -> tuple[str, str, str]:
    """點一顆範例標籤就填入並判定，一步完成。標籤的身分以參數傳入，不捕獲。"""
    text = sample_text(short_label)
    card, echo = inquiry_submit(text)
    return text, card, echo


# ---------------------------------------------------------------------------
# 介面組裝
# ---------------------------------------------------------------------------

HEADER = """
<div class="sg-head">
<h1>這是詐騙嗎</h1>
<p>貼上你收到的可疑訊息，看看它像不像詐騙、依據是什麼，以及你現在可以做什麼。</p>
</div>
"""

PRACTICE_NOTE = (
    '<div class="note">你扮演詐騙方打字，對方由系統扮演。'
    "對方每一輪只會講一條新看到的訊號，<b>聊不起來是正常的</b>。</div>"
)

PRIVACY_NOT_LOGGED = (
    "<p><b>不會被儲存。</b>這個版本兩個模式都不留任何記錄；對話內容只存在於這個"
    "瀏覽器分頁，關掉或重新整理就消失。</p>"
)

PRIVACY_LOGGED = (
    "<p><b>你貼上的內容會被寫進記錄，兩個模式都一樣。</b>寫進去之前會先蓋掉"
    "身分證字號、手機號碼、市話與信用卡號四種。<b>姓名、地址與銀行帳號不在這四種"
    "裡面</b>，會原樣留在記錄裡。不要在這裡貼上你不想被留下來的東西。</p>"
)

PRIVACY_REST = """
<p><b>不會送到外部服務。</b>比對用的名單與規則都在本機，判斷過程不對外連線。</p>
<p>伺服器的連線記錄只有網址與狀態碼，不含你打的字。若這個服務是跑在別人的雲端平台上，
該平台自己的系統記錄不在我們控制範圍內 —— 未被接住的錯誤訊息可能落在那裡。</p>
"""


def privacy_note(logger: TranscriptLogger | None) -> str:
    """隱私說明。**依記錄器掛了沒有產生，不是一段無條件的固定文字。**

    原本那段常數無條件宣告「『這是詐騙嗎』模式不留任何記錄」。兩個模式共用記錄點
    之後，那句話在有掛記錄器的部署上是假的，而一個會說謊的隱私說明比沒有隱私說明
    更糟 —— 讀它的人正是因為在意才點開它。

    掛上時的文字明講會被寫進記錄，並指出遮蔽只涵蓋四個類型；
    MUST NOT 說成「已去識別化」。
    """
    first = PRIVACY_NOT_LOGGED if logger is None else PRIVACY_LOGGED
    return (
        "<details><summary>你的訊息會被怎麼處理</summary>"
        f'<div class="note">{first}{PRIVACY_REST}</div></details>'
    )


def samples_listing() -> str:
    """範例庫的出處清單。CC BY-SA 4.0 的姓名標示要求以每筆的連結滿足。

    收在頁尾的可展開區塊：授權標示是法律要求，必須留著，但它不是使用者來這裡
    要解決的問題，不該佔掉第一眼的版面。
    """
    items = "".join(
        f"<li>{demo_ui.escaped(sample.label)} —— "
        f'<a href="{demo_ui.escaped(sample.source_uri)}" target="_blank" rel="noopener">出處</a>'
        "</li>"
        for sample in SAMPLES
    )
    return (
        "<details><summary>範例訊息的來源與授權（Cofacts，CC BY-SA 4.0）</summary>"
        '<div class="note"><p>範例訊息取自 '
        '<a href="https://cofacts.tw" target="_blank" rel="noopener">Cofacts 真的假的</a>'
        " 的開放資料，依 CC BY-SA 4.0 釋出，與本專案其餘部分的 MIT 授權不同；"
        f"散布它或其衍生內容時須以相同條款釋出。</p><ul>{items}</ul></div></details>"
    )


def transcript_notice(logger: TranscriptLogger | None) -> str:
    """記錄器已掛上時的顯著標示，**兩個模式都掛**。沒掛時不顯示。

    沒掛時回空字串而不是一句「本模式不會記錄」：那兩個狀態的差別要在畫面上
    看得見，而「什麼都沒說」與「說了會記錄」已經是兩個不同的畫面。

    ⚠️ 文案 MUST NOT 說成「已去識別化」：寫進記錄之前只蓋掉四個辨識類型，
    姓名與地址原樣留著。
    """
    if logger is None:
        return ""
    return (
        '<div class="notice">本模式的輸入會被記錄：'
        "你在這裡打的每一則訊息都會被寫進伺服器的記錄檔，寫進去之前會先蓋掉"
        "身分證字號、手機號碼、市話與信用卡號四種。"
        "<b>姓名、地址與銀行帳號不在這四種裡面</b>，會原樣留在記錄裡。"
        "不要在這裡貼上真實的個人資料。</div>"
    )


def build_demo() -> gr.Blocks:
    """組裝介面。

    `gr.Blocks` 而非 `gr.Interface`：flagging（使用者按下 flag 會把原文連同輸出
    寫進 `.gradio/flagged/dataset.csv`，一個落地的檔案）只存在於 `gr.Interface`，
    Blocks 沒有這條路徑。這是顯式的選擇而不是預設值，連同 `analytics_enabled=False`
    一起構成「框架內建的提交功能已關閉」。

    兩個模式是兩個 `gr.Tab`，各自持有自己的 `gr.State` 與輸出元件 ——
    狀態隔離因此是結構上的，不靠任何清空邏輯維持。

    **不嘗試鎖定明暗模式。** `gr.Blocks` 沒有 theme 參數，`launch(theme=)` 接的是
    調色盤不是明暗模式，而使用者可以在頁尾的設定面板自己切，設定 persist 在
    瀏覽器。唯一的解法是不寫死顏色 —— 見 `demo_ui.CSS`。
    """
    with gr.Blocks(title="這是詐騙嗎 · scam-guard", analytics_enabled=False) as demo:
        gr.HTML(HEADER)

        with gr.Tabs():
            with gr.Tab("這是詐騙嗎"):
                gr.HTML(transcript_notice(TRANSCRIPT_LOGGER))
                inquiry_input = gr.Textbox(
                    label="貼上你收到的訊息",
                    lines=5,
                    placeholder="把整則訊息貼進來。若是一段轉傳的對話，請在每則之間空一行。",
                )
                with gr.Row(elem_classes="chips"):
                    inquiry_chips = [
                        gr.Button(sample.short_label, size="sm", scale=0) for sample in SAMPLES
                    ]
                inquiry_send = gr.Button("看看這是不是詐騙", variant="primary")
                inquiry_card = gr.HTML()
                inquiry_echo = gr.HTML()

            with gr.Tab("詐騙對練"):
                gr.HTML(PRACTICE_NOTE)
                gr.HTML(transcript_notice(TRANSCRIPT_LOGGER))
                practice_card = gr.HTML()
                practice_ranking = gr.HTML()
                practice_conversation = gr.HTML()
                practice_input = gr.Textbox(
                    label="你（扮演詐騙方）",
                    lines=2,
                    placeholder="打一句詐騙方會說的話",
                )
                with gr.Row(elem_classes="chips"):
                    practice_chips = [
                        gr.Button(sample.short_label, size="sm", scale=0) for sample in SAMPLES
                    ]
                practice_send = gr.Button("送出", variant="primary")
                practice_status = gr.HTML(f'<div class="note">{demo_ui.POLISH_NOT_INJECTED}</div>')
                practice_messages_state = gr.State([])
                practice_replies_state = gr.State([])
                practice_spoken_state = gr.State([])

        gr.HTML(f'<div class="sg-foot">{privacy_note(TRANSCRIPT_LOGGER)}{samples_listing()}</div>')

        practice_inputs = [practice_messages_state, practice_replies_state, practice_spoken_state]
        practice_outputs = [
            practice_messages_state,
            practice_replies_state,
            practice_spoken_state,
            practice_conversation,
            practice_card,
            practice_ranking,
            practice_status,
            practice_input,
        ]
        inquiry_outputs = [inquiry_card, inquiry_echo]

        inquiry_send.click(inquiry_submit, inputs=[inquiry_input], outputs=inquiry_outputs)
        practice_send.click(
            practice_submit, inputs=[practice_input, *practice_inputs], outputs=practice_outputs
        )
        # 標籤的身分以一個常數 `gr.State` 進 handler。迴圈裡不定義任何函式 ——
        # 捕獲迴圈變數的寫法會讓八顆按鈕全部送出最後一筆。
        for sample, chip in zip(SAMPLES, inquiry_chips):
            chip.click(
                inquiry_from_sample,
                inputs=[gr.State(sample.short_label)],
                outputs=[inquiry_input, *inquiry_outputs],
            )
        for sample, chip in zip(SAMPLES, practice_chips):
            chip.click(
                practice_from_sample,
                inputs=[gr.State(sample.short_label), *practice_inputs],
                outputs=practice_outputs,
            )

    return demo


if __name__ == "__main__":
    # Gradio 6 把 `css` 從 Blocks 的建構子移到 `launch()`。
    build_demo().launch(css=demo_ui.CSS)
