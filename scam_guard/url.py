"""URL 抽取、主機正規化與可註冊網域 —— 純函式，無網路、無 DNS。

此模組唯一的檔案讀取是 `PublicSuffixList.load()`，且 MUST 由呼叫端顯式觸發。
import 本模組不讀取任何檔案，即使 PSL 快照不存在亦不失敗 ——
`scam_guard/` 現有的測試沒有一個需要 PSL，若 import 時就讀檔，
一個沒跑過 `tools/fetch_psl.py` 的環境連 `pytest` 都跑不起來，而 CI 正是這樣的環境。

**與 `normalize.URL_PATTERN` 的分工。** `normalize.py` 的 `URL_PATTERN`
回答的是「哪裡不可以切句」，允許寬鬆（`https://x.cc/a點擊領取` 整段不切，
後果是句子變長）。本模組回答的是「這段字是不是 URL、它的主機是什麼」，
必須嚴格 —— 同一段字若原樣送去比對黑名單，鍵會是一段中文。
兩者是兩個用途，兩個都要留，`URL_PATTERN` 不動。

**主機保證完整，query 不保證完整。** 切句分隔符（`。`、`!`、`?`、`;`、換行）
沒有一個是合法的主機字元，所以主機不可能被切開；但不帶 scheme 的寫法
不在切句遮罩的保護範圍內，其 `?` 之後的內容可能落在下一個句子。
`ExtractedUrl.url` 因此 MUST NOT 被當成「訊息中完整的 URL」。
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path

from scam_guard.normalize import Document
from scam_guard.types import Coord

PSL_FILENAME = "public_suffix_list.dat"
PSL_MANIFEST_FILENAME = "manifest.json"
PSL_FETCH_COMMAND = "python -m tools.fetch_psl"

ICANN_BEGIN = "===BEGIN ICANN DOMAINS==="
ICANN_END = "===END ICANN DOMAINS==="
PRIVATE_BEGIN = "===BEGIN PRIVATE DOMAINS==="
PRIVATE_END = "===END PRIVATE DOMAINS==="

PSL_MANIFEST_FIELDS = ("source_url", "fetched_at", "sha256", "rule_count", "icann_rule_count")


class Section(Enum):
    """PSL 的區段。ICANN 是 IANA 委派的後綴，PRIVATE 是託管平台自行登記的。"""

    ICANN = "icann"
    PRIVATE = "private"


class RuleKind(Enum):
    """PSL 的三種規則形式。"""

    NORMAL = "normal"
    WILDCARD = "wildcard"
    EXCEPTION = "exception"


class PublicSuffixList:
    """Public Suffix List 的解析與比對。

    自行實作而不加 `tldextract`：後者第一次使用時會上網抓 PSL，
    關掉它只靠一個建構子參數，而那個參數漏寫不會有任何 lint 或測試抓到，
    只會在部署後產生一次對外請求；抓不到時它又會靜默退回內嵌快照，
    把「這份 PSL 有多舊」藏在套件版本裡。PSL 的語法只有三條規則，
    解析加比對約三十行，且有官方公開的測試向量可以逐項對答案。
    """

    def __init__(self, rules: dict[str, tuple[RuleKind, Section]]) -> None:
        self._rules = rules
        self._icann_tlds = frozenset(
            suffix.rsplit(".", 1)[-1]
            for suffix, (_, section) in rules.items()
            if section is Section.ICANN
        )

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def icann_tlds(self) -> frozenset[str]:
        """ICANN 區段第一層標籤的集合，即 IANA 實際委派的全部 TLD。

        供不帶 scheme 的抽取判定「這個字串的最後一個標籤是不是真的 TLD」——
        抽取器因此不需要自己的 TLD 清單，與可註冊網域計算共用同一份資料，
        也共用同一個過期判定。PRIVATE 區段的第一層標籤不納入。
        """
        return self._icann_tlds

    @classmethod
    def parse(cls, text: str) -> "PublicSuffixList":
        """解析 PSL 的檔案內容。缺少 ICANN 區段標記或規則數為 0 時拋 `ValueError`。"""
        rules: dict[str, tuple[RuleKind, Section]] = {}
        section: Section | None = None
        seen_icann = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                if ICANN_BEGIN in stripped:
                    section = Section.ICANN
                    seen_icann = True
                elif PRIVATE_BEGIN in stripped:
                    section = Section.PRIVATE
                elif ICANN_END in stripped or PRIVATE_END in stripped:
                    section = None
                continue
            if not stripped or section is None:
                continue
            if stripped.startswith("!"):
                kind = RuleKind.EXCEPTION
                body = stripped[1:]
            elif stripped.startswith("*."):
                kind = RuleKind.WILDCARD
                body = stripped
            else:
                kind = RuleKind.NORMAL
                body = stripped
            rules[_encode_suffix(body)] = (kind, section)
        if not seen_icann:
            raise ValueError(f"PSL 檔案缺少 {ICANN_BEGIN} 標記，無法判定規則所屬區段")
        if not rules:
            raise ValueError("PSL 檔案的規則數為 0")
        return cls(rules)

    @classmethod
    def load(cls, path: str | Path, *, max_age_days: int) -> "PublicSuffixList":
        """自本機快照目錄載入。`max_age_days` 為必填的僅限關鍵字參數。

        `max_age_days` **沒有預設值**：給一個預設值等於替呼叫端決定
        「多舊算太舊」，而那是部署環境的知識。後綴的變動以月計，
        PSL 的合理值遠大於黑名單的，但「遠大於」不是「不用管」。

        一致性檢查（檔案存在、manifest 欄位齊全、sha256 相符、規則數非 0、
        有 ICANN 區段、未超過 `max_age_days`）全部在此完成 ——
        查詢路徑每則訊息跑數次，不能在那裡做檢查。
        """
        directory = Path(path)
        snapshot = directory / PSL_FILENAME
        manifest_path = directory / PSL_MANIFEST_FILENAME
        if not snapshot.is_file():
            raise FileNotFoundError(
                f"PSL 快照不存在：{snapshot}。請先執行 `{PSL_FETCH_COMMAND}` 取得"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"PSL manifest 不存在：{manifest_path}。請先執行 `{PSL_FETCH_COMMAND}` 取得"
            )
        manifest = read_manifest(manifest_path)
        for field in PSL_MANIFEST_FIELDS:
            if field not in manifest:
                raise ValueError(
                    f"PSL manifest 缺少欄位 {field!r}：{manifest_path}，"
                    f"實際欄位為 {sorted(manifest)}"
                )
        raw = snapshot.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != manifest["sha256"]:
            raise ValueError(
                f"PSL 快照的 sha256 與 manifest 不符：檔案為 {digest}、"
                f"manifest 為 {manifest['sha256']}"
            )
        fetched_at = parse_iso_date(manifest["fetched_at"], "fetched_at")
        age_days = (utc_today() - fetched_at).days
        if age_days > max_age_days:
            raise ValueError(
                f"PSL 快照過期：fetched_at 為 {fetched_at.isoformat()}、距今 {age_days} 天、"
                f"max_age_days 為 {max_age_days}。請重新執行 `{PSL_FETCH_COMMAND}`"
            )
        return cls.parse(raw.decode("utf-8"))

    def public_suffix(self, host: str) -> str | None:
        """主機的公共後綴。

        依 PSL 的比對演算法：取匹配規則中標籤數最多者為 prevailing rule，
        例外規則優先於萬用規則；無任何規則匹配時視為 `*`（最後一個標籤）。
        由左往右掃描標籤起點，第一個命中的就是標籤數最多的那個。
        """
        if not host:
            return None
        labels = host.split(".")
        if any(not label for label in labels):
            # 前導點、尾端點或連續點的主機不是合法主機（PSL 官方測試向量對
            # `.com`、`.example.com` 的期望都是「沒有可註冊網域」）。
            return None
        for i in range(len(labels)):
            candidate = ".".join(labels[i:])
            rule = self._rules.get(candidate)
            if rule is not None and rule[0] is RuleKind.EXCEPTION:
                # 例外規則優先，其公共後綴為該規則去掉最左邊的標籤。
                return ".".join(labels[i + 1 :])
            if rule is not None and rule[0] is RuleKind.NORMAL:
                return candidate
            wildcard = self._rules.get(".".join(["*", *labels[i + 1 :]]))
            if wildcard is not None and wildcard[0] is RuleKind.WILDCARD:
                return candidate
        return labels[-1]

    def registrable_domain(self, host: str) -> str | None:
        """公共後綴再往左多取一個標籤。主機本身即為後綴或為 IP 字面值時回傳 `None`。

        **同時採用 ICANN 與 PRIVATE 兩個區段。** 只用 ICANN 段時
        `phish123.wixsite.com` 的可註冊網域會是 `wixsite.com`，於是 165 黑名單裡
        任何一筆 `*.wixsite.com` 的釣魚頁，以可註冊網域比對就會讓全平台的站台命中；
        網域年齡也會回報 Wix 的註冊年份，與那個釣魚頁無關。
        殘留風險（未登記於 PRIVATE 區段的託管平台）由 `url_check` 以
        `hard=False` 承接。
        """
        if not host or is_ip_literal(host):
            return None
        suffix = self.public_suffix(host)
        if suffix is None or host == suffix:
            return None
        suffix_labels = suffix.split(".")
        labels = host.split(".")
        if len(labels) <= len(suffix_labels):
            return None
        return ".".join(labels[len(labels) - len(suffix_labels) - 1 :])


# 與 TLD 撞名的常見副檔名。每一項都同時滿足兩件可各自驗證的事：
# 它是 PSL ICANN 區段裡真實存在的 TLD，而且它是常見副檔名。
#
#   zip / mov —— 2023 年開放的 gTLD，開放當時即因與壓縮檔、影片檔撞名而受批評
#   app / dev —— 2018 年開放的 gTLD，同時是 iOS 與前端專案常見的檔名尾綴
#   sh        —— 聖赫勒拿島，同時是 shell 腳本
#   py        —— 巴拉圭，同時是 Python 原始碼
#   md        —— 摩爾多瓦，同時是 Markdown
#   pl        —— 波蘭，同時是 Perl
#   rs        —— 塞爾維亞，同時是 Rust
#   so        —— 索馬利亞，同時是共享函式庫
#
# 只套用於**不帶 scheme** 的那一趟。`https://report.zip/a` 有 scheme，
# 作者明示了它是 URL，沒有歧義。
# 代價明說：`readme.md` 若真的是一個網站就抓不到。方向是對的 ——
# 誤抽一個檔名會讓 TLD 風險與網域年齡對一個不存在的網域發訊號，
# 漏抽一個罕見寫法只是少一個訊號。
EXTENSION_LIKE_TLDS = frozenset({"zip", "mov", "app", "dev", "sh", "py", "md", "pl", "rs", "so"})

_SCHEME_PATTERN = re.compile(r"(?i)https?://[^\s<>\"'）」』】]+")
_BARE_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9.\-@/])"
    r"(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}"
    r"(?::\d{1,5})?"
    r"(?:/[^\s<>\"'）」』】]*)?"
)

# 尾綴剝除。`add-normalize-text` 把全形 `！？；` 收斂成 ASCII，所以抽取時
# **分不出** `https://evil.com/a?` 結尾的 `?` 是 query 起始還是句末問號 ——
# 它們現在是同一個字元。regex 解不了（`?` 在 URL 裡合法），因此以後處理解。
_TRAILING_PUNCT = "!?;.,:"

# 候選字串的合法字元。RFC 3986 的 URI 只由 ASCII 組成，非 ASCII 必須百分比編碼，
# 所以第一個非 ASCII 字元就是 URL 的結束。
#
# 這一條是**抽取器與切句遮罩分歧的地方**，而且是刻意的。遮罩的 `\S+` 會吃掉
# 緊接在網址後的中文（`https://x.cc/a點擊領取` 整段視為網址），當遮罩它是對的
# ——寧可少切不可切壞；當抽取結果就是錯的：`ExtractedUrl.url` 會被寫進 `detail`
# 呈現給使用者，而「網址 https://x.cc/a點擊領取 在黑名單上」是一句假話。
# 代價：路徑中帶未編碼中文的合法網址會被截短。那類網址的**主機**仍完整，
# 而本 PR 的四個訊號沒有一個讀 path。
_URI_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
)
_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{", "）": "（", "」": "「", "』": "『", "】": "【"}

_IPV4_PATTERN = re.compile(r"\A(?:\d{1,3}\.){3}\d{1,3}\Z")
_IPV6_CHARS = frozenset("0123456789abcdefABCDEF:.")


@dataclass(frozen=True, slots=True)
class ExtractedUrl:
    """一個抽取到的 URL，攜帶**結構事實**，不攜帶任何判斷。

    欄位分兩類。查詢鍵：`host`、`raw_host`、`registrable_domain`、`coords`。
    結構事實：`is_ip_literal`、`has_userinfo`、`scheme_present`、`idna_failed` ——
    抽取過程中順手就知道、之後再算就得重新解析一次 URL 的東西。

    沒有任何「風險」「可疑」「分數」欄位。判斷全部在 `url_check.py`：
    抽取層一旦開始判斷，消融實驗就關不掉它。

    - `url` —— 正規化後的 URL 字串，也是去重的鍵。
      **主機保證完整，query 不保證完整**：不帶 scheme 的寫法其 `?` 之後
      可能落在下一個句子（切句遮罩只保護帶 scheme 與 `www.` 的寫法）。
    - `host` —— punycode（ASCII）、已轉小寫、已移除前導 `www.`，不含連接埠、
      使用者資訊與路徑。黑名單比對 MUST 用這個欄位。
      `idna_failed` 為 True 時沒有 punycode 形式可用，此欄位退為 `raw_host` 的值。
    - `raw_host` —— 正規化後的 Unicode 形式，未轉 punycode。
      混合文字系統判定與品牌相似度 MUST 用這個欄位 ——
      `аpple.com`（西里爾 а）轉成 `xn--pple-43d.com` 之後那個資訊就沒了。
    - `coords` —— 該 URL 出現過的全部句子座標，依出現順序。
      長度即出現次數（重複貼同一個連結本身可能是訊號）。
    """

    url: str
    host: str
    raw_host: str
    registrable_domain: str | None
    is_ip_literal: bool
    has_userinfo: bool
    scheme_present: bool
    idna_failed: bool
    coords: tuple[Coord, ...]


def unicode_host(raw_host: str) -> str:
    """主機的 Unicode 正規化：去前後空白與尾端點、轉小寫、去前導 `www.`。"""
    host = raw_host.strip().strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_host(raw_host: str) -> str:
    """主機的正規化並轉為 punycode。空主機或無法以 IDNA 編碼時拋 `ValueError`。

    黑名單快照的鍵與查詢的鍵 MUST 出自這一個函式 —— `tools/` 不自行實作一份。
    三份 165 資料的寫法不一致（160055 的網址帶 `www.` 前綴、176455 的網域不帶），
    若兩邊各算一次，不一致時不會有任何錯誤訊息，只會「明明在清單裡卻查不到」。
    """
    host = unicode_host(raw_host)
    if not host:
        raise ValueError(f"主機為空：raw_host={raw_host!r}")
    labels = []
    for label in host.split("."):
        if not label:
            raise ValueError(f"主機含空標籤：host={host!r}")
        if label.isascii():
            labels.append(label)
            continue
        try:
            labels.append(label.encode("idna").decode("ascii"))
        except UnicodeError as exc:
            raise ValueError(f"主機的標籤無法以 IDNA 編碼：host={host!r}、label={label!r}") from exc
    return ".".join(labels)


def is_ip_literal(host: str) -> bool:
    """主機是否為 IP 位址字面值（IPv4 點分十進位，或方括號包住的 IPv6）。"""
    return bool(_IPV4_PATTERN.match(host)) or (host.startswith("[") and host.endswith("]"))


def extract_urls(doc: Document, psl: PublicSuffixList) -> list[ExtractedUrl]:
    """自 `Document` 逐句抽取 URL，去重並累積座標。

    抽取在 `doc.sentences`（**正規化後**）上做，不在 `raw_sentences` 上 ——
    正規化把全形 URL 拉平（`ｈｔｔｐｓ：／／ｅｖｉｌ．ｃｏｍ`）、
    移除零寬字元與雙向覆寫字元，三件都是我們要的。
    代價是原文的規避痕跡不在抽取結果裡，但痕跡沒有遺失：
    `coords` 指向句子，`doc.raw_at(coord)` 就是原文片段。

    去重以**正規化後的 URL 字串**為鍵：同一個 URL 在訊息中出現三次
    （「點這裡 https://x/ 或複製 https://x/ 到瀏覽器」）若產生三筆，
    下游就會對同一個主機產生三筆內容完全相同的 `CheckResult`。
    `check.py` 說的「訊息含三個 URL 時應各產一筆」講的是三個**不同**的 URL。

    順序為首次出現順序，對同一輸入穩定 —— 讓 `Verdict.checks` 可寫成測試斷言，
    也讓消融實驗的兩次執行可逐行比對。

    `psl` 為顯式參數而非模組層級的全域：spec 的簡寫是 `extract_urls(doc)`，
    但可註冊網域與無 scheme 抽取的 TLD 集合都來自 PSL，
    而 PSL 的載入 MUST 是呼叫端顯式觸發的動作。
    """
    order: list[str] = []
    coords_by_url: dict[str, list[Coord]] = {}
    built: dict[str, ExtractedUrl] = {}
    for sentence, coord in zip(doc.sentences, doc.coords, strict=True):
        for candidate in _candidates(sentence, psl.icann_tlds):
            extracted = _build(candidate, psl)
            if extracted is None:
                continue
            if extracted.url not in coords_by_url:
                order.append(extracted.url)
                coords_by_url[extracted.url] = []
                built[extracted.url] = extracted
            coords_by_url[extracted.url].append(coord)
    results = []
    for url in order:
        base = built[url]
        results.append(
            ExtractedUrl(
                url=base.url,
                host=base.host,
                raw_host=base.raw_host,
                registrable_domain=base.registrable_domain,
                is_ip_literal=base.is_ip_literal,
                has_userinfo=base.has_userinfo,
                scheme_present=base.scheme_present,
                idna_failed=base.idna_failed,
                coords=tuple(coords_by_url[url]),
            )
        )
    return results


def read_manifest(path: Path) -> dict[str, object]:
    """讀取一份 manifest。非 JSON 物件時拋 `ValueError`。"""
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"manifest 必須為 JSON 物件：{path}，實為 {type(loaded).__name__}")
    return loaded


def parse_iso_date(value: object, field: str) -> date:
    """把 manifest 中的 ISO 8601 字串解析為日期。不合法時拋 `ValueError` 並指出欄位。"""
    if not isinstance(value, str):
        raise ValueError(f"manifest 的 {field} 必須為字串，實為 {value!r}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"manifest 的 {field} 不是合法的 ISO 8601 時間：{value!r}") from exc


def utc_today() -> date:
    """UTC 的今天。過期判定以此為基準，不用本地時區。"""
    return datetime.now(timezone.utc).date()


@dataclass(frozen=True, slots=True)
class _Candidate:
    text: str
    scheme_present: bool


def _candidates(sentence: str, icann_tlds: frozenset[str]) -> list[_Candidate]:
    """一個句子裡的 URL 候選，依出現順序。分兩趟：帶 scheme 與不帶 scheme。"""
    found: list[tuple[int, _Candidate]] = []
    scheme_spans: list[tuple[int, int]] = []
    for match in _SCHEME_PATTERN.finditer(sentence):
        scheme_spans.append((match.start(), match.end()))
        text = _strip_trailing(match.group(), icann_tlds)
        if text:
            found.append((match.start(), _Candidate(text=text, scheme_present=True)))
    for match in _BARE_PATTERN.finditer(sentence):
        if any(start <= match.start() < end for start, end in scheme_spans):
            continue
        text = _strip_trailing(match.group(), icann_tlds)
        if not text:
            continue
        # 最後一個標籤是否為真實 TLD 在此以 `icann_tlds` 判定，不寫進 regex ——
        # 一千四百多個 TLD 塞進 alternation 會讓 regex 隨資料變動，
        # 而且無法對「為什麼這個字串沒被抽到」給出可讀的解釋。
        host_part = text.split("/", 1)[0].split(":", 1)[0]
        tld = host_part.rsplit(".", 1)[-1].lower()
        if tld not in icann_tlds or tld in EXTENSION_LIKE_TLDS:
            continue
        found.append((match.start(), _Candidate(text=text, scheme_present=False)))
    found.sort(key=lambda item: item[0])
    return [candidate for _, candidate in found]


def _strip_trailing(text: str, icann_tlds: frozenset[str]) -> str:
    """截到第一個非 URI 字元與主機尾巴，再剝除結尾的標點與**不成對**的右括號、右引號。

    括號以「左右計數」判定而非以位置判定：
    `（詳見 https://evil.com/a）` 的 `）` 不是 URL 的一部分，
    但 `https://x.cc/wiki/A_(B)` 的 `)` 是。
    """
    text = _trim_host_tail(_truncate_path_at_non_uri(text), icann_tlds)
    while text:
        stripped = text.rstrip(_TRAILING_PUNCT)
        if stripped != text:
            text = stripped
            continue
        closer = text[-1]
        if closer in _BRACKET_PAIRS and text.count(_BRACKET_PAIRS[closer]) < text.count(closer):
            text = text[:-1]
            continue
        break
    return text


def _truncate_path_at_non_uri(text: str) -> str:
    """把候選字串截在**路徑之後**第一個非 URI 字元處。

    RFC 3986 的 URI 只由 ASCII 組成，非 ASCII 必須百分比編碼，
    所以路徑裡的第一個中文字就是 URL 的結束。這是抽取器與切句遮罩
    刻意分歧的地方：遮罩的 `\\S+` 吃掉緊接在網址後的中文是對的
    （寧可少切不可切壞），抽取結果吃掉它就是錯的 ——
    `url` 會被寫進 `detail`，「網址 https://x.cc/a點擊領取 在黑名單上」是一句假話。

    截斷只從**路徑起點**算起，authority 不截 —— `https://中国.com/a` 的
    主機是合法的 IDN。主機後直接接中文而沒有路徑分隔符的寫法由
    `_trim_host_tail()` 處理。
    """
    _, path_start = _authority_span(text)
    for position in range(path_start, len(text)):
        if text[position] not in _URI_CHARS:
            return text[:position]
    return text


def _authority_span(text: str) -> tuple[int, int]:
    """候選字串中 authority 的 `[start, end)` 區間。"""
    scheme_end = text.find("://")
    start = scheme_end + 3 if scheme_end != -1 else 0
    end = len(text)
    for separator in "/?#":
        position = text.find(separator, start)
        if position != -1:
            end = min(end, position)
    return start, end


def _trim_host_tail(text: str, icann_tlds: frozenset[str]) -> str:
    """主機的最後一個標籤不是真實 TLD 且帶中文尾巴時，截在那個中文字元處。

    這條補的是 `_truncate_path_at_non_uri()` 的缺口：**沒有路徑分隔符**的寫法。
    Cofacts 樣本實測到 `https://6000.gov.tw會在3月22日8時起,開放民眾登記。`
    這種形態（628 個抽取結果中 3 個，0.5%），主機會帶著整段中文，
    而黑名單比對的鍵就成了一段中文。

    條件寫得很窄，只在兩件事同時成立時才動手：最後一個標籤**不是**真實 TLD，
    且該標籤內有非 ASCII 字元。因此合法的 IDN 主機（`中国.com`、`台灣銀行.tw`）
    完全不受影響 —— 它們的最後一個標籤是真實 TLD。
    """
    start, end = _authority_span(text)
    authority = text[start:end]
    hostport = authority.rpartition("@")[2]
    host = hostport.partition(":")[0] if hostport.count(":") == 1 else hostport
    if "." not in host:
        return text
    last_label = host.rsplit(".", 1)[-1].lower()
    if last_label in icann_tlds or _punycode_or_none(last_label) in icann_tlds:
        return text
    offset = text.index(host, start) + len(host) - len(last_label)
    for position, character in enumerate(last_label):
        if not character.isascii():
            return text[: offset + position]
    return text


def _punycode_or_none(label: str) -> str | None:
    if label.isascii():
        return None
    try:
        return label.encode("idna").decode("ascii")
    except UnicodeError:
        return None


def _build(candidate: _Candidate, psl: PublicSuffixList) -> ExtractedUrl | None:
    """把一個候選字串解析成 `ExtractedUrl`。形狀不是主機時回傳 `None`。

    解析依 RFC 3986：`@` 之前的部分是使用者資訊而不是主機。
    `https://post.gov.tw@evil.com/` 的主機是 `evil.com` ——
    不照這條解析，抽出來的 host 會是攻擊者想讓人看到的那一個。
    """
    rest = candidate.text
    scheme = ""
    if candidate.scheme_present:
        scheme, _, rest = rest.partition("://")
        scheme = scheme.lower()
    authority_end = len(rest)
    for separator in "/?#":
        position = rest.find(separator)
        if position != -1:
            authority_end = min(authority_end, position)
    authority = rest[:authority_end]
    tail = rest[authority_end:]

    has_userinfo = "@" in authority
    hostport = authority.rpartition("@")[2] if has_userinfo else authority

    if hostport.startswith("["):
        closing = hostport.find("]")
        if closing == -1:
            return None
        raw = hostport[1:closing]
        port = hostport[closing + 1 :]
        if not raw or not set(raw) <= _IPV6_CHARS:
            return None
        ip_literal = True
        host_text = f"[{raw}]"
    else:
        head, _, maybe_port = hostport.rpartition(":")
        if head and maybe_port.isdigit():
            raw, port = head, ":" + maybe_port
        else:
            raw, port = hostport, ""
        if not raw:
            return None
        ip_literal = bool(_IPV4_PATTERN.match(raw))
        host_text = raw.lower()

    raw_host = unicode_host(raw)
    if not raw_host:
        return None
    try:
        host = normalize_host(raw)
        idna_failed = False
    except ValueError:
        # 編碼失敗**不丟棄**：丟棄等於安靜地少一個訊號，而攻擊者可以刻意構造
        # 一個編不出來的主機來躲掉全部四個 URL 檢查。真實網站的主機幾乎必然
        # 編得出來（編不出來就沒辦法被 DNS 解析），所以失敗本身是高精確度的異常訊號，
        # 由 `url_host_shape` 當成訊號處理。
        host = raw_host
        idna_failed = True
    registrable = None if (idna_failed or ip_literal) else psl.registrable_domain(host)
    if not ip_literal:
        host_text = host
    prefix = f"{scheme}://" if candidate.scheme_present else ""
    return ExtractedUrl(
        url=f"{prefix}{host_text}{port}{tail}",
        host=host,
        raw_host=raw_host,
        registrable_domain=registrable,
        is_ip_literal=ip_literal,
        has_userinfo=has_userinfo,
        scheme_present=candidate.scheme_present,
        idna_failed=idna_failed,
        coords=(),
    )


def _encode_suffix(suffix: str) -> str:
    """PSL 規則的每個標籤以 punycode 儲存，使查詢時不需要兩種表示法。"""
    labels = []
    for label in suffix.split("."):
        if label == "*" or label.isascii():
            labels.append(label.lower())
            continue
        try:
            labels.append(label.encode("idna").decode("ascii"))
        except UnicodeError as exc:
            raise ValueError(
                f"PSL 規則的標籤無法以 IDNA 編碼：rule={suffix!r}、label={label!r}"
            ) from exc
    return ".".join(labels)
