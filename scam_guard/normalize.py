"""字元層正規化與原文位置對映。

此模組不 import 專案內任何模組（含 `types.py`），只依賴標準庫 ——
正規化是最底層的關注點，它不需要知道訊息、請求或檢查的存在。

正規化的核心限制是**不可逆性必須被補償**：拆字、全形半形混用、零寬字元插入
本身就是規避偵測要抓的東西，也是要呈現給使用者看的東西，而正規化會消滅它們。
因此 `normalize_text()` 除了正規化後的文字，還回傳字元層的位置對映，
使正規化後的任意區間可映回原文的對應片段。
"""

import unicodedata
from dataclasses import dataclass

# 不可見字元 —— 以顯式清單列舉而非以 Unicode 類別（`Cf`）判定。
# 類別判定涵蓋面更廣，但清單不可見：沒有人能在 review 時說出它到底刪了什麼，
# 而且它會隨 Python 的 Unicode 資料版本改變行為。顯式清單可測、可審、
# 可在發現新的規避字元時單行擴充，是常數而非邏輯。
# 以 `\u` 轉義書寫而非直接貼字元：不可見字元貼進原始碼後在 review 時同樣看不見。
INVISIBLE = (
    "\u200b\u200c\u200d\u2060\ufeff"  # 零寬空格、零寬非連字、零寬連字、字連接符、BOM
    "\u00ad"  # 軟連字號
    "\u200e\u200f"  # 左至右、右至左標記
    "\u202a\u202b\u202c\u202d\u202e"  # 雙向嵌入、覆寫與還原
    "\u2066\u2067\u2068\u2069"  # 雙向隔離與還原
)

# 需統一為半形空格的空白字元。**不含換行** —— 換行由 `normalize_text()` 另行
# 收斂（`\r\n` 與 `\r` 皆成為 `\n`），且必須保留為換行，切句依賴它。
# NFKC 已處理其中多數（U+3000、U+00A0 等），此處仍逐一列出，
# 使「哪些字元被視為空白」可被測試逐項驗證，而非隱含在 NFKC 的行為裡。
WHITESPACE = (
    "\t"
    "\u00a0\u1680"  # 不斷行空格、Ogham 空格
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u202f\u205f\u3000"  # 窄不斷行空格、數學中等空格、表意空格
)

# NFKC 未涵蓋的標點等價寫法。NFKC 已把全形 `！？；，：` 映到 ASCII、
# 把半形 `｡` 映到 `。`，此表只補它沒處理到的同一標點的其他寫法。
#
# 刻意**不**收錄兩組對映：
#
# - ASCII `.` 不映到 `。` —— `.` 出現在網址、小數與英文縮寫裡，
#   把它當句末標點會把 `https://reurl.cc/abc` 切成三段。
#   中文詐騙訊息的句末用 `。`，不用 `.`。
# - `、` 不併入 `,` —— 頓號是列舉分隔，逗號是語氣停頓。規則層需要在句內
#   以逗號再切子句，把 `股票、基金、期貨` 切成三個子句是錯的。
#
# 一般原則：正規化處理**書寫層**的差異，不處理**語意層**的差異。
# 判斷兩個標點是不是同一個意思，那是規則層的知識。
PUNCT = {
    "❗": "!",  # U+2757 重驚嘆號
    "❕": "!",  # U+2755 白色驚嘆號
    "❓": "?",  # U+2753 重問號
    "❔": "?",  # U+2754 白色問號
}


@dataclass(frozen=True)
class NormalizedText:
    """正規化後的文字，以及它到原文的字元層位置對映。

    `offsets[i]` 是 `text[i]` 在 `raw` 中的來源起始索引，尾端多一個哨兵
    `len(raw)`，因此正規化後的任意區間可映回原文：

        raw_span(a, b) == raw[offsets[a]:offsets[b]]

    被刪除的字元（零寬字元）不佔 `text` 的位置，因此自然被歸入**前一個**區間；
    開頭被刪除的字元則歸入第一個區間（`offsets[0]` 恆為 0），使全文區間
    映回等於原文。這正是我們要的 —— 呈現給使用者的片段必須看得見規避痕跡。

    `text` 為空（輸入全為不可見字元）時，對映僅剩哨兵一個元素，
    此時沒有任何區間可查詢，全文映回的性質不適用。
    """

    raw: str
    text: str
    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.text) + 1:
            raise ValueError(
                f"位置對映長度必須為文字長度加一：len(offsets)={len(self.offsets)}、"
                f"len(text)={len(self.text)}"
            )
        for i in range(1, len(self.offsets)):
            if self.offsets[i] < self.offsets[i - 1]:
                raise ValueError(
                    f"位置對映必須單調不減：offsets[{i - 1}]={self.offsets[i - 1]}、"
                    f"offsets[{i}]={self.offsets[i]}"
                )
        if self.text and self.offsets[0] != 0:
            raise ValueError(f"位置對映的首元素必須為 0，實為 {self.offsets[0]}")
        if self.offsets[-1] != len(self.raw):
            raise ValueError(
                f"位置對映的哨兵必須為原文長度 {len(self.raw)}，實為 {self.offsets[-1]}"
            )

    def raw_span(self, a: int, b: int) -> str:
        """回傳正規化後區間 `[a, b)` 在原文中的對應片段。"""
        return self.raw[self.offsets[a] : self.offsets[b]]


def _map_char(ch: str) -> str:
    """單一原文字元的正規化結果，可能為空字串（刪除）或多個字元（NFKC 展開）。

    依序套用：不可見字元判定（刪除）→ NFKC → 空白統一 → 標點對映。
    全部為一對零、一對一或一對多的映射，**沒有多對一** ——
    這讓位置對映的建構保持機械化：每個輸出字元的來源恰好是一個輸入字元。
    """
    if ch in INVISIBLE:
        return ""
    out = []
    for c in unicodedata.normalize("NFKC", ch):
        if c in WHITESPACE:
            out.append(" ")
        elif c in PUNCT:
            out.append(PUNCT[c])
        else:
            out.append(c)
    return "".join(out)


def normalize_text(raw: str) -> NormalizedText:
    """正規化一段文字，並產出它到原文的位置對映。

    **逐字元**套用 NFKC，而非整串套用：整串 `unicodedata.normalize("NFKC", s)`
    不告訴你輸出的第 i 個字元來自輸入的第幾個字元，而事後用 `difflib` 對齊是
    啟發式的，在重複字元上會猜錯位置 —— 猜錯的結果是呈現給使用者的證據片段
    指到別的地方，而這種錯沒有任何機制會報告它。

    已知代價：逐字元套用不會執行跨字元的組合（組合附加符號、諺文音節組合）。
    台灣中文詐騙語料幾乎不含這類序列，且「少正規化一點」的後果是漏掉一點命中，
    不是產生錯誤答案。

    刻意**不做**：繁簡轉換（一對多映射，轉錯會把中性詞變成命中詞）、
    大小寫轉換（全大寫本身是訊號）、連續空白壓縮（空白插入是拆字規避手法，
    `add-evasion-check` 要數的正是那些空白）、PII 遮蔽、任何內容層的改寫。
    """
    chars: list[str] = []
    offsets: list[int] = []
    i = 0
    total = len(raw)
    while i < total:
        ch = raw[i]
        step = 1
        if ch == "\r":
            # 換行寫法收斂：`\r\n` 與 `\r` 皆成為 `\n`。`\r\n` 是唯一的多對一
            # 映射，被吃掉的 `\n` 歸入同一個輸出字元的原文區間。
            piece = "\n"
            if i + 1 < total and raw[i + 1] == "\n":
                step = 2
        else:
            piece = _map_char(ch)
        for c in piece:
            chars.append(c)
            offsets.append(i)
        i += step
    if chars:
        # 開頭被刪除的字元沒有「前一個區間」可歸，歸入第一個區間，
        # 使 `raw_span(0, len(text))` 等於原文。
        offsets[0] = 0
    offsets.append(total)
    return NormalizedText(raw=raw, text="".join(chars), offsets=tuple(offsets))
