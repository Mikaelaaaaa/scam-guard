"""子句切分與否定範疇 —— 言語行為比對的位置基礎。

`add-split-sentences` 刻意把逗號與頓號留在句子裡，唯一的理由就是本模組：
「您的驗證碼是 123456，請勿告訴他人」是正當的一次性密碼簡訊，
能擋掉它的只有「請勿」，而「請勿」管轄的範圍必須以標點界定，不能以字元距離界定。

本模組只做字串上的位置計算，不 import `scam_guard` 內的任何模組。
"""

CLAUSE_BREAK = ",:"
"""子句分隔符：**只有逗號與冒號**，頓號不切。

寫成正規化後的形狀 —— NFKC 已把全形 `，：` 收斂為 ASCII `,` `:`，
而 `、` 維持原樣（`normalize.py` 的 `PUNCT` 明白記錄了不併入的理由）。

**頓號不切的後果是可陳述的**：

    不要把帳號、密碼告訴任何人

切在頓號上會得到「不要把帳號」與「密碼告訴任何人」兩個子句，
第二個子句失去了管轄它的否定詞 —— 否定詞與它管轄的列舉項被標點切開，
是這個機制唯一的結構性失效模式。同理「股票、基金、期貨」是一個列舉，
不是三個子句。

**冒號切的理由**：「本行公告：請至下列連結更新資料」這種「來源標籤 + 要求」
的結構若不切，來源標籤會被算進同一個子句，干擾接收者判定。
中文訊息裡冒號幾乎只用於提示與列舉引導，成本低。

被否決的替代方案：**以空白為子句邊界**。中文不以空白分詞，正常訊息的空白數
接近 0，這條規則在中文語料上退化成「永遠只有一個子句」。
"""

NEGATION = frozenset(
    {
        "不要",
        "不可",
        "不能",
        "不得",
        "不會",
        "不用",
        "不需",
        "無須",
        "毋須",
        "勿",
        "請勿",
        "切勿",
        "嚴禁",
        "禁止",
        "別",
        "莫",
        "絕不",
        "千萬不",
    }
)
"""否定詞。清單是常數不是邏輯，發現漏網的否定形式時單行擴充。

**完整性無法證明**，這是已知缺口。漏一個否定詞的後果是**誤報**
（該擋的沒擋），而誤報是本專案的強制驗收指標 —— 因此清單寧可寬。

反方向的代價已知且接受：「別」「莫」以子字串比對，
會被「特別」「個別」「莫名」誤觸而判成否定，後果是**漏報**（規則不命中）。
兩個方向都無法同時最佳化時，本專案選擇漏報。
"""


def _strip_span(text: str, a: int, b: int) -> tuple[int, int]:
    """回傳去除前後空白後的區間。整段皆為空白時回傳 `a >= b` 的空區間。"""
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def split_clauses(sentence: str) -> list[tuple[int, int]]:
    """把句子切成子句，回傳子句在**句內**的 `[start, end)` 區間清單。

    回傳區間而非字串：否定範疇與接收者判定要問的是「誰在誰前面」，
    切成字串之後位置就得由呼叫端自己再算一次，而算錯不會有人報告。

    分隔符本身不屬於任何子句；前後空白去除；僅由空白組成的片段不產生子句。
    句中不含 `CLAUSE_BREAK` 時回傳單一涵蓋全句的區間，不拋例外 ——
    這是常態而非例外情況，短句本來就沒有停頓標點。

    **子句切分不改變證據粒度。** `CheckResult.evidence` 仍是句子座標
    `(訊息序號, 訊息內句子序號)`，子句編號不外流到跨層的共用語言裡。
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(sentence):
        if char in CLAUSE_BREAK:
            spans.append((start, index))
            start = index + 1
    spans.append((start, len(sentence)))

    clauses: list[tuple[int, int]] = []
    for a, b in spans:
        a, b = _strip_span(sentence, a, b)
        if a < b:
            clauses.append((a, b))
    return clauses


def is_negated(sentence: str, clause: tuple[int, int], predicate_pos: int) -> bool:
    """述語是否落在否定範疇內。

        否定成立 ⟺ ∃ 否定詞 n，n 與述語 p 同子句，且 index(n) < index(p)

    **簽章刻意不含任何距離或視窗參數。** `add-split-sentences` 的 design 已用
    一個反例否決了字元距離視窗：

        請不要在任何情況下告訴任何人
                └──── 9 字 ────┘

    「不要」距「告訴」9 個字，視窗要涵蓋它得開到 10 字以上，而 10 字的視窗
    會跨過逗號誤傷隔壁子句。視窗這個參數在兩個方向上都錯，沒有可用的取值。

    反方向同樣重要，它證明子句級範疇不是「把視窗開到無限大」：

        不要告訴別人，把驗證碼傳給我
        └─ 子句 1 ─┘└─── 子句 2 ───┘

    子句 1 的「不要」管不到子句 2。任何以「句子內有沒有否定詞」為條件的作法，
    都會被前面一句無害的否定整句蓋掉 —— 那是詐騙者只要加一句廢話就能繞過的漏洞。

    中文否定詞前置於述語，因此不處理後置否定。
    已知不處理：雙重否定（「不得不告訴我」）會被判為否定成立 → 不命中 → 漏報。

    `predicate_pos` 不在 `clause` 內時拋 `ValueError` —— 那代表呼叫端把別的
    子句的位置傳了進來，靜默回傳布林值會讓錯誤的範疇判定沒有人發現。
    """
    start, end = clause
    if not start <= predicate_pos < end:
        raise ValueError(f"述語位置不在子句內：predicate_pos={predicate_pos}、clause={clause}")
    return any(sentence.find(word, start, predicate_pos) != -1 for word in NEGATION)
