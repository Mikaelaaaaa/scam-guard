"""為開發樣本加上旗標。**標記，不刪除。**

這個模組的名字叫 filter，但它一筆都不丟。這是刻意的落差，寫在這裡免得有人
照字面去刪資料 —— 名字沿用 workplan 不改，改名要動 workplan 而那會影響別人的分支。

**因為兩個下游對同一批資料的判斷相反。** 純網址訊息對 `rule-signals` 是雜訊
（一行 URL 裡沒有話術，規則層不可能命中，把它算進命中率等於用一個規則層無法
作答的題目扣它的分），但對 `url-signals` 這正是最乾淨的素材。刪掉，`url-signals`
就得自己再抓一次。

四個旗標各有指名的消費者：

| 旗標 | 消費者 |
|------|--------|
| `url_only` | `url-signals` —— 沒有文字訊號干擾的純 URL 素材 |
| `line_export` | `add-testset` —— 真正的多則對話，跨訊息脈絡測試案例的候選 |
| `too_short` | 本 change 的手動檢視（安全網，攔正規化後的內容殘骸） |
| `too_long` | `add-context-limits` 的壓力測試素材 |

⚠️ `line_export` 是最可能落空的一個：它只標記不解析，`add-testset` 不接手就白標了。

用法：

    python -m tools.message_filter \\
        data/cofacts_scam.jsonl data/cofacts_hard_negative.jsonl \\
        --out data/dev_sample.jsonl

取用方式就是一行 `jq`：

    jq -c 'select(.flags | index("url_only"))' data/dev_sample.jsonl   # url-signals 要的
    jq -c 'select(.flags | length == 0)'       data/dev_sample.jsonl   # rule-signals 要的

單一檔案而非分檔：一筆可以同時符合兩個旗標（實測有 `url_only` 且 `too_long` 的
組合），分檔就得決定它進哪一個，而任何決定都會讓某一邊少看到東西。
旗標是集合，檔案不是。
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from scam_guard.normalize import normalize_text

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
"""網址的最小詞法樣式。

**刻意不共用 `add-url-extract`（`url-signals` PR）的樣式。** 兩個理由：

1. 共用會讓 `dev-data` 依賴 `url-signals`。`dev-data` 在 workplan 上
   `depends-on: []`，能與其他 PR 平行開發，靠的就是不依賴任何人。
2. 兩者要回答的問題不同。`add-url-extract` 要的是 registrable domain、
   短網址展開、正規化後的比對鍵；這裡只要回答「除了網址還有沒有別的東西」，
   抓錯邊界的代價是某一則的旗標多一兩個字元的殘留，不影響判定。

代價是兩處的樣式會各自演化，此處記為刻意的重複。`add-url-extract` 完成後
本模組 **MAY** 改用它，但 **MUST NOT** 被要求這麼做 —— 一旦要求，
平行開發的前提就沒了。
"""

LINE_EXPORT_PATTERN = re.compile(r"\[LINE\][^\n]*聊天記錄")
"""LINE 自己產生的聊天記錄匯出檔標頭。格式固定，真實例子：

    [LINE] 與金融正義處理陳先生的聊天記錄
    儲存日期： 2024/09/10 11:35

    2024/08/19（一）
    上午09:53	小麥	 您好
    上午09:54	金融正義處理陳先生	Anya把您的情況大概跟我講了一下⋯

只比對第一行 —— 第二行的 `儲存日期：` 同樣是 LINE 產生的，但要求它同時出現會
漏掉使用者貼上時把該行刪掉的匯出檔，而第一行本身已經是沒有人會手打的字串，
精確度實質為 1。1,200 則中命中 1 則（0.2%）。

**MUST NOT 以時間戳、冒號或說話者樣式推測未帶標頭的對話片段。** 實測寬鬆的
啟發式（兩行以上的時間戳，或三處以上的「暱稱：」）在 1,200 則中標了 79 則
（6.6%），人工檢視**全部是誤判** —— 訂單通知型的詐騙訊息（冒號是欄位分隔）、
政策條列、兼職廣告。猜的後果是把 6.6% 的正常訊息從開發樣本裡剔除，
而且沒有任何機制會告訴你剔錯了：你只會看到樣本裡少了訂單通知型詐騙，
然後以為那種手法不存在。
"""

URL_ONLY_RESIDUAL = 10
"""移除網址後的殘留非空白字元數下界，低於此值即視為純網址。

門檻敏感度（門檻值 → 被標記的比例。以本模組**實際重跑**於 2026-09-14 抓取的
600 正例 + 600 難負例，`createdAt DESC` 最新六頁）：

| 門檻 | 3 | 5 | 8 | 10 | 15 | 20 | 30 |
|------|---|---|---|----|----|----|-----|
| 正例 | 11.7% | 11.7% | 11.7% | **11.8%** | 12.8% | 13.2% | 17.7% |
| 難負例 | 10.7% | 10.7% | 10.8% | **11.0%** | 11.5% | 12.3% | 14.7% |

3 到 10 之間結果幾乎不動，10 之後開始爬。**門檻選在平台的右端**：
再往左沒有差別，再往右就開始把有內容的訊息掃進來。

⚠️ 正例這一列與 design 記錄的數字逐格相符，**難負例這一列沒有重現** ——
design 記的是 15.7% → 16.0% → 20.0%。本次的難負例樣本有 78% 屬「政策宣導」、
20% 屬「商業廣告」，而 `實驗結果.md` 記的兩類基準率是 12% 與 24%，
混合比例不同足以造成這個差距，但**沒有證據能確定**那就是原因 ——
design 沒有記錄它那 600 則的分類組成。此處只記事實：重現不出來。
兩者相差 1.45 倍，未達 tasks 8.2 要求回頭查樣式的門檻。

不是 0 的理由：實測的純網址訊息確實有殘留 —— `https://youtu.be/...` 前面掛一個
表情符號、後面跟一個「請看」。
"""

MIN_NORM_LEN = 8
"""`norm_len` 的下界。低於此值即 `too_short`。

**刻意設得很低。** 原本的直覺是 20，實測推翻了 —— 20 以下有
`蝦皮帳號出租`（6 字，收購金融帳戶的廣告，真實詐騙）與
`是需要您寄出名下所有的卡片`（13 字，`add-speech-act-rules` 要抓的教科書例子）。
短不等於沒用。

真正該剔除的是「使用者送進 Cofacts 時自己打的提問」，而那與長度無關，
也沒有可靠的判定方式（以問號結尾不行，詐騙者也問問題）。**因此不做，記為已知限制。**

下界設 8 的作用本來是：擋掉正規化後幾乎沒有內容的殘骸。

⚠️ **實跑結果比 design 預期的更糟。** design 預期 1,200 則中標記 3 則（0.25%），
實際只標到 **1 則**（0.08%），而那一筆是 `蝦皮帳號出租` —— 正是 design 自己
列為「短但有內容的真實詐騙」的例子，不是殘骸。也就是說這個旗標目前**唯一**的
命中就是一個誤標。

沒有刪掉它（tasks 8.9 的自毀條款是「一筆都沒標到」，而它標到了一筆），
但它的價值比 design 承認的還低，留著只剩「長度界線在程式碼裡有個明確的位置」
這一個理由。`add-testset` 若要用無旗標子集，應知道它少了這一則真實詐騙。
"""

MAX_NORM_LEN = 2000
"""`norm_len` 的上界。高於此值即 `too_long`。1,200 則中超過的有 5 則（0.4%）。

實跑分佈（正例／難負例）：中位數 94／127、p90 327／430、p99 1,113／1,209、
最長 9,746／2,967。design 記的是「最長 10,000，疑似 Cofacts 端的長度上限」——
本次最長 9,746，沒有觀察到 10,000，因此**不重複那個推測**。
選 2,000 的理由是與 `add-context-limits` 相容 ——
那裡的 `max_chars` 是 50,000，單則不超過 2,000 時 100 則的則數上限至少還能放
25 則；若單則可以到 10,000，五則就吃掉字元預算，則數上限形同虛設。
"""

FLAG_ORDER = ("url_only", "line_export", "too_short", "too_long")
"""旗標的固定輸出順序，使輸出可逐行比對。"""

REQUIRED_FIELDS = ("id", "text", "label")

SNIPPET_LENGTH = 40


class MessageFilterError(Exception):
    """輸入不符合 `tools.cofacts_fetch` 的輸出格式，或同一筆有兩個版本。"""


def _url_only(normalized: str) -> bool:
    """含網址，且移除全部網址後剩餘的非空白字元少於 `URL_ONLY_RESIDUAL`。

    「含網址」這個前提是必要的 —— 少了它，一則五個字的純文字訊息也會因為殘留
    字元少於 10 而被當成純網址，而它連一個網址都沒有。
    """
    if not URL_PATTERN.search(normalized):
        return False
    residual = URL_PATTERN.sub(" ", normalized)
    return len("".join(residual.split())) < URL_ONLY_RESIDUAL


def _flags(normalized: str) -> tuple[str, ...]:
    """旗標判定，全部在**正規化後**的文字上進行。

    正規化後而非原文：`norm_len` 的尺度必須與 `add-context-limits` 的字元預算
    一致，而 `url_only` 若在原文上算，塞一萬個零寬字元就能讓一則純網址訊息
    看起來有內容 —— 兩個判定用同一份文字才不會出現這種縫。
    """
    found: list[str] = []
    if _url_only(normalized):
        found.append("url_only")
    if LINE_EXPORT_PATTERN.match(normalized.lstrip()):
        found.append("line_export")
    length = len(normalized)
    if length < MIN_NORM_LEN:
        found.append("too_short")
    if length > MAX_NORM_LEN:
        found.append("too_long")
    return tuple(flag for flag in FLAG_ORDER if flag in found)


def flag_message(text: str) -> tuple[str, ...]:
    """一則訊息的旗標，順序固定。純函式，同時成立的旗標全部回傳。

    旗標之間不互斥 —— 一串超過 2,000 字元的網址同時是 `url_only` 與 `too_long`。
    """
    return _flags(normalize_text(text).text)


def _parse_line(line: str, source: Path, line_number: int) -> dict:
    """解析一行輸入，缺欄位即中止。

    **不用 `.get(key, "")` 帶過。** 缺欄位代表輸入檔不是 `tools.cofacts_fetch`
    產的，那是要修的問題不是要容忍的問題。
    """
    record = json.loads(line)
    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            raise MessageFilterError(f"{source} 第 {line_number} 行缺少欄位 {field_name}")
    return record


def read_records(paths: list[Path]) -> tuple[list[dict], int]:
    """讀入全部輸入檔並以 `id` 去重，回傳 `(紀錄, 去重筆數)`。順序為輸入檔的順序。

    去重的責任在這裡而不在 `tools.cofacts_fetch` 的寫入端：續抓
    （`--after` 選得比中斷點早）可能產生重複行，而寫入端沒有前一趟的記憶，
    要擋就得先讀回整個檔案。這裡本來就要把整個檔案讀進來，成本趨近於零。

    同 `id` 但內文不同 **MUST raise** —— 那不是續抓造成的（同一篇文章的內文
    不會因為抓兩次而不同），它代表輸入檔混了文章被編輯過的兩份資料。
    安靜挑一份等於隨機決定用哪個版本。
    """
    records: list[dict] = []
    by_id: dict[str, dict] = {}
    duplicates = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = _parse_line(line, path, line_number)
                article_id = record["id"]
                if article_id in by_id:
                    existing = by_id[article_id]
                    if existing["text"] != record["text"]:
                        raise MessageFilterError(
                            f"id {article_id!r} 出現兩次且內文不同，無法判斷該用哪個版本："
                            f"先前為 {existing['text'][:SNIPPET_LENGTH]!r}、"
                            f"此筆（{path} 第 {line_number} 行）為 "
                            f"{record['text'][:SNIPPET_LENGTH]!r}"
                        )
                    duplicates += 1
                    continue
                by_id[article_id] = record
                records.append(record)
    return records, duplicates


def annotate(records: list[dict]) -> list[dict]:
    """為每一筆加上 `flags` 與 `norm_len`，保留輸入的全部欄位。**一筆都不丟。**

    `norm_len` 同時輸出，使界線可被重新檢驗而不必重跑抓取：

        jq -c 'select(.norm_len < 15)' data/dev_sample.jsonl | wc -l

    `norm_len` 是資料，`too_short` / `too_long` 只是目前同意的切點 ——
    兩個旗標可以有爭議，`norm_len` 不會。
    """
    annotated: list[dict] = []
    for record in records:
        normalized = normalize_text(record["text"]).text
        annotated.append({**record, "flags": list(_flags(normalized)), "norm_len": len(normalized)})
    return annotated


def _report(records: list[dict], duplicates: int) -> None:
    print(f"去重 {duplicates} 筆，輸出 {len(records)} 筆", file=sys.stderr)
    for flag in FLAG_ORDER:
        count = sum(1 for record in records if flag in record["flags"])
        share = count / len(records) if records else 0.0
        print(f"  {flag}：{count} 筆（{share:.1%}）", file=sys.stderr)
    clean = sum(1 for record in records if not record["flags"])
    print(f"  無旗標：{clean} 筆", file=sys.stderr)


def _require_git_ignored(path: Path) -> None:
    """輸出路徑必須被 git 忽略。以 `git check-ignore` 實測，不目視確認 `.gitignore`。

    與 `tools.cofacts_fetch` 的同名函式重複。不共用是因為兩個模組各自拋自己的
    例外型別，而共用一個就得讓其中一邊 import 另一邊的私有名稱。十行的重複
    比一條沒有理由的模組相依便宜。
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    if result.returncode == 1:
        raise MessageFilterError(
            f"輸出路徑 {path} 未被 git 忽略。開發樣本是真實民眾送進 Cofacts 的訊息，"
            f"不得進版控。請改用 data/ 之下的路徑。"
        )
    raise MessageFilterError(
        f"git check-ignore 無法判定 {path}："
        f"returncode={result.returncode}、stderr={result.stderr.strip()!r}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.message_filter",
        description="為 tools.cofacts_fetch 的輸出加上旗標與 norm_len。標記，不刪除。",
    )
    parser.add_argument("inputs", nargs="+", help="一至多個 JSONL 輸入檔")
    parser.add_argument("--out", required=True, help="輸出的 JSONL 路徑，必須被 git 忽略")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    try:
        _require_git_ignored(out_path)
        records, duplicates = read_records([Path(item) for item in args.inputs])
    except MessageFilterError as error:
        print(f"處理失敗：{error}", file=sys.stderr)
        return 1

    annotated = annotate(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        for record in annotated:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    _report(annotated, duplicates)
    print(f"完成：寫入 {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
