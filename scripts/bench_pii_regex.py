"""在 tw-PII-bench（910 題）上量測四條 regex 的 precision / recall / F1。

執行：`python -m scripts.bench_pii_regex`

**取得資料集**（未 gated，公開可取，與模型本身的 gated 狀態無關）：

    mkdir -p data/tw-pii-bench
    cd data/tw-pii-bench
    curl -LO https://huggingface.co/datasets/lianghsun/tw-PII-bench/resolve/main/data/short.parquet
    curl -LO https://huggingface.co/datasets/lianghsun/tw-PII-bench/resolve/main/data/mid.parquet
    curl -LO https://huggingface.co/datasets/lianghsun/tw-PII-bench/resolve/main/data/long.parquet

`data/` 已被 `.gitignore` 排除，語料不進版控。讀 parquet 需要 `pyarrow`，
那是本腳本的**操作者本機需求**，不是專案依賴 —— `[project] dependencies`
仍為空陣列，`pytest` 不需要它。

**座標對齊。** 辨識跑在 `normalize_text()` 之後的文字上（與 production 一致：
NFKC 把全形數字收斂為半形之後才辨識得到），而 gold span 的索引是**原文**的。
910 題中有 9 題正規化會改變長度，所以不能假設兩套索引相同 ——
以 `NormalizedText.offsets` 把命中區間映回原文座標再比對。

**判定準則為區間重疊，不是邊界完全相同。** gold 的 `private_phone` 區間
是否含分隔符、是否含「電話：」前綴，資料集沒有承諾；要求邊界完全相同會把
邊界差一格的正確命中算成誤判，而本量測要回答的問題是「遮到的是不是個資」。

**輸出兩個 precision，因為「算錯」有兩種而代價差很多。**

- **嚴格 precision** —— 命中的區間對上一個**同類**的 gold span。
  這是與已發表數字唯一可比的欄位。
- **誤遮率** —— 命中的區間**完全不重疊任何 gold span**，也就是遮掉了資料集
  認為不是個資的字。這才是本層失敗模式的直接量測，也是
  `add-redact-apply` 用來估「log 裡有幾個無謂的 placeholder」的那個數字。

兩者之間的差額是**標籤錯位**：實測中 `TW_ID` 的多數「誤判」是駕照號碼與
軍人補給證號 —— 它們與國民身分證共用 `[A-Z][12]\\d{8}` 的形狀而且通過同一個
checksum，在字元層無法區分。遮掉它們不是 over-redaction，遮掉的確實是個資，
只是類型標籤說錯了。把這兩種錯混成一個數字會讓這一層看起來比實際危險。
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from scam_guard import pii
from scam_guard.normalize import normalize_text

DATA_DIR = Path("data/tw-pii-bench")
SPLITS = ("short", "mid", "long")

# 本專案的封閉集合 → tw-PII-bench 的 gold 標籤。
#
# 資料集的 19 個標籤裡**沒有信用卡**，所以 `CREDIT_CARD` 的 gold 數為零：
# 它命中什麼都是誤判，recall 無從計算。這不是資料集的缺陷，是兩邊的封閉集合
# 本來就不相同 —— 誠實記下來，不要用「沒有可比的 gold」把它從表裡藏掉。
#
# 手機與市話在資料集裡同屬 `private_phone`，不細分。
GOLD_LABELS: dict[str, tuple[str, ...]] = {
    pii.TW_ID: ("tw_national_id",),
    pii.TW_MOBILE: ("private_phone",),
    pii.TW_LANDLINE: ("private_phone",),
    pii.CREDIT_CARD: (),
}

IN_SCOPE_LABELS: tuple[str, ...] = ("tw_national_id", "private_phone")
"""封閉集合涵蓋得到的 gold 標籤。recall 的分母只算這些，不算 77 類全體。"""

PUBLISHED_WEIGHTED_F1 = (
    ("openai/privacy-filter（基礎模型）", 54.1),
    ("lianghsun/privacy-filter-tw", 72.7),
)
"""已發表的加權平均 F1（%）。來源為**模型卡**，非本專案量測，且無第三方複現。"""


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _load_rows() -> list[dict]:
    """讀入三個 split 的全部題目。缺檔時 raise 並指名路徑，不靜默略過。"""
    rows: list[dict] = []
    for split in SPLITS:
        path = DATA_DIR / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"tw-PII-bench 檔案不存在：{path}（取得方式見本模組 docstring）"
            )
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _predictions(text: str) -> list[tuple[int, int, str]]:
    """回傳 `(原文起點, 原文終點, 類型)`，索引已由正規化座標映回原文座標。"""
    norm = normalize_text(text)
    return [
        (norm.offsets[span.start], norm.offsets[span.end], span.entity_type)
        for span in pii.find_pii(norm.text)
    ]


def _is_true_positive(prediction: tuple[int, int, str], gold_spans: Sequence[dict]) -> bool:
    start, end, entity_type = prediction
    labels = GOLD_LABELS[entity_type]
    return any(
        gold["label"] in labels and _overlaps(start, end, gold["start"], gold["end"])
        for gold in gold_spans
    )


def _hits_any_gold(prediction: tuple[int, int, str], gold_spans: Sequence[dict]) -> bool:
    """命中的區間是否碰到**任何**類型的 gold span，不論標籤是否對得上。"""
    start, end, _ = prediction
    return any(_overlaps(start, end, gold["start"], gold["end"]) for gold in gold_spans)


def _is_covered(gold: dict, predictions: Sequence[tuple[int, int, str]]) -> bool:
    return any(
        gold["label"] in GOLD_LABELS[entity_type]
        and _overlaps(start, end, gold["start"], gold["end"])
        for start, end, entity_type in predictions
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _percent(value: float, denominator: int) -> str:
    if denominator == 0:
        return "  n/a"
    return f"{value * 100:5.1f}%"


def main() -> int:
    rows = _load_rows()

    hits = dict.fromkeys(pii.ENTITY_TYPES, 0)
    correct = dict.fromkeys(pii.ENTITY_TYPES, 0)
    over_redacted = dict.fromkeys(pii.ENTITY_TYPES, 0)
    gold_total = dict.fromkeys(IN_SCOPE_LABELS, 0)
    gold_covered = dict.fromkeys(IN_SCOPE_LABELS, 0)

    for row in rows:
        gold_spans = row["spans"]
        predictions = _predictions(row["text"])
        for prediction in predictions:
            hits[prediction[2]] += 1
            if _is_true_positive(prediction, gold_spans):
                correct[prediction[2]] += 1
            if not _hits_any_gold(prediction, gold_spans):
                over_redacted[prediction[2]] += 1
        for gold in gold_spans:
            if gold["label"] not in IN_SCOPE_LABELS:
                continue
            gold_total[gold["label"]] += 1
            if _is_covered(gold, predictions):
                gold_covered[gold["label"]] += 1

    total_hits = sum(hits.values())
    total_correct = sum(correct.values())
    total_over_redacted = sum(over_redacted.values())
    total_gold = sum(gold_total.values())
    total_covered = sum(gold_covered.values())
    precision = _ratio(total_correct, total_hits)
    recall = _ratio(total_covered, total_gold)
    f1 = _ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0

    print(f"tw-PII-bench：{len(rows)} 題（short / mid / long 三個 split）")
    print()
    print("四條 regex 的命中、嚴格 precision 與誤遮率")
    print(f"{'類型':<14}{'命中':>6}{'同類':>6}{'嚴格 precision':>16}{'誤遮':>6}{'誤遮率':>10}")
    for entity_type in pii.ENTITY_TYPES:
        count = hits[entity_type]
        print(
            f"{entity_type:<14}{count:>6}{correct[entity_type]:>6}"
            f"{_percent(_ratio(correct[entity_type], count), count):>16}"
            f"{over_redacted[entity_type]:>6}"
            f"{_percent(_ratio(over_redacted[entity_type], count), count):>10}"
        )
    print(
        f"{'整體':<14}{total_hits:>6}{total_correct:>6}{_percent(precision, total_hits):>16}"
        f"{total_over_redacted:>6}"
        f"{_percent(_ratio(total_over_redacted, total_hits), total_hits):>10}"
    )
    print()
    print("「同類」= 對上同類型的 gold span；「誤遮」= 完全不重疊任何 gold span。")
    print("兩者的差額是標籤錯位（駕照、軍人補給證號與國民身分證共用形狀與 checksum），")
    print("遮掉的仍然是個資，只是標籤說錯 —— 不計入誤遮。")
    print()
    print("⚠️ 「誤遮」欄是**上界**，不是誤遮數本身。首次執行時的 54 筆已逐一人工檢視，")
    print("   全部都是資料集未標註的真電話或真證號（公司客服專線、住家電話、分機前的")
    print("   市話）。也就是本語料上遮到**非個資**的次數為零，這一欄量到的是資料集的")
    print("   標註覆蓋率而不是我們的誤判率。數字若變動 MUST 重新人工檢視再引用。")
    print()
    print("封閉集合涵蓋得到的 gold 標籤上的 recall")
    print(f"{'gold 標籤':<18}{'總數':>6}{'抓到':>6}{'recall':>12}")
    for label in IN_SCOPE_LABELS:
        print(
            f"{label:<18}{gold_total[label]:>6}{gold_covered[label]:>6}"
            f"{_percent(_ratio(gold_covered[label], gold_total[label]), gold_total[label]):>12}"
        )
    print(f"{'整體':<18}{total_gold:>6}{total_covered:>6}{_percent(recall, total_gold):>12}")
    print()
    print(f"整體 F1（precision 與上方 recall 的調和平均）：{f1 * 100:.1f}%")
    print()
    print("與已發表的加權平均 F1 並列（來源為模型卡，非本專案量測）")
    for name, value in PUBLISHED_WEIGHTED_F1:
        print(f"  {name:<34}{value:>5.1f}%")
    print(f"  {'本專案四條 regex（整體 F1）':<30}{f1 * 100:>5.1f}%")
    print()
    print("⚠️ 三項不可比之處，讀這張表之前必須先讀：")
    print("  (a) 我們只做 4 類，benchmark 涵蓋 19 個標籤（模型宣稱 77 類）——")
    print("      **recall 必然很低，而且低是設計結果**，不是缺陷。把我們的 recall")
    print("      或 F1 拿去跟模型的加權平均 F1 比較是沒有意義的。")
    print("  (b) **可比的是 precision**，因為 over-redaction 是這一層唯一的失敗模式。")
    print("      對照的對象是模型卡自述的 hard-negative 誤報率 77.5%。")
    print("  (c) benchmark 為合成語料，與 Cofacts 真實訊息的分佈不同。")
    print("      `實驗結果.md` 的四個誤判字串仍是主要驗收依據，本表是補充不是取代。")
    print()
    print("仍不可宣稱：真實 Cofacts 語料上的 over-redaction 率仍不可量測，")
    print("那需要 `dev-data` 的樣本。`規劃.md` M5 驗收為**部分完成**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
