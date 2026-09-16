"""下載對話語料快照，過內容閘門，落地為對話 ham 的候選單則。

執行：`python -m tools.fetch_conversation_corpus [--reuse-download]`

**外部格式的知識全部在這個檔案裡** —— 下載網址、CSV 欄位、逐則抽取、內容閘門
的過濾規則、長度下限、去重。`scam_guard/` 不知道對話語料如何取得，`tools/eval/`
只認識落地後的 JSON Lines。

## 兩個閘門，順序不能顛倒：先內容，再授權

一份授權再乾淨的語料，若內容不是口語對話，就修不了「你好被判詐騙」這個問題 ——
它只會教分類器別的東西。所以 working corpus 先過內容閘門（MUST 是口語對話），
再談授權。

**working corpus：`zake7749/Gossiping-Chinese-Corpus`**（PTT 八卦版問答，
原生台灣繁中，Apache-2.0，GitHub raw 直接下載、非 gated）。抽「答」（推文）為
ham 單則 —— 推文是最接近使用者實際會打的口語短句。

## 內容閘門：三道過濾 + 長度下限 + 精確去重

1. 丟 `沒有資料` 標記的筆（無回覆的佔位，非對話）。
2. 丟被**規則層 `hard=True` 或 165 黑名單**命中的候選 —— 以 `tools/eval/run` 的
   規則與 URL registry（**不含 n-gram 分類器**）跑過每則，命中即丟。
3. 丟明確詐騙標記詞表（`SCAM_MARKER_TERMS`）命中者。

**MUST NOT 以 n-gram 分類器過濾候選。** 那會循環，而且會恰好丟掉「你好」這類
**最想留下的 hard-negative** —— 分類器現在正把它們判成詐騙，用它過濾等於把要修的
東西先刪掉。八卦版的髒話、政治、情緒性內容**不丟**：真實人類雜訊正是要的 ham 分布；
只有極短/純符號（`MIN_CONTENT_CHARS`）以長度下限處置。

## 原生台灣繁中，MUST NOT 用 OpenCC

working corpus 已是台灣繁中，轉換只會引入雜訊。OpenCC 是**來源條件式**的一步，
只在啟用簡體備援（CrossWOZ / KdConv）時才以 `s2twp` 全量轉繁 —— 本 change 的量
在 Gossiping 上充足，備援不啟用，因此本檔不 import `opencc`（見 design 與
`pyproject` 的 `eval` extra 說明）。`manifest` 記錄 `converted = false`。

## 語料不進版控

輸出寫入 `data/conversation/`（已 gitignored）。`manifest` 記錄語料的
release/commit，使第三人可重建同一份輸入。切分不在此檔 —— 由
`tools/eval/dataset.py` 依 `sha256(正規化文字)` 首位元組確定性切分。
"""

import argparse
import csv
import json
import operator
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from scam_guard.check import CheckRegistry
from scam_guard.ngram import document_text
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.pipeline import detect
from scam_guard.types import Message, Request
from scam_guard.weights import WeightTable, load_weights
from tools.eval.run import build_registry, load_blocklist, load_psl

CORPUS_NAME = "Gossiping-Chinese-Corpus"
CORPUS_REPO = "zake7749/Gossiping-Chinese-Corpus"
CORPUS_COMMIT = "65b7e3630a560223a2b4d702d78d120d5ff1e8dd"
"""pinned commit（2024-10-18）。第三人以此 commit 重跑得到同一份輸入。"""

CORPUS_DATA_PATH = "data/Gossiping-QA-Dataset-2_0.csv"
RAW_URL = f"https://raw.githubusercontent.com/{CORPUS_REPO}/{CORPUS_COMMIT}/{CORPUS_DATA_PATH}"
CORPUS_LICENSE = "Apache-2.0（作者 zake7749 以此授權釋出彙整資料集）"
CONVERTED_TO_TRADITIONAL = False
"""是否經 OpenCC 轉繁。working corpus 原生台灣繁中，故為 False。"""

NO_DATA_MARKER = "沒有資料"
"""無回覆的佔位標記（Gossiping 中約 1,078 筆答為此值），非對話，丟。"""

MIN_CONTENT_CHARS = 4
"""正規化後最短長度。**沒有實測依據，是一個起點** —— 用以濾掉「推」「XD」這類
極短或純符號的推文，它們不承載可用的口語 n-gram。八卦版的髒話與情緒性內容
**不**被此規則丟，那是要保留的人類雜訊。"""

SCAM_MARKER_TERMS: tuple[str, ...] = (
    "監管帳戶",
    "保證獲利",
    "穩賺",
    "包賺",
    "加賴",
    "加 line",
    "私訊領取",
    "私訊我",
    "投資群組",
    "帶你賺",
    "老師帶單",
    "內線",
    "博弈",
    "點數卡",
    "解除分期",
    "解除設定",
    "monitored",
)
"""明確詐騙標記詞表，與規則層詞表分開維護。八卦版偶有轉貼詐騙截圖文字或招攬，
命中者不進 ham。**這不是規則層** —— 規則層與 165 黑名單的命中由 `_rule_gate`
另行處理（第二道過濾），本詞表是第三道，補規則層擋不住的招攬用語。"""

TARGET = 800
"""對話 ham 的目標數量。對齊 `cofacts_ham_ad` 的原始 `target`，使分類器的 ham
曝光在「正式宣導/廣告文」與「口語對話」兩語域間平衡。去重後確定性切分約 400/400。"""

DEFAULT_OUT_DIR = Path("data/conversation")
CACHE_FILENAME = "gossiping_qa.csv"
SUBSET_FILENAME = "conversation_ham.jsonl"
MANIFEST_FILENAME = "manifest.json"

TIMEOUT_SECONDS = 300.0
USER_AGENT = "scam-guard/fetch_conversation_corpus"


class FetchError(Exception):
    """取得或過濾失敗。訊息一律含足以定位問題的資訊。"""


@dataclass(frozen=True)
class Candidate:
    """一則通過廉價過濾與去重的候選。

    `select_key` 是**加鹽雜湊**，用於確定性排序：以它取前 N 個時，其未加鹽的
    內容雜湊（`id`）首位元組仍近似均勻，切分因此仍約 50/50。兩個雜湊獨立是刻意的
    —— 若排序鍵與切分鍵同源，取「雜湊最小的前 N 個」會讓首位元組全部偏小、
    全部落進 tune。
    """

    select_key: str
    id: str
    text: str


def download(url: str, dest: Path, *, reuse: bool) -> None:
    """下載語料到 `dest`。`reuse` 為真且檔案已存在時不重抓（人明確指定的旗標）。"""
    if reuse and dest.is_file():
        print(f"重用既有下載：{dest}", file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise FetchError(f"下載 {url} 失敗：回應狀態碼 {response.status}")
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise FetchError(f"下載 {url} 失敗：回應狀態碼 {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"下載 {url} 失敗：連線錯誤 {exc.reason}") from exc
    dest.write_bytes(payload)
    print(f"下載完成：{dest}（{len(payload)} B）", file=sys.stderr)


def normalized(text: str) -> str:
    """與 `detect()` 完全相同的一條正規化路徑。去重與切分皆以此為準。"""
    return document_text(build_document([Message(text=text)], DEFAULT_LIMITS))


def content_id(normalized_text: str) -> str:
    """正規化文字的 sha256 十六進位。**同時是去重鍵與切分鍵的來源** ——
    `dataset.Sample.split` 讀它的首位元組（前兩碼）決定 tune/holdout。"""
    return sha256(normalized_text.encode("utf-8")).hexdigest()


def select_key(normalized_text: str) -> str:
    """加鹽雜湊的排序鍵。與 `content_id` 獨立，使取前 N 個不偏斜切分。"""
    return sha256(("select:" + normalized_text).encode("utf-8")).hexdigest()


def has_scam_marker(text: str) -> bool:
    """明確詐騙標記詞表命中（大小寫不敏感，涵蓋 `加 line` 這類含英文的招攬用語）。"""
    lowered = text.lower()
    return any(term.lower() in lowered for term in SCAM_MARKER_TERMS)


def iter_answers(csv_path: Path) -> list[str]:
    """讀出「答」欄的全部原文。欄位缺失即 raise 並印出實際標頭。"""
    with csv_path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise FetchError(f"語料為空：{csv_path}")
        if header[:2] != ["question", "answer"]:
            raise FetchError(
                f"語料標頭不符：{header!r}，預期以 ['question', 'answer'] 開頭（{csv_path}）"
            )
        answers = [row[1] for row in reader if len(row) >= 2]
    if not answers:
        raise FetchError(f"語料未解析出任何「答」：{csv_path}")
    return answers


def cheap_candidates(answers: list[str]) -> list[Candidate]:
    """廉價過濾（`沒有資料`、長度、詐騙標記詞表）+ 精確去重，回傳去重後的候選。

    去重以正規化文字為準：正規化後完全相同的句子只留一則，避免同一句跨 tune/holdout
    造成洩漏。規則層/165 黑名單這道**昂貴**的過濾不在此處，留給 `_rule_gate` 逐則跑。
    """
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for answer in answers:
        stripped = answer.strip()
        if stripped == NO_DATA_MARKER:
            continue
        if has_scam_marker(stripped):
            continue
        text = normalized(stripped)
        if len(text) < MIN_CONTENT_CHARS:
            continue
        identifier = content_id(text)
        if identifier in seen:
            continue
        seen.add(identifier)
        candidates.append(Candidate(select_key=select_key(text), id=identifier, text=stripped))
    return candidates


def _rule_gate(registry: CheckRegistry, table: WeightTable, text: str) -> bool:
    """該候選是否被**規則層 `hard=True` 或 165 黑名單**命中。命中即應丟。

    用的是 `tools/eval/run.build_registry` 的規則與 URL registry，**不含 n-gram
    分類器**（非循環過濾）。`short_circuit=False` 使全部檢查都跑，不因先命中而略過。
    """
    verdict = detect(Request.from_text(text), registry, table, short_circuit=False)
    blocklist_name = table.roles["blocklist_exact"]
    for result in verdict.checks:
        if not result.hit:
            continue
        if result.hard or result.name == blocklist_name:
            return True
    return False


def select_ham(candidates: list[Candidate], target: int) -> tuple[list[Candidate], int]:
    """以確定性排序逐則跑昂貴的規則閘門，收集到 `target` 則為止。

    回傳 `(選中的候選, 被規則閘門丟掉的則數)`。昂貴閘門只跑到湊滿 target，
    不對全部候選跑 —— 774k 則跑 `detect()` 不切實際，而確定性排序保證每次跑到的
    是同一批候選。
    """
    ordered = sorted(candidates, key=operator.attrgetter("select_key"))
    psl = load_psl(Path("data/psl"))
    registry = build_registry(psl, store=load_blocklist(Path("data/blocklist"), psl))
    table = load_weights()
    selected: list[Candidate] = []
    rejected = 0
    for candidate in ordered:
        if _rule_gate(registry, table, candidate.text):
            rejected += 1
            continue
        selected.append(candidate)
        if len(selected) >= target:
            break
    return selected, rejected


def write_subset(out_dir: Path, selected: list[Candidate]) -> Path:
    """寫出 `conversation_ham.jsonl`，依 `id` 排序使兩次執行位元組相同。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SUBSET_FILENAME
    rows = sorted(selected, key=operator.attrgetter("id"))
    with path.open("w", encoding="utf-8") as stream:
        for candidate in rows:
            stream.write(
                json.dumps({"id": candidate.id, "text": candidate.text}, ensure_ascii=False) + "\n"
            )
    return path


def write_manifest(
    out_dir: Path,
    subset_path: Path,
    *,
    n_answers: int,
    n_cheap: int,
    n_rule_rejected: int,
    n_selected: int,
) -> Path:
    """寫出語料 manifest：pinning、過濾與去重規則、各階段筆數與子集 sha256。"""
    payload = subset_path.read_bytes()
    manifest = {
        "corpus": CORPUS_NAME,
        "repo": CORPUS_REPO,
        "commit": CORPUS_COMMIT,
        "data_path": CORPUS_DATA_PATH,
        "raw_url": RAW_URL,
        "license": CORPUS_LICENSE,
        "converted_to_traditional": CONVERTED_TO_TRADITIONAL,
        "opencc_config": None,
        "extraction": "抽「答」（PTT 推文）為 ham 單則",
        "content_gate": {
            "no_data_marker": NO_DATA_MARKER,
            "min_content_chars": MIN_CONTENT_CHARS,
            "scam_marker_terms": list(SCAM_MARKER_TERMS),
            "rule_layer": "規則層 hard=True 或 165 黑名單命中者（非 n-gram 分類器）",
            "dedup": "正規化文字精確去重（sha256 相同者只留一則）",
        },
        "counts": {
            "raw_answers": n_answers,
            "after_cheap_gate_deduped": n_cheap,
            "rule_gate_rejected": n_rule_rejected,
            "selected": n_selected,
            "target": TARGET,
        },
        "subset_file": SUBSET_FILENAME,
        "subset_sha256": sha256(payload).hexdigest(),
    }
    path = out_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.fetch_conversation_corpus",
        description="取得對話語料、過內容閘門、落地為對話 ham 候選（不進版控）。",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument(
        "--reuse-download",
        action="store_true",
        help="快取檔已存在時不重抓。人明確指定，不自動判斷。",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    cache_path = out_dir / CACHE_FILENAME
    try:
        download(RAW_URL, cache_path, reuse=args.reuse_download)
        answers = iter_answers(cache_path)
        print(f"[data] 抽出「答」{len(answers)} 則", file=sys.stderr)
        candidates = cheap_candidates(answers)
        print(f"[gate] 廉價過濾 + 去重後 {len(candidates)} 則", file=sys.stderr)
        selected, rejected = select_ham(candidates, args.target)
        if len(selected) < args.target:
            raise FetchError(
                f"通過內容閘門的對話 ham 只有 {len(selected)} 則，少於目標 {args.target}。"
                f"working corpus 量不足 —— 依 design 此時才啟用簡體備援（CrossWOZ/KdConv）"
                f"並加 OpenCC s2twp，本 change 不預先實作那條路徑。"
            )
        subset_path = write_subset(out_dir, selected)
        manifest_path = write_manifest(
            out_dir,
            subset_path,
            n_answers=len(answers),
            n_cheap=len(candidates),
            n_rule_rejected=rejected,
            n_selected=len(selected),
        )
    except FetchError as error:
        print(f"失敗：{error}", file=sys.stderr)
        return 1
    print(
        f"[out] {subset_path}（{len(selected)} 則，規則閘門丟 {rejected} 則）；"
        f"manifest {manifest_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
