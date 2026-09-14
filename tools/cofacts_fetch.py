"""從 Cofacts GraphQL API 取得開發用的訊息語料，落地為 JSONL。

Cofacts 是目前唯一公開、可程式取得、且經人工查核的台灣中文訊息庫。
這個模組只負責**把資料取回來**：不做內容層的判斷（那屬 `add-message-filter`）、
不做人工標註（那屬 `add-testset`）、不做快取或增量更新（重跑就是更新）。

用法：

    python -m tools.cofacts_fetch --label scam --out data/cofacts_scam.jsonl
    python -m tools.cofacts_fetch --label hard-negative \\
        --out data/cofacts_hard_negative.jsonl

中斷後以失敗訊息給出的游標續抓：

    python -m tools.cofacts_fetch --label scam --out data/cofacts_scam.jsonl \\
        --after WzE3MjIxNDEzMTY3MDAsMTcy

**輸出不進版控。** 抓回來的是真實民眾送進 Cofacts 的訊息，含詐騙方的帳號與
受害者貼出的對話；且 Cofacts 開放資料為 CC BY-SA 4.0，與本專案的 MIT 授權衝突。
`--out` 的路徑若未被 git 忽略，本模組拒絕執行。姓名標示以每筆的 `source_uri` 滿足。

**失敗一律 raise，不重試、不降級、不部分成功。** 一趟只有約 104 次請求、
手動執行、重跑成本是秒級；重試的正確做法是一整套決定，寫它是拿複雜度換一個
不存在的問題。抓到一半的 JSONL 會被下游當成完整輸入，而沒有任何機制會發現它不完整。
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ENDPOINT = "https://api.cofacts.tw/graphql"

USER_AGENT = "scam-guard/0.1.0 (research)"
"""可識別本專案的 User-Agent。**不可為空** —— 實測不帶此標頭時伺服器回 403。"""

ARTICLE_URI_PREFIX = "https://cofacts.tw/article/"

PAGE_SIZE = 100

REQUEST_INTERVAL = 0.2
"""每頁之間的固定間隔（秒）。

Cofacts 是非營利組織的公開服務，沒有公布速率限制。104 次請求加總 21 秒的延遲
可忽略，而被封鎖的代價遠大於此。這是禮貌不是最佳化，沒有依據，也不值得調參。
"""

SELECTORS: dict[str, dict] = {
    "scam": {
        # 分類「詐騙」。查核結果 RUMOR 即人工查核為假訊息。
        "categoryIds": ["nD2n7nEBrIRcahlYwQoW"],
        "replyTypes": ["RUMOR"],
        "articleTypes": ["TEXT"],
    },
    "hard-negative": {
        # 分類「優惠措施、新法規、政策宣導」與「商業廣告」。
        # 查核結果 NOT_RUMOR（含有正確訊息）或 OPINIONATED（個人意見）。
        "categoryIds": ["mj2n7nEBrIRcahlYdArf", "nz2o7nEBrIRcahlYBgqQ"],
        "replyTypes": ["NOT_RUMOR", "OPINIONATED"],
        "articleTypes": ["TEXT"],
    },
}
"""兩組**寫死**的選取條件。2026-09-14 實測筆數：scam 7,110、hard-negative 3,256。

筆數會隨 Cofacts 成長，此處記的是量級不是事實斷言。

選取條件寫死而非開放成 CLI 參數是刻意的：selector 是這份資料的定義，不是使用者
偏好。開放它會讓 `data/cofacts_scam.jsonl` 這個檔名與它的內容脫鉤，而檔案不進
版控、沒有任何地方記錄它是用什麼條件抓的。要換條件就改這裡並在 commit 訊息裡說明。

`replyTypes` 是難負例品質的關鍵。不加它的話池子有 9,174 則，其中 2,506 則（27%）
查核為 RUMOR —— 那是**確實不實**的政策謠言，拿它當 ham 會汙染負例；另有 1,937 則
根本沒有查核回覆，沒有任何標籤依據。

**未納入的第三個池子，留給 `add-testset`：** 分類為「詐騙」但查核為 NOT_RUMOR /
OPINIONATED 的文字訊息，2026-09-14 實測 874 則 —— 民眾覺得像詐騙、主動送查、
結果查核為真，比商業廣告更貼近真正的誤判邊界。此處不抓，因為開發期手動檢視要的是
「常見的樣子」不是「最邊緣的樣子」。
"""

QUERY = (
    """
query FetchArticles($filter: ListArticleFilter, $after: String) {
  ListArticles(
    filter: $filter
    orderBy: [{ createdAt: DESC }]
    first: %d
    after: $after
  ) {
    totalCount
    edges {
      cursor
      node {
        id
        text
        createdAt
        articleType
        articleCategories { categoryId status }
        articleReplies { replyType status }
      }
    }
  }
}
"""
    % PAGE_SIZE
)
"""排序為 `createdAt: DESC`：開發期要看的是現在的詐騙型態，2016 年的訊息對規則設計
沒有參考價值，`--limit` 截斷時留下的才會是最新的那一段。

代價：DESC 排序下抓取期間新增的文章排在游標之前，本趟看不到，因此
`len(records) < totalCount` 是**正常**結果，`totalCount` 不能拿來斷言完整性
（拿它斷言會製造假警報）。完整性靠三條確定性的檢查撐住：游標必須前進、
`id` 不得重複、終止條件必須是空頁、達到 `--limit`、或取得的筆數達到第一頁的
`totalCount`（見 `fetch_selector`：把池子抓完之後 API 會回一個哨兵游標）。

**刻意不取 `pageInfo`** —— 它給的是整份篩選結果的首尾游標，不是本頁的，
拿它分頁會原地打轉。不取回來就不會有人誤用（見 `_collect_page`）。
"""


class CofactsFetchError(Exception):
    """抓取過程中任何導致整趟中止的狀況。訊息一律含足以定位問題的資訊。"""


def _build_request(payload: dict) -> urllib.request.Request:
    """組出一次 GraphQL 請求。標頭在此處集中決定，使 User-Agent 可被單獨測試。"""
    return urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )


def _post(payload: dict) -> tuple[int, str]:
    """送出一次請求，回傳 `(HTTP 狀態碼, 回應內容)`。

    非 2xx 由 `urllib` 以 `HTTPError` 表達，此處轉回狀態碼與回應內容 ——
    「狀態碼是多少」是呼叫端要據以組出錯誤訊息的資訊（它知道頁次），
    不是這裡能處理的事。測試以 monkeypatch 取代本函式，因此所有錯誤判定
    都在呼叫端而非此處。
    """
    request = _build_request(payload)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def _selector(label: str) -> dict:
    if label not in SELECTORS:
        raise ValueError(f"未知的標籤 {label!r}，可用的標籤為 {sorted(SELECTORS)}")
    return SELECTORS[label]


def _payload(selector: dict, cursor: str | None) -> dict:
    return {"query": QUERY, "variables": {"filter": selector, "after": cursor}}


def _raise_for_status(status: int, page: int, body: str) -> None:
    """狀態非 200 即中止。**只陳述事實，不推測原因** —— 臆測會讓人往錯的方向查。"""
    if status != 200:
        raise CofactsFetchError(
            f"第 {page} 頁的 HTTP 狀態為 {status}，非 200。"
            f"送出的 User-Agent 為 {USER_AGENT!r}。"
            f"回應內容前 200 字元：{body[:200]!r}"
        )


def _parse_listing(body: str, page: int) -> dict:
    """把回應內容解析成 `ListArticles` 物件，任何缺漏或錯誤即中止。

    **API 改版不做偵測。** GraphQL 沒有版本號可以比對，而猜測式的 schema 檢查
    （「有新欄位就用新的、否則用舊的」）正是被禁止的臆測性 fallback。改版會以
    這裡的「缺欄位」或「errors 非空」在第一頁就炸掉，訊息裡有欄位名。
    """
    payload = json.loads(body)
    if "errors" in payload and payload["errors"]:
        first = payload["errors"][0]
        message = first["message"] if "message" in first else "(回應未帶 message 欄位)"
        path = first["path"] if "path" in first else "(回應未帶 path 欄位)"
        raise CofactsFetchError(
            f"第 {page} 頁的 GraphQL 回應含錯誤：message={message!r}、path={path!r}"
        )
    if "data" not in payload or not isinstance(payload["data"], dict):
        raise CofactsFetchError(f"第 {page} 頁的回應缺少 data 物件：{body[:200]!r}")
    data = payload["data"]
    if "ListArticles" not in data:
        raise CofactsFetchError(f"第 {page} 頁的 data 缺少欄位 ListArticles")
    listing = data["ListArticles"]
    for field_name in ("totalCount", "edges"):
        if field_name not in listing:
            raise CofactsFetchError(f"第 {page} 頁的 ListArticles 缺少欄位 {field_name}")
    if not isinstance(listing["edges"], list):
        raise CofactsFetchError(f"第 {page} 頁的 ListArticles.edges 不是陣列：{listing['edges']!r}")
    return listing


def _normal_values(items: list, key: str, field_name: str, article_id: str, page: int) -> list[str]:
    """取出 `status == "NORMAL"` 的項目的 `key` 值。

    Cofacts 的分類與查核回覆都可被撤銷，撤銷後仍留在回應裡但 `status` 為
    `DELETED`。把撤銷的查核當成有效標籤，等於用已被推翻的結論當依據。
    """
    if not isinstance(items, list):
        raise CofactsFetchError(
            f"第 {page} 頁的 node id={article_id} 的 {field_name} 不是陣列：{items!r}"
        )
    values: list[str] = []
    for item in items:
        for required in (key, "status"):
            if required not in item:
                raise CofactsFetchError(
                    f"第 {page} 頁的 node id={article_id} 的 {field_name} 有一項缺少欄位 {required}"
                )
        if item["status"] == "NORMAL":
            values.append(item[key])
    return values


def _to_record(node: dict, label: str, fetched_at: str, page: int) -> dict:
    """把一個 GraphQL node 組成一行落地紀錄。

    每一行都重複 `label` 與 `fetched_at`：這個檔案的主要用法是
    `grep 監管帳戶 data/*.jsonl` 然後看那一行，抽出來的那一行必須自己交代
    它是什麼、哪裡來的。重複的代價是幾十 KB。

    `reply_types` 保留而不是丟掉，即使 selector 已經篩過一次 —— 難負例是這個
    專案的關鍵資產，「這則為什麼算負例」必須在資料裡看得見。
    """
    if "id" not in node:
        raise CofactsFetchError(
            f"第 {page} 頁有一筆 node 缺少欄位 id；該 node 的欄位為 {sorted(node)}"
        )
    article_id = node["id"]
    for field_name in ("text", "createdAt", "articleCategories", "articleReplies"):
        if field_name not in node:
            raise CofactsFetchError(f"第 {page} 頁的 node id={article_id} 缺少欄位 {field_name}")
    return {
        "id": article_id,
        "text": node["text"],
        "created_at": node["createdAt"],
        "label": label,
        "category_ids": _normal_values(
            node["articleCategories"], "categoryId", "articleCategories", article_id, page
        ),
        "reply_types": _normal_values(
            node["articleReplies"], "replyType", "articleReplies", article_id, page
        ),
        "source_uri": ARTICLE_URI_PREFIX + article_id,
        "fetched_at": fetched_at,
    }


def _collect_page(
    selector: dict,
    cursor: str | None,
    page: int,
    label: str,
    fetched_at: str,
    seen: set[str],
    remaining: int | None,
) -> tuple[list[dict], str | None, int]:
    """取一頁，回傳 `(本頁紀錄, 下一頁的游標, totalCount)`。edges 為空時游標為 `None`。

    **下一頁的游標取自 `edges[-1]["cursor"]`，不是 `pageInfo.lastCursor`。**
    後者是整份篩選結果最後一筆（2016 年的那一筆）的游標，不是本頁的；拿它當
    `after` 傳回去，第二頁只會回一筆且游標不再變化 —— 迴圈永遠不結束，
    或以為抓完了而靜默截短。實測第一次就是這樣拿到「103 筆」的假結果。
    因此 `QUERY` 根本不取 `pageInfo`，而「游標必須前進」是對此的保險。
    """
    status, body = _post(_payload(selector, cursor))
    _raise_for_status(status, page, body)
    listing = _parse_listing(body, page)
    edges = listing["edges"]
    if not edges:
        return [], None, listing["totalCount"]

    last_edge = edges[-1]
    if "cursor" not in last_edge:
        raise CofactsFetchError(f"第 {page} 頁最後一個 edge 缺少欄位 cursor")
    next_cursor = last_edge["cursor"]
    if next_cursor == cursor:
        raise CofactsFetchError(f"第 {page} 頁的游標未前進，仍為 {next_cursor!r}；分頁無法繼續")

    return (
        _records(edges, label, fetched_at, seen, remaining, page),
        next_cursor,
        listing["totalCount"],
    )


def _records(
    edges: list,
    label: str,
    fetched_at: str,
    seen: set[str],
    remaining: int | None,
    page: int,
) -> list[dict]:
    """把一頁的 edges 轉成落地紀錄。重複的 article id 即中止。"""
    records: list[dict] = []
    for edge in edges:
        if remaining is not None and len(records) >= remaining:
            break
        if "node" not in edge:
            raise CofactsFetchError(f"第 {page} 頁有一個 edge 缺少欄位 node")
        record = _to_record(edge["node"], label, fetched_at, page)
        if record["id"] in seen:
            raise CofactsFetchError(
                f"第 {page} 頁出現重複的 article id {record['id']!r}；分頁已取得同一筆兩次"
            )
        seen.add(record["id"])
        records.append(record)
    return records


def _resume_hint(command: str, cursor: str | None) -> str:
    """失敗訊息附上的命令，可直接複製貼回命令列。

    續抓是**明確指定的**，不是自動 fallback —— 人看到錯誤、人決定續抓、人貼游標。
    系統沒有在任何地方靜默地改變行為。
    """
    if cursor is None:
        return f"整趟抓取中止。重跑命令：\n  {command}"
    return f"整趟抓取中止。續抓命令（自本頁的起始游標接續）：\n  {command} --after {cursor}"


def fetch(label: str, out_path: Path, limit: int | None = None, after: str | None = None) -> int:
    """走完分頁把結果寫進 `out_path`，回傳寫入筆數。任一頁失敗即整趟失敗。

    `after` 指定時以 append 模式開檔（續抓接在既有內容之後），未指定時覆寫。
    每頁結束即 flush，使中斷時已寫入的部分都是完整的行。

    續抓可能產生重複的 `id`（若續抓點選得比中斷點早）。**不在寫入端擋** ——
    寫入端沒有前一趟的記憶，要擋就得先讀回整個檔案。去重由
    `tools.message_filter` 在讀取時做，成本在那裡趨近於零。
    """
    return fetch_selector(
        _selector(label),
        label,
        out_path,
        f"python -m tools.cofacts_fetch --label {label} --out {out_path}",
        limit=limit,
        after=after,
    )


def fetch_selector(
    selector: dict,
    label: str,
    out_path: Path,
    resume_command: str,
    *,
    limit: int | None = None,
    after: str | None = None,
) -> int:
    """以一組**顯式傳入**的選取條件走完分頁。`fetch()` 是它的具名 selector 版本。

    第二個消費者是 `tools/eval/build_testset.py`：測試集的三個 Cofacts 子集各有
    自己的 selector（定義於 `tools/eval/selectors.py`，因為 selector 是那份
    資料集的定義），但分頁、游標前進檢查、重複 id 偵測與失敗訊息只該有一份實作。

    `resume_command` 由呼叫端提供，因為失敗訊息裡那行命令必須是**呼叫端自己**
    的命令 —— 印一行指向本模組的續抓指令給一個用別組 selector 的呼叫端，
    是一個看起來有幫助但貼上去會失敗的訊息。

    **抓到第一頁回報的 `totalCount` 為止即停止。** 這一條是把整個池子抓完之後
    實測出來的：7,113 筆抓完（第 71 頁只有 13 筆）之後，該頁最後一個 edge 的
    游標是 `Wy05MjIzMzcyMDM2ODU0Nzc2MDAwLDM0NTYwXQ==`，解碼後含
    `-9223372036854776000`（`Long.MIN_VALUE` 附近的哨兵值），拿它續抓時後端以
    `parse_exception: failed to parse date field` 回錯。開發期只抓六頁，
    這個邊界從未被觸及。

    目標值取**第一頁**的 `totalCount` 而不是每頁重讀：DESC 排序下，抓取期間
    新增的文章排在游標之前，本趟本來就看不到它們，而 `totalCount` 會把它們算進去 ——
    每頁重讀會讓終止條件追著一個永遠達不到的數字跑。
    """
    fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    seen: set[str] = set()
    total = 0
    page = 0
    cursor = after
    pool_size: int | None = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a" if after else "w", encoding="utf-8") as stream:
        while True:
            remaining = None if limit is None else limit - total
            try:
                records, next_cursor, total_count = _collect_page(
                    selector, cursor, page, label, fetched_at, seen, remaining
                )
            except CofactsFetchError as error:
                raise CofactsFetchError(
                    f"{error}\n{_resume_hint(resume_command, cursor)}"
                ) from error

            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            total += len(records)
            print(
                f"第 {page} 頁：本頁 {len(records)} 筆、累計 {total} 筆、"
                f"totalCount {total_count}、下一頁游標 {next_cursor!r}",
                file=sys.stderr,
            )

            if pool_size is None:
                pool_size = total_count
            if next_cursor is None:
                break
            if limit is not None and total >= limit:
                break
            if total >= pool_size:
                break
            cursor = next_cursor
            page += 1
            time.sleep(REQUEST_INTERVAL)
    return total


def _require_git_ignored(path: Path) -> None:
    """輸出路徑必須被 git 忽略。以 `git check-ignore` 實測，不目視確認 `.gitignore`。

    目視確認過的 `.gitignore` 被別的 change 改壞是可能的，而後果是把真實民眾訊息
    推上遠端。
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
        raise CofactsFetchError(
            f"輸出路徑 {path} 未被 git 忽略。抓取結果是真實民眾送進 Cofacts 的訊息，"
            f"且 Cofacts 開放資料為 CC BY-SA 4.0，與本專案的 MIT 授權衝突，"
            f"不得進版控。請改用 data/ 之下的路徑。"
        )
    raise CofactsFetchError(
        f"git check-ignore 無法判定 {path}："
        f"returncode={result.returncode}、stderr={result.stderr.strip()!r}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.cofacts_fetch",
        description="從 Cofacts 取得開發用的訊息語料，輸出 JSONL。",
    )
    parser.add_argument("--label", required=True, help=f"選取條件，可用：{sorted(SELECTORS)}")
    parser.add_argument("--out", required=True, help="輸出的 JSONL 路徑，必須被 git 忽略")
    parser.add_argument("--limit", type=int, default=None, help="最多取得的筆數")
    parser.add_argument("--after", default=None, help="起始游標，用於中斷後續抓")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    try:
        _require_git_ignored(out_path)
        total = fetch(args.label, out_path, args.limit, args.after)
    except (CofactsFetchError, ValueError) as error:
        print(f"抓取失敗：{error}", file=sys.stderr)
        return 1
    print(f"完成：{args.label} 共 {total} 筆，寫入 {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
