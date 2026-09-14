"""`tools.cofacts_fetch` 的離線測試。

**全部離線。** 任何測試都不得發出網路請求 —— `_post` 一律以 monkeypatch 取代。
外部服務會變、會慢、會擋，而這些測試要驗的是分頁邏輯與失敗處理，不是 Cofacts。
"""

import json

import pytest

from tools import cofacts_fetch


class FakePost:
    """依序回傳預先準備好的回應，並記下每次送出的 payload。

    回應用完仍被呼叫即失敗 —— 「多打了一次」正是 `--limit` 與終止條件要驗的事。
    """

    def __init__(self, responses: list[tuple[int, str]]) -> None:
        self._responses = list(responses)
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> tuple[int, str]:
        self.payloads.append(payload)
        if not self._responses:
            raise AssertionError(f"_post 被呼叫第 {len(self.payloads)} 次，超過預期的回應數")
        return self._responses.pop(0)

    @property
    def afters(self) -> list[str | None]:
        return [payload["variables"]["after"] for payload in self.payloads]

    @property
    def filters(self) -> list[dict]:
        return [payload["variables"]["filter"] for payload in self.payloads]


class SleepRecorder:
    """取代 `time.sleep`，記下每次的秒數，使測試不真的等待。"""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _allow_any_path(path) -> None:
    """monkeypatch 用的替身：tmp_path 不在 repo 內，git check-ignore 會判定未被忽略。"""


def _node(article_id: str, text: str = "內容") -> dict:
    return {
        "id": article_id,
        "text": text,
        "createdAt": "2026-09-12T10:11:59.220Z",
        "articleType": "TEXT",
        "articleCategories": [{"categoryId": "nD2n7nEBrIRcahlYwQoW", "status": "NORMAL"}],
        "articleReplies": [{"replyType": "RUMOR", "status": "NORMAL"}],
    }


def _edges(prefix: str, article_ids: list[str]) -> list[dict]:
    return [
        {"cursor": f"{prefix}{index}", "node": _node(article_id)}
        for index, article_id in enumerate(article_ids)
    ]


def _response(
    edges: list[dict], total: int = 500, last_cursor: str | None = None
) -> tuple[int, str]:
    listing: dict = {"totalCount": total, "edges": edges}
    if last_cursor is not None:
        listing["pageInfo"] = {"lastCursor": last_cursor}
    return 200, json.dumps({"data": {"ListArticles": listing}})


def _ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(count)]


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> SleepRecorder:
    recorder = SleepRecorder()
    monkeypatch.setattr(cofacts_fetch.time, "sleep", recorder)
    return recorder


@pytest.fixture
def out_path(tmp_path):
    return tmp_path / "out.jsonl"


def _install(monkeypatch: pytest.MonkeyPatch, responses: list[tuple[int, str]]) -> FakePost:
    fake = FakePost(responses)
    monkeypatch.setattr(cofacts_fetch, "_post", fake)
    return fake


# --- 選取條件 -------------------------------------------------------------


def test_scam_filter(monkeypatch, out_path, sleeps):
    fake = _install(monkeypatch, [_response([])])
    cofacts_fetch.fetch("scam", out_path)
    assert fake.filters[0]["categoryIds"] == ["nD2n7nEBrIRcahlYwQoW"]
    assert fake.filters[0]["replyTypes"] == ["RUMOR"]


def test_hard_negative_filter(monkeypatch, out_path, sleeps):
    fake = _install(monkeypatch, [_response([])])
    cofacts_fetch.fetch("hard-negative", out_path)
    assert fake.filters[0]["categoryIds"] == [
        "mj2n7nEBrIRcahlYdArf",
        "nz2o7nEBrIRcahlYBgqQ",
    ]
    assert fake.filters[0]["replyTypes"] == ["NOT_RUMOR", "OPINIONATED"]


@pytest.mark.parametrize("label", ["scam", "hard-negative"])
def test_only_text_articles(monkeypatch, out_path, sleeps, label):
    fake = _install(monkeypatch, [_response([])])
    cofacts_fetch.fetch(label, out_path)
    assert fake.filters[0]["articleTypes"] == ["TEXT"]


def test_unknown_label(out_path):
    with pytest.raises(ValueError) as excinfo:
        cofacts_fetch.fetch("spam", out_path)
    assert "spam" in str(excinfo.value)
    assert "hard-negative" in str(excinfo.value)


# --- 請求標頭 -------------------------------------------------------------


def test_user_agent_header():
    request = cofacts_fetch._build_request({"query": "{}"})
    user_agent = request.get_header("User-agent")
    assert user_agent
    assert "scam-guard" in user_agent


# --- 分頁 -----------------------------------------------------------------


def test_three_pages(monkeypatch, out_path, sleeps):
    fake = _install(
        monkeypatch,
        [
            _response(_edges("a", _ids("a", 100))),
            _response(_edges("b", _ids("b", 40))),
            _response([]),
        ],
    )
    total = cofacts_fetch.fetch("scam", out_path)
    assert total == 140
    assert fake.afters == [None, "a99", "b39"]
    assert sleeps.calls == [cofacts_fetch.REQUEST_INTERVAL] * 2


def test_page_info_last_cursor_is_ignored(monkeypatch, out_path, sleeps):
    """`pageInfo.lastCursor` 誤用的迴歸測試 —— 它是整份清單最後一筆的游標。"""
    fake = _install(
        monkeypatch,
        [
            _response(_edges("a", _ids("a", 3)), last_cursor="整份清單的最後一筆"),
            _response([]),
        ],
    )
    cofacts_fetch.fetch("scam", out_path)
    assert fake.afters[1] == "a2"


def test_cursor_must_advance(monkeypatch, out_path, sleeps):
    fake = _install(
        monkeypatch,
        [
            _response([{"cursor": "同一個游標", "node": _node("x0")}]),
            _response([{"cursor": "同一個游標", "node": _node("x1")}]),
        ],
    )
    with pytest.raises(cofacts_fetch.CofactsFetchError) as excinfo:
        cofacts_fetch.fetch("scam", out_path)
    message = str(excinfo.value)
    assert "同一個游標" in message
    assert "第 1 頁" in message
    assert len(fake.payloads) == 2


def test_limit_stops_requesting(monkeypatch, out_path, sleeps):
    fake = _install(
        monkeypatch,
        [
            _response(_edges("a", _ids("a", 100))),
            _response(_edges("b", _ids("b", 100))),
            _response(_edges("c", _ids("c", 100))),
        ],
    )
    total = cofacts_fetch.fetch("scam", out_path, limit=250)
    assert total == 250
    assert len(fake.payloads) == 3
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 250


# --- 失敗處理 -------------------------------------------------------------


def test_http_403(monkeypatch, out_path, sleeps):
    _install(monkeypatch, [(403, "forbidden")])
    with pytest.raises(cofacts_fetch.CofactsFetchError) as excinfo:
        cofacts_fetch.fetch("scam", out_path)
    message = str(excinfo.value)
    assert "403" in message
    assert cofacts_fetch.USER_AGENT in message
    assert "第 0 頁" in message


def test_graphql_errors(monkeypatch, out_path, sleeps):
    body = json.dumps({"errors": [{"message": "語法錯誤", "path": ["ListArticles"]}]})
    _install(monkeypatch, [(200, body)])
    with pytest.raises(cofacts_fetch.CofactsFetchError) as excinfo:
        cofacts_fetch.fetch("scam", out_path)
    message = str(excinfo.value)
    assert "語法錯誤" in message
    assert "ListArticles" in message


def test_node_missing_text(monkeypatch, out_path, sleeps):
    node = _node("x0")
    del node["text"]
    _install(monkeypatch, [_response([{"cursor": "c0", "node": node}])])
    with pytest.raises(cofacts_fetch.CofactsFetchError) as excinfo:
        cofacts_fetch.fetch("scam", out_path)
    message = str(excinfo.value)
    assert "text" in message
    assert "第 0 頁" in message


def test_duplicate_article_id(monkeypatch, out_path, sleeps):
    _install(
        monkeypatch,
        [
            _response(_edges("a", ["重複的 id", "a1"])),
            _response(_edges("b", ["b0", "b1"])),
            _response(_edges("c", ["c0", "重複的 id"])),
        ],
    )
    with pytest.raises(cofacts_fetch.CofactsFetchError) as excinfo:
        cofacts_fetch.fetch("scam", out_path)
    message = str(excinfo.value)
    assert "重複的 id" in message
    assert "第 2 頁" in message


def test_failure_message_carries_resume_command(monkeypatch, out_path, sleeps):
    _install(
        monkeypatch,
        [
            _response(_edges("a", _ids("a", 2))),
            (500, "boom"),
        ],
    )
    with pytest.raises(cofacts_fetch.CofactsFetchError) as excinfo:
        cofacts_fetch.fetch("scam", out_path)
    assert "--after a1" in str(excinfo.value)


def test_main_returns_nonzero_without_success_message(monkeypatch, tmp_path, capsys, sleeps):
    monkeypatch.setattr(cofacts_fetch, "_require_git_ignored", _allow_any_path)
    _install(monkeypatch, [(500, "boom")])
    target = tmp_path / "out.jsonl"
    code = cofacts_fetch.main(["--label", "scam", "--out", str(target)])
    assert code == 1
    assert "完成" not in capsys.readouterr().err


# --- 撤銷的分類與查核結果 ---------------------------------------------------


def test_deleted_reply_is_excluded(monkeypatch, out_path, sleeps):
    node = _node("x0")
    node["articleReplies"] = [
        {"replyType": "RUMOR", "status": "NORMAL"},
        {"replyType": "NOT_RUMOR", "status": "DELETED"},
    ]
    _install(monkeypatch, [_response([{"cursor": "c0", "node": node}]), _response([])])
    cofacts_fetch.fetch("scam", out_path)
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["reply_types"] == ["RUMOR"]


def test_deleted_category_is_excluded(monkeypatch, out_path, sleeps):
    node = _node("x0")
    node["articleCategories"] = [
        {"categoryId": "nD2n7nEBrIRcahlYwQoW", "status": "NORMAL"},
        {"categoryId": "被撤銷的分類", "status": "DELETED"},
    ]
    _install(monkeypatch, [_response([{"cursor": "c0", "node": node}]), _response([])])
    cofacts_fetch.fetch("scam", out_path)
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["category_ids"] == ["nD2n7nEBrIRcahlYwQoW"]


# --- 落地格式 -------------------------------------------------------------


def test_one_line_per_record(monkeypatch, out_path, sleeps):
    _install(
        monkeypatch,
        [
            _response(_edges("a", _ids("a", 100))),
            _response(_edges("b", _ids("b", 40))),
            _response([]),
        ],
    )
    cofacts_fetch.fetch("scam", out_path)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 140
    for line in lines:
        assert json.loads(line)["label"] == "scam"


def test_chinese_is_not_escaped(monkeypatch, out_path, sleeps):
    edges = [{"cursor": "c0", "node": _node("x0", text="您的帳戶已遭凍結")}]
    _install(monkeypatch, [_response(edges), _response([])])
    cofacts_fetch.fetch("scam", out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "您的帳戶已遭凍結" in content
    assert "\\u" not in content


def test_text_with_newline_stays_one_line(monkeypatch, out_path, sleeps):
    edges = [{"cursor": "c0", "node": _node("x0", text="第一行\n第二行\n第三行")}]
    _install(monkeypatch, [_response(edges), _response([])])
    cofacts_fetch.fetch("scam", out_path)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "第一行\n第二行\n第三行"


def test_source_uri(monkeypatch, out_path, sleeps):
    edges = [{"cursor": "c0", "node": _node("3b32nkpcfp97f")}]
    _install(monkeypatch, [_response(edges), _response([])])
    cofacts_fetch.fetch("scam", out_path)
    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["source_uri"] == "https://cofacts.tw/article/3b32nkpcfp97f"


# --- 續抓 -----------------------------------------------------------------


def test_after_is_sent_and_output_is_appended(monkeypatch, out_path, sleeps):
    out_path.write_text('{"id":"既有"}\n', encoding="utf-8")
    edges = [{"cursor": "c0", "node": _node("x0")}]
    fake = _install(monkeypatch, [_response(edges), _response([])])
    cofacts_fetch.fetch("scam", out_path, after="起始游標")
    assert fake.afters[0] == "起始游標"
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "既有"
    assert json.loads(lines[1])["id"] == "x0"
