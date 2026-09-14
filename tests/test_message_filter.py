"""`tools.message_filter` 的測試。旗標測試全部以行內字串為輸入，不讀檔案。"""

import json

import pytest

from tools import message_filter

TIER_A_EXAMPLE = "是需要您寄出名下所有的卡片"
"""13 字的真實 Tier-A 例子。下界不設 20 的迴歸測試就靠它。"""

LINE_EXPORT = (
    "[LINE] 與金融正義處理陳先生的聊天記錄\n"
    "儲存日期： 2024/09/10 11:35\n"
    "\n"
    "2024/08/19（一）\n"
    "上午09:53\t小麥\t 您好\n"
    "上午09:54\t金融正義處理陳先生\tAnya把您的情況大概跟我講了一下\n"
)

ORDER_NOTICE = (
    "系統監測到您的帳戶存在「重複訂購／定期扣款訂購」\n"
    "訂單資訊如下：\n"
    "原始訂購日期：2024/08/01\n"
    "訂單金額：946\n"
    "客服專線：0800-000-000\n"
)


def _allow_any_path(path) -> None:
    """monkeypatch 用的替身：tmp_path 不在 repo 內，git check-ignore 會判定未被忽略。"""


def _record(article_id: str, text: str, label: str = "scam") -> dict:
    return {
        "id": article_id,
        "text": text,
        "created_at": "2026-09-12T10:11:59.220Z",
        "label": label,
        "category_ids": ["nD2n7nEBrIRcahlYwQoW"],
        "reply_types": ["RUMOR"],
        "source_uri": f"https://cofacts.tw/article/{article_id}",
        "fetched_at": "2026-09-14T09:12:33Z",
    }


def _write(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- 旗標：url_only -------------------------------------------------------


def test_url_only_bare_url():
    assert message_filter.flag_message("https://m.click108.com.tw/") == ("url_only",)


def test_url_only_with_small_residual():
    assert "url_only" in message_filter.flag_message("https://youtu.be/abcdefg 請看")


def test_not_url_only_with_real_content():
    text = "https://reurl.cc/abcdef 這是一則完整的說明文字總共有三十個字用來測試殘留門檻"
    assert "url_only" not in message_filter.flag_message(text)


def test_not_url_only_without_url():
    flags = message_filter.flag_message("做票抓到了")
    assert "url_only" not in flags
    assert "too_short" in flags


# --- 旗標：line_export ----------------------------------------------------


def test_line_export_header():
    assert "line_export" in message_filter.flag_message(LINE_EXPORT)


def test_order_notice_is_not_line_export():
    """寬鬆啟發式在此誤判（冒號是欄位分隔不是說話者），迴歸測試。"""
    assert "line_export" not in message_filter.flag_message(ORDER_NOTICE)


def test_timestamps_without_header_are_not_line_export():
    text = "他09:53傳訊息給我，我09:58才回，結果他說已經來不及了要我立刻匯款過去才能處理"
    assert "line_export" not in message_filter.flag_message(text)


# --- 旗標：長度 -----------------------------------------------------------


def test_too_short():
    assert message_filter.flag_message("一二三四五") == ("too_short",)


def test_thirteen_chars_is_not_too_short():
    assert message_filter.flag_message(TIER_A_EXAMPLE) == ()


def test_boundaries_are_not_flagged():
    assert message_filter.flag_message("一" * message_filter.MIN_NORM_LEN) == ()
    assert message_filter.flag_message("一" * message_filter.MAX_NORM_LEN) == ()


def test_too_long():
    assert message_filter.flag_message("一" * 8386) == ("too_long",)


def test_zero_width_chars_are_not_counted():
    text = "\u200b" * 10 + "一" * message_filter.MIN_NORM_LEN
    records = message_filter.annotate([_record("x0", text)])
    assert records[0]["norm_len"] == message_filter.MIN_NORM_LEN
    assert records[0]["flags"] == []


def test_flags_can_co_occur():
    text = "https://example.com/" + "a" * 2500
    assert message_filter.flag_message(text) == ("url_only", "too_long")


# --- 讀入與去重 -----------------------------------------------------------


def test_missing_text_field(tmp_path):
    path = tmp_path / "in.jsonl"
    records = [_record(f"x{index}", "內容一二三四五六七八") for index in range(12)]
    del records[11]["text"]
    _write(path, records)
    with pytest.raises(message_filter.MessageFilterError) as excinfo:
        message_filter.read_records([path])
    message = str(excinfo.value)
    assert "12" in message
    assert "text" in message


def test_missing_label_field(tmp_path):
    path = tmp_path / "in.jsonl"
    record = _record("x0", "內容一二三四五六七八")
    del record["label"]
    _write(path, [record])
    with pytest.raises(message_filter.MessageFilterError) as excinfo:
        message_filter.read_records([path])
    assert "label" in str(excinfo.value)


def test_duplicate_id_same_text_is_deduplicated(tmp_path, capsys):
    path = tmp_path / "in.jsonl"
    record = _record("x0", "內容一二三四五六七八")
    _write(path, [record, record])
    records, duplicates = message_filter.read_records([path])
    assert len(records) == 1
    assert duplicates == 1
    message_filter._report(message_filter.annotate(records), duplicates)
    assert "去重 1 筆" in capsys.readouterr().err


def test_duplicate_id_different_text_raises(tmp_path):
    path = tmp_path / "in.jsonl"
    _write(
        path,
        [
            _record("x0", "第一個版本的內文一二三四五六七八"),
            _record("x0", "第二個版本的內文一二三四五六七八"),
        ],
    )
    with pytest.raises(message_filter.MessageFilterError) as excinfo:
        message_filter.read_records([path])
    message = str(excinfo.value)
    assert "x0" in message
    assert "第一個版本的內文" in message
    assert "第二個版本的內文" in message


# --- 輸出 -----------------------------------------------------------------


def test_nothing_is_dropped(tmp_path, monkeypatch, capsys):
    path = tmp_path / "in.jsonl"
    records = [_record(f"x{index}", "這是一則有足夠長度的正常訊息內容") for index in range(80)]
    records += [_record(f"u{index}", "https://example.com/abcdef") for index in range(20)]
    _write(path, records)
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(message_filter, "_require_git_ignored", _allow_any_path)
    assert message_filter.main([str(path), "--out", str(out_path)]) == 0
    written = _read(out_path)
    assert len(written) == 100
    assert sum(1 for record in written if "url_only" in record["flags"]) == 20
    assert sum(1 for record in written if not record["flags"]) == 80


def test_input_fields_are_preserved():
    record = _record("3b32nkpcfp97f", "這是一則有足夠長度的正常訊息內容")
    annotated = message_filter.annotate([record])[0]
    for key, value in record.items():
        assert annotated[key] == value
    assert annotated["flags"] == []
    assert annotated["norm_len"] == len(record["text"])


def test_two_inputs_merge_into_one_output(tmp_path, monkeypatch):
    scam_path = tmp_path / "scam.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    _write(scam_path, [_record("s0", "這是一則詐騙訊息的內容範例", label="scam")])
    _write(
        negative_path,
        [_record("n0", "這是一則政策宣導的內容範例", label="hard-negative")],
    )
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(message_filter, "_require_git_ignored", _allow_any_path)
    message_filter.main([str(scam_path), str(negative_path), "--out", str(out_path)])
    written = _read(out_path)
    assert [record["label"] for record in written] == ["scam", "hard-negative"]


def test_chinese_is_not_escaped(tmp_path, monkeypatch):
    path = tmp_path / "in.jsonl"
    _write(path, [_record("x0", "您的帳戶已遭凍結請立即處理")])
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(message_filter, "_require_git_ignored", _allow_any_path)
    message_filter.main([str(path), "--out", str(out_path)])
    content = out_path.read_text(encoding="utf-8")
    assert "您的帳戶已遭凍結請立即處理" in content
    assert "\\u" not in content
