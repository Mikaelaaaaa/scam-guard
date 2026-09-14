"""呈現：依據的組裝規則、三張禁用表、可查證性、動作目錄的准入條件。"""

import ast
from pathlib import Path

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document, build_document
from scam_guard.pipeline import detect
from scam_guard.render import (
    ACTIONS,
    CALIBRATION_CLAIMS,
    CONTRADICTION_NOTE,
    HARM_ALLOWED,
    QUOTE_SEPARATOR,
    TRUNCATION_NOTE,
    _has_speculative_term,
    _has_verdict_claim,
    _is_verifiable,
    choose_actions,
    render_evidence,
)
from scam_guard.scoring import compute_score
from scam_guard.types import CheckResult, Coord, Message, Request, ScamType
from scam_guard.weights import load_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE = load_weights()

AUTHORITY_SCRIPT = ("safe_account", "atm_operation", "secrecy_demand", "remote_control_tool")
EIGHT_GROUPS = (
    ("safe_account", True),
    ("solicit_otp", True),
    ("prepay_to_receive", True),
    ("guaranteed_return", False),
    ("url_brand", False),
    ("evasion_invisible", False),
    ("parcel_notice", False),
    ("identity_reset", False),
)

VERIFY = "撥打 165 反詐騙專線查證"
DELAY_ACTION = "查證前不要依訊息指示操作"


def hit(
    name: str,
    detail: str = "此訊息要求提供簡訊驗證碼；銀行不會這樣要求",
    *,
    hard: bool = False,
    evidence: tuple[Coord, ...] = (),
) -> CheckResult:
    return CheckResult(
        name=name,
        hit=True,
        detail=detail,
        evidence=list(evidence),
        scam_types=[ScamType.FAKE_AUTHORITY],
        hard=hard,
    )


def miss(name: str) -> CheckResult:
    return CheckResult(name=name, hit=False, detail="未命中")


def a_document(*texts: str) -> Document:
    return build_document([Message(text=text) for text in texts])


def rendered(results: list[CheckResult], doc: Document) -> list[str]:
    return render_evidence(results, compute_score(results, TABLE), doc, TABLE)


# --- 依據由結果組裝 ------------------------------------------------------


def test_every_line_maps_back_to_a_detail() -> None:
    doc = a_document("請將款項匯入監管帳戶", "跟著老師操作保證獲利")
    results = [
        hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 0),)),
        hit("guaranteed_return", "承諾保證獲利", evidence=((1, 0),)),
    ]

    lines = rendered(results, doc)
    details = {result.detail for result in results}

    assert all(line.split(QUOTE_SEPARATOR)[0] in details for line in lines)


def test_no_facts_are_added() -> None:
    doc = a_document("網域 evil.com 註冊於 6 天前")
    detail = "網域 evil.com 註冊於 6 天前（2026-09-08）"
    results = [hit("domain_age", detail, evidence=((0, 0),))]

    (line,) = rendered(results, doc)

    assert line.split(QUOTE_SEPARATOR)[0] == detail
    assert "高風險" not in line


def test_only_two_constants_are_free_text() -> None:
    doc = a_document("有人傳這個給我。請匯到監管帳戶")
    results = [
        hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 1),)),
        hit("quotation", "命中 1 類引述標記：來源歸屬", evidence=((0, 0),)),
    ]

    lines = rendered(results, doc)
    details = {result.detail for result in results}
    free = [line for line in lines if line.split(QUOTE_SEPARATOR)[0] not in details]

    assert free == [CONTRADICTION_NOTE]


# --- 三張禁用表與可查證性 ------------------------------------------------


def test_speculative_term_is_blocked() -> None:
    assert _has_speculative_term("這個網域可疑") is True


def test_concrete_fact_passes() -> None:
    line = "網域 evil.com 註冊於 6 天前（2026-09-08）"

    assert _has_speculative_term(line) is False
    assert _has_verdict_claim(line) is False
    assert _is_verifiable(line) is True


def test_verdict_claim_is_blocked() -> None:
    assert _has_verdict_claim("這是詐騙訊息") is True


def test_official_dataset_name_passes() -> None:
    """「詐」字本身不禁 —— 引用官方名稱是依據的範本而非反例。"""
    line = "evil.com 於 2026-08 列入 165 反詐騙諮詢專線_遭停止解析涉詐網站"

    assert _has_verdict_claim(line) is False
    assert _has_speculative_term(line) is False
    assert _is_verifiable(line) is True


def test_line_without_number_name_or_quote_is_blocked() -> None:
    assert _is_verifiable("這則訊息的語氣讓人覺得怪怪的") is False


def test_quoted_original_text_is_verifiable() -> None:
    assert _is_verifiable("此訊息含拆字痕跡：「監 管 帳 戶」") is True


def test_real_output_passes_every_text_rule() -> None:
    doc = a_document("請將款項匯入監管帳戶")
    results = [hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 0),))]

    for line in rendered(results, doc):
        assert not _has_speculative_term(line)
        assert not _has_verdict_claim(line)
        assert _is_verifiable(line)


def test_no_calibration_claims_anywhere() -> None:
    doc = a_document("請將款項匯入監管帳戶")
    results = [hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 0),))]
    score = compute_score(results, TABLE)
    lines = render_evidence(results, score, doc, TABLE) + choose_actions(
        results, score, False, TABLE
    )

    assert not [line for line in lines if any(word in line for word in CALIBRATION_CLAIMS)]


# --- 原文片段 -----------------------------------------------------------


def test_evasion_evidence_shows_the_raw_shape() -> None:
    doc = a_document("請匯到監 管 帳 戶")
    results = [hit("evasion_split_word", "目標詞被分隔字元拆開", evidence=((0, 0),))]

    (line,) = rendered(results, doc)

    assert "監 管 帳 戶" in line


def test_out_of_range_coordinate_propagates() -> None:
    doc = a_document("請匯到監管帳戶")
    results = [hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((9, 9),))]

    with pytest.raises(KeyError):
        rendered(results, doc)


def test_phone_number_in_the_quote_is_not_redacted() -> None:
    """`Verdict` 回給的是送出這則訊息的人本人，對他遮蔽自己的訊息不保護任何人。"""
    doc = a_document("請回撥 0912345678")
    results = [hit("identity_reset", "要求回撥訊息裡的號碼", evidence=((0, 0),))]

    (line,) = rendered(results, doc)

    assert "0912345678" in line


# --- 排序、去重與上限 ---------------------------------------------------


def test_lines_are_ordered_by_contribution() -> None:
    doc = a_document("請匯入監管帳戶。保證獲利。包裹待領")
    results = [
        hit("guaranteed_return", "承諾保證獲利", evidence=((0, 1),)),
        hit("parcel_notice", "冒稱物流通知", evidence=((0, 2),)),
        hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 0),)),
    ]

    lines = rendered(results, doc)

    assert lines[0].startswith("要求匯入監管帳戶")


def test_four_rules_in_one_group_render_one_line() -> None:
    doc = a_document("請匯入監管帳戶")
    results = [
        hit(name, f"假檢警腳本第 {index} 條", hard=True, evidence=((0, 0),))
        for index, name in enumerate(AUTHORITY_SCRIPT)
    ]

    assert len(rendered(results, doc)) == 1


def test_eight_groups_are_capped_at_five_lines() -> None:
    doc = a_document("請匯入監管帳戶")
    results = [
        hit(name, f"第 {index} 條依據", hard=hard, evidence=((0, 0),))
        for index, (name, hard) in enumerate(EIGHT_GROUPS)
    ]
    results.extend(miss(name) for name in ("url_shortener", "domain_age"))

    lines = rendered(results, doc)

    assert len(lines) == int(TABLE.threshold("max_evidence_lines")) == 5
    assert len(results) == 10


def test_evidence_contains_no_internal_identifier() -> None:
    registry = CheckRegistry()
    rule = hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 0),))
    registry.register(StaticCheck("safe_account", [rule]))
    registry.register(StaticCheck("evasion_invisible", []))

    verdict = detect(Request.from_text("請匯入監管帳戶"), registry, TABLE)
    names = {check.name for check in registry.enabled()}

    assert not [line for line in verdict.evidence for name in names if name in line]


# --- 動作目錄 -----------------------------------------------------------


def test_every_action_is_from_the_catalogue() -> None:
    results = [hit("solicit_otp", "此訊息要求提供驗證碼", hard=True, evidence=((0, 0),))]
    texts = {action.text for action in ACTIONS}

    chosen = choose_actions(results, compute_score(results, TABLE), False, TABLE)

    assert set(chosen) <= texts


def test_every_catalogue_entry_declares_an_allowed_harm() -> None:
    assert all(action.harm_if_genuine in HARM_ALLOWED for action in ACTIONS)


def test_verification_action_has_no_harm() -> None:
    (verify,) = [action for action in ACTIONS if action.text == VERIFY]

    assert verify.harm_if_genuine == "無"


def test_delay_action_is_marked_recoverable() -> None:
    (delay,) = [action for action in ACTIONS if action.text == DELAY_ACTION]

    assert delay.harm_if_genuine == "可回復的延遲"


def test_irreversible_and_unexecutable_actions_are_absent() -> None:
    texts = {action.text for action in ACTIONS}

    for banned in ("立即封鎖並刪除", "報警", "不要理會", "小心詐騙"):
        assert banned not in texts


def test_credential_group_gets_the_credential_action() -> None:
    results = [hit("solicit_otp", "此訊息要求提供驗證碼", hard=True, evidence=((0, 0),))]

    chosen = choose_actions(results, compute_score(results, TABLE), False, TABLE)

    assert any("驗證碼" in text for text in chosen)


def test_url_group_gets_the_link_action() -> None:
    doc = a_document("請點 https://evil.com")
    results = [hit("url_blocklist", "命中 165 涉詐網站清單", hard=True, evidence=((0, 0),))]

    chosen = choose_actions(results, compute_score(results, TABLE), False, TABLE)

    assert any("連結" in text for text in chosen)
    assert doc.sentences


def test_actions_are_capped() -> None:
    doc = a_document("請匯入監管帳戶")
    results = [
        hit(name, f"第 {index} 條依據", hard=hard, evidence=((0, 0),))
        for index, (name, hard) in enumerate(EIGHT_GROUPS)
    ]

    chosen = choose_actions(results, compute_score(results, TABLE), False, TABLE)

    assert len(chosen) == int(TABLE.threshold("max_actions")) == 3
    assert doc.sentences


def test_actions_exist_even_when_the_type_is_none() -> None:
    registry = CheckRegistry()
    registry.register(
        StaticCheck(
            "evasion_invisible",
            [
                CheckResult(
                    name="evasion_invisible",
                    hit=True,
                    detail="訊息中含 1 個零寬空格",
                    evidence=[(0, 0)],
                )
            ],
        )
    )

    verdict = detect(Request.from_text("匯款"), registry, TABLE)

    assert verdict.scam_type is None
    assert verdict.actions != []


def test_no_generated_action_text() -> None:
    """實作中沒有以字串拼接或格式化產生建議文字的程式碼。"""
    tree = ast.parse((REPO_ROOT / "scam_guard" / "render.py").read_text(encoding="utf-8"))
    chosen = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "choose_actions"
    ]
    formats = [
        node
        for node in ast.walk(chosen[0])
        if isinstance(node, (ast.JoinedStr, ast.BinOp))
        or (isinstance(node, ast.Attribute) and node.attr == "format")
    ]

    assert formats == []


# --- 沉默與拒答 ---------------------------------------------------------


class StaticCheck:
    """回傳固定結果的假檢查。"""

    stage = Stage.LOCAL

    def __init__(self, name: str, results: list[CheckResult]) -> None:
        self.name = name
        self.results = results

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        return list(self.results)


def test_a_message_with_no_signal_is_answered_with_silence() -> None:
    registry = CheckRegistry()
    registry.register(StaticCheck("solicit_otp", []))

    verdict = detect(Request.from_text("明天見"), registry, TABLE)

    assert verdict.evidence == []
    assert verdict.actions == []
    assert verdict.scam_probability is None


def test_abstaining_with_hits_is_not_silent() -> None:
    doc = a_document("有人傳這個給我。請匯到監管帳戶")
    results = [
        hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 1),)),
        hit("quotation", "命中 1 類引述標記：來源歸屬", evidence=((0, 0),)),
    ]
    score = compute_score(results, TABLE)

    assert render_evidence(results, score, doc, TABLE) != []
    assert choose_actions(results, score, True, TABLE) != []


def test_abstaining_drops_the_recoverable_delay_actions() -> None:
    doc = a_document("有人傳這個給我。請匯到監管帳戶")
    results = [
        hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 1),)),
        hit("quotation", "命中 1 類引述標記：來源歸屬", evidence=((0, 0),)),
    ]

    chosen = choose_actions(results, compute_score(results, TABLE), True, TABLE)
    delayed = {action.text for action in ACTIONS if action.harm_if_genuine == "可回復的延遲"}

    assert not set(chosen) & delayed
    assert doc.sentences


def test_abstaining_still_suggests_verification() -> None:
    doc = a_document("有人傳這個給我")
    results = [hit("quotation", "命中 1 類引述標記：來源歸屬", evidence=((0, 0),))]

    chosen = choose_actions(results, compute_score(results, TABLE), True, TABLE)

    assert VERIFY in chosen
    assert doc.sentences


# --- 矛盾 ---------------------------------------------------------------


def test_awareness_post_end_to_end() -> None:
    registry = CheckRegistry()
    registry.register(
        StaticCheck(
            "safe_account",
            [hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 1),))],
        )
    )
    registry.register(
        StaticCheck(
            "quotation",
            [hit("quotation", "命中 1 類引述標記：宣導框架", evidence=((0, 2),))],
        )
    )

    verdict = detect(
        Request.from_text("最近很多假檢警詐騙。會叫你把錢匯到監管帳戶。千萬不要相信"),
        registry,
        TABLE,
    )

    assert verdict.scam_probability is None
    assert CONTRADICTION_NOTE in verdict.evidence
    assert any("監管帳戶" in line for line in verdict.evidence)
    assert any("引述標記" in line for line in verdict.evidence)
    assert verdict.actions != [VERIFY]
    assert VERIFY in verdict.actions


def test_contradiction_note_is_absent_when_the_flag_is_false() -> None:
    doc = a_document("請匯到監管帳戶")
    results = [hit("safe_account", "要求匯入監管帳戶", hard=True, evidence=((0, 0),))]

    assert CONTRADICTION_NOTE not in rendered(results, doc)


def test_contradiction_note_states_system_state_not_message_nature() -> None:
    assert "宣導" not in CONTRADICTION_NOTE
    assert "轉發" not in CONTRADICTION_NOTE
    assert CONTRADICTION_NOTE.startswith("系統")


# --- 截斷 ---------------------------------------------------------------


def long_conversation() -> Document:
    return build_document([Message(text=f"第 {index} 則訊息") for index in range(101)])


def test_truncation_is_stated() -> None:
    doc = long_conversation()
    results = [hit("guaranteed_return", "承諾保證獲利")]

    lines = render_evidence(results, compute_score(results, TABLE), doc, TABLE)

    assert TRUNCATION_NOTE.format(dropped=doc.dropped_messages) in lines
    assert "前 1 則訊息因超過上限未納入判斷" in lines


def test_truncation_is_not_stated_when_hard_evidence_exists() -> None:
    doc = long_conversation()
    results = [hit("safe_account", "要求匯入監管帳戶", hard=True)]

    lines = render_evidence(results, compute_score(results, TABLE), doc, TABLE)

    assert not [line for line in lines if "未納入判斷" in line]


# --- 界線 ---------------------------------------------------------------


def test_module_docstring_records_the_redaction_decision() -> None:
    import scam_guard.render

    assert "遮蔽" in scam_guard.render.__doc__
    assert "add-redact-apply" in scam_guard.render.__doc__


def test_module_does_not_import_pipeline_or_rules() -> None:
    source = (REPO_ROOT / "scam_guard" / "render.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert not any("scam_guard.pipeline" in line for line in imports)
    assert not any("scam_guard.rules" in line for line in imports)
