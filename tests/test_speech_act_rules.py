"""言語行為規則 —— 三個機制、規則目錄與整合行為。"""

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import Document, build_document
from scam_guard.pipeline import detect
from scam_guard.rules.speech_act import (
    CODE_EXEMPT_RULES,
    HARD_WEIGHT,
    HELP_CHANNELS,
    RELATIONSHIP_RULE,
    SPEECH_ACT_RULES,
    TIER_A_RULES,
    TIER_B_RULES,
    WEAK_WEIGHT,
    RelationshipRule,
    SpeechActRule,
    has_self_contained_code,
    register_speech_act_rules,
)
from scam_guard.types import CheckResult, Message, Request, ScamType


class FakeLlm:
    """假的 `EXPENSIVE` 檢查，用來觀察短路有沒有發生。"""

    name = "llm"
    stage = Stage.EXPENSIVE

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]:
        self.calls += 1
        return []


def rule_named(name: str) -> SpeechActRule:
    return next(rule for rule in SPEECH_ACT_RULES if rule.name == name)


def run(name: str, *texts: str) -> list[CheckResult]:
    """對指定規則跑一次比對，回傳它的結果（未命中時為空陣列）。"""
    req = Request(messages=[Message(text=text) for text in texts])
    return rule_named(name)(req, build_document(req.messages))


def run_all(*texts: str) -> list[CheckResult]:
    req = Request(messages=[Message(text=text) for text in texts])
    doc = build_document(req.messages)
    return [result for rule in SPEECH_ACT_RULES for result in rule(req, doc)]


# --- 三個機制：言語行為、自帶碼豁免、受益者 ---------------------------------


def test_solicit_otp_hits() -> None:
    results = run("solicit_otp", "請把剛收到的驗證碼告訴我")

    assert [(r.hit, r.hard, r.weight) for r in results] == [(True, True, HARD_WEIGHT)]
    assert results[0].evidence == [(0, 0)]
    assert results[0].scam_types == [ScamType.ACCOUNT_TAKEOVER]


def test_providing_otp_does_not_hit() -> None:
    """台灣每天發出數百萬封的標準一次性密碼簡訊。"""
    assert run("solicit_otp", "您的驗證碼是 123456，請勿告訴他人") == []


def test_negation_in_earlier_clause_does_not_shield() -> None:
    assert len(run("solicit_otp", "不要告訴別人，把驗證碼傳給我")) == 1


def test_self_contained_code_spans_the_whole_message() -> None:
    """兩行的一次性密碼簡訊：換行是兩個句子，豁免仍以整則訊息為範圍。"""
    req = Request(messages=[Message(text="您的驗證碼為 482913\n請勿提供他人")])
    doc = build_document(req.messages)

    assert len(doc.sentences) == 2
    assert has_self_contained_code(doc, 0) == "482913"


def test_exemption_downgrades_but_keeps_the_record() -> None:
    results = run("solicit_otp", "您的驗證碼是 482913，請把驗證碼傳給我")

    assert len(results) == 1
    assert results[0].hit is True
    assert results[0].hard is False
    assert results[0].weight == WEAK_WEIGHT
    assert results[0].evidence == [(0, 0)]
    assert "482913" in results[0].detail
    assert "降級" in results[0].detail


def test_exemption_removes_the_short_circuit() -> None:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    llm = FakeLlm()
    registry.register(llm)

    detect(Request.from_text("您的驗證碼是 482913，請把驗證碼傳給我"), registry)

    assert llm.calls == 1


def test_amount_does_not_trigger_the_exemption() -> None:
    """豁免是攻擊者可控的輸入，金額不得成為取得它的六個字元。"""
    results = run("solicit_otp", "請匯 50000 元到指定帳戶，順便把驗證碼給我")

    assert [r.hard for r in results] == [True]


def test_long_digit_string_is_not_a_self_contained_code() -> None:
    req = Request(messages=[Message(text="訂單802045734652已成立")])

    assert has_self_contained_code(build_document(req.messages), 0) is None


def test_self_directed_receiver_is_required() -> None:
    assert run("solicit_otp", "把驗證碼傳給我") != []
    assert run("solicit_otp", "請勿將驗證碼告知他人") == []
    assert run("solicit_otp", "驗證碼已送出") == []


# --- `secrecy_demand` —— 誤判風險最高的一條 ---------------------------------


def test_secrecy_demand_hits_blocked_help_channel() -> None:
    results = run("secrecy_demand", "這是偵查不公開的案件，不要告訴你的家人")

    assert [(r.hard, r.scam_types) for r in results] == [(True, [ScamType.FAKE_AUTHORITY])]


def test_secrecy_demand_ignores_generic_third_party() -> None:
    """迴歸：把「他人」收進求助管道清單，等於把每一封一次性密碼簡訊判成假檢警。"""
    assert run("secrecy_demand", "請勿將驗證碼告知他人") == []


def test_secrecy_demand_requires_negative_polarity() -> None:
    """極性是規則的欄位 —— 寫成「有否定就取消」時這條規則永遠不會命中。"""
    assert run("secrecy_demand", "請告訴你的家人") == []


def test_help_channels_exclude_generic_terms() -> None:
    assert HELP_CHANNELS.isdisjoint({"任何人", "他人", "別人", "第三方", "外人", "對外"})


def test_secrecy_demand_is_also_exempted_by_a_self_contained_code() -> None:
    assert CODE_EXEMPT_RULES == {"solicit_otp", "secrecy_demand"}


# --- 規則目錄 ---------------------------------------------------------------


def test_rule_names_are_unique_snake_case() -> None:
    names = [rule.name for rule in SPEECH_ACT_RULES]

    assert len(names) == 21
    assert len(set(names)) == 21
    assert all(name.replace("_", "").isalnum() and name.islower() for name in names)


def test_exactly_ten_hard_rules_each_with_a_fact() -> None:
    hard = [rule for rule in SPEECH_ACT_RULES if rule.hard]

    assert len(hard) == 10
    assert all(rule.fact for rule in hard)
    assert len(TIER_A_RULES) == 10
    assert len(TIER_B_RULES) == 10


def test_weight_has_exactly_two_distinct_values() -> None:
    weights = {result.weight for text in HIT_SAMPLES for result in run_all(text)}

    assert weights == {HARD_WEIGHT, WEAK_WEIGHT}


def test_scam_types_are_enum_members() -> None:
    for rule in SPEECH_ACT_RULES:
        for scam_type in getattr(rule, "scam_types", ()):
            assert isinstance(scam_type, ScamType)


def test_relationship_rule_emits_no_type() -> None:
    results = run(RELATIONSHIP_RULE, "我在杜拜做工程師。加我LINE，我們私聊。")

    assert results[0].hit is True
    assert results[0].scam_types == []


def test_no_rule_emits_romance_investment() -> None:
    """`ROMANCE_INVESTMENT` 由 `add-type-resolve` 合成，不由任何單一規則輸出。"""
    for rule in SPEECH_ACT_RULES:
        assert ScamType.ROMANCE_INVESTMENT not in getattr(rule, "scam_types", ())


def test_direct_scam_type_emission_is_rejected() -> None:
    with pytest.raises(ValueError, match="ROMANCE_INVESTMENT"):
        SpeechActRule(
            name="bad_rule",
            summary="",
            predicates=frozenset({"聊"}),
            polarity=rule_named("guaranteed_return").polarity,
            scam_types=(ScamType.ROMANCE_INVESTMENT,),
        )


def test_every_scam_type_except_romance_investment_has_a_producer() -> None:
    produced = {
        scam_type for rule in SPEECH_ACT_RULES for scam_type in getattr(rule, "scam_types", ())
    }

    assert produced == set(ScamType) - {ScamType.ROMANCE_INVESTMENT}


def test_all_rules_are_local_stage() -> None:
    assert all(rule.stage is Stage.LOCAL for rule in SPEECH_ACT_RULES)


def test_hard_rules_must_declare_receivers_and_fact() -> None:
    with pytest.raises(ValueError, match="接收者"):
        SpeechActRule(
            name="no_receiver",
            summary="",
            predicates=frozenset({"匯"}),
            polarity=rule_named("safe_account").polarity,
            hard=True,
            fact="某個事實",
        )
    with pytest.raises(ValueError, match="事實"):
        SpeechActRule(
            name="no_fact",
            summary="",
            predicates=frozenset({"匯"}),
            polarity=rule_named("safe_account").polarity,
            receivers=frozenset({"我"}),
            hard=True,
        )


def test_hard_detail_states_a_fact() -> None:
    results = run("safe_account", "請將款項匯入監管帳戶")

    assert "我國法制不存在監管帳戶" in results[0].detail


HIT_SAMPLES: tuple[str, ...] = (
    "請把剛收到的驗證碼告訴我",
    "請把信用卡號與末三碼回傳給我",
    "請提供網銀帳號密碼給我",
    "請把存摺與提款卡寄給我",
    "請至ATM解除分期付款設定",
    "請將款項匯入監管帳戶",
    "這是偵查不公開的案件，不要告訴你的家人",
    "請安裝 AnyDesk 讓我協助您",
    "您中獎了，請先匯手續費到指定帳戶才能領取獎金",
    "請您點選連結完成賣家認證開通商店",
    "跟著老師操作保證獲利",
    "我們結婚後就能團聚，機票錢請先幫我匯過來",
    "請提供身分證正反面照片以便建檔",
    "外約先儲值訂金才會派妹妹過去",
    "請把遊戲點數卡序號傳給我",
    "我換號碼了，這是我的新號碼",
    "今天是最後期限，急難救助善款請匯款至指定帳戶",
    "您的包裹因地址不全無法派送，請更新收件地址",
    "免聯徵免對保，當日撥款",
    "誠徵兼職人員，日領五千免經驗",
    "我在杜拜做工程師。加我LINE，我們私聊。",
)
"""每條規則至少一個會命中的樣本，順序與 `SPEECH_ACT_RULES` 對應。"""


def test_every_rule_has_a_hitting_sample() -> None:
    for rule, text in zip(SPEECH_ACT_RULES, HIT_SAMPLES):
        assert run(rule.name, text) != [], f"{rule.name} 未命中它自己的樣本：{text}"


COUNTEREXAMPLES: tuple[tuple[str, str, bool], ...] = (
    # 最後一欄是**這條規則在此反例上的實際行為**，不是它應該有的行為。
    # `True` 代表這是已知且被接受的誤判方向 —— design 逐條標註過它們，
    # 而防詐宣導文那幾條正是 `add-quotation-check` 存在的理由（引述命中會否決短路）。
    ("solicit_otp", "請把剛才簡訊的驗證碼念給我，我幫您完成掛失", True),
    ("solicit_card_secret", "請提供信用卡號與有效期限以保留訂位", False),
    ("solicit_bank_credentials", "這是模擬釣魚信，請勿提供網銀密碼給我", False),
    ("deliver_bank_instrument", "媽，把存摺寄給我，我幫你去辦", True),
    ("atm_operation", "他會叫你去 ATM 解除分期，千萬不要去", True),
    ("safe_account", "我國沒有監管帳戶這種東西", False),
    ("secrecy_demand", "請勿將驗證碼告知他人", False),
    ("remote_control_tool", "請安裝 TeamViewer 我幫你看一下", True),
    ("prepay_to_receive", "租屋需先付兩個月押金", False),
    ("seller_verification", "蝦皮官方提醒:賣家認證請至 App 內完成", False),
    ("guaranteed_return", "有人說保證獲利，那是詐騙", True),
    ("romance_pretext", "我下週出差，機票錢先幫我墊一下", False),
    ("identity_docs", "租屋簽約時房東會要求提供身分證影本", True),
    ("escort_deposit", "餐廳訂位需先付訂金", False),
    ("game_code", "我幫你買的點數卡序號等下傳給你", True),
    ("identity_reset", "我換號碼了，記得更新一下通訊錄", True),
    ("charity_personal_account", "本會急難救助專戶帳號如下，捐款請註明用途", False),
    ("parcel_notice", "您的包裹已送達超商，請憑取件碼領取", False),
    ("loan_no_check", "本行貸款需辦理聯徵查詢", False),
    ("high_pay_no_skill", "誠徵工讀生，時薪 190 元", False),
    (RELATIONSHIP_RULE, "我在杜拜做工程師。", False),
)


def test_known_counterexamples_behave_as_documented() -> None:
    """每條規則的已知反例都有斷言，包含那些**已知會誤判**的反例。

    對這幾條斷言「不命中」會是假的 —— design 明白寫著它們的誤判方向真實存在
    （客服代操作、家人代辦、朋友代買點數、宣導文引述）。
    釘住實際行為，改動使誤判擴大或消失時測試會說話。
    """
    assert {name for name, _, _ in COUNTEREXAMPLES} == {rule.name for rule in SPEECH_ACT_RULES}
    for name, text, expected_hit in COUNTEREXAMPLES:
        assert bool(run(name, text)) is expected_hit, f"{name} 在反例上的行為改變了：{text}"


# --- 關係經營 ---------------------------------------------------------------


def test_relationship_needs_two_categories_in_different_sentences() -> None:
    results = run(RELATIONSHIP_RULE, "我在杜拜做工程師。加我LINE，我們私聊。")

    assert len(results) == 1
    assert results[0].evidence == [(0, 0), (0, 1)]


def test_relationship_single_category_does_not_hit() -> None:
    assert run(RELATIONSHIP_RULE, "我在杜拜做工程師。我是醫生。") == []


def test_relationship_does_not_span_messages() -> None:
    """扁平的 `Document` 讓跨句規則容易誤跨訊息，這條釘住它不會。"""
    results = run(
        RELATIONSHIP_RULE,
        "我在杜拜做工程師。",
        "在忙嗎。",
        "今天天氣不錯。",
        "我想照顧你一輩子。",
    )

    assert results == []


def test_relationship_rule_name_is_an_exported_constant() -> None:
    assert RelationshipRule().name == RELATIONSHIP_RULE
    assert RELATIONSHIP_RULE == "relationship_building"


# --- 整合 -------------------------------------------------------------------


def benign_request() -> Request:
    return Request.from_text("今天天氣不錯，我們晚點約在捷運站見。")


def test_registry_records_all_twenty_one_rules() -> None:
    registry = CheckRegistry()
    register_speech_act_rules(registry)

    checks = detect(benign_request(), registry).checks

    assert len(checks) == 21
    assert all(not result.hit for result in checks)


def test_tier_a_hit_skips_expensive_checks() -> None:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    llm = FakeLlm()
    registry.register(llm)

    detect(Request.from_text("請將款項匯入監管帳戶"), registry)

    assert llm.calls == 0


def test_tier_b_hit_does_not_skip_expensive_checks() -> None:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    llm = FakeLlm()
    registry.register(llm)

    detect(Request.from_text("跟著老師操作保證獲利"), registry)

    assert llm.calls == 1


def test_disabling_one_rule_leaves_the_rest() -> None:
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    registry.disable("secrecy_demand")

    names = [result.name for result in detect(benign_request(), registry).checks]

    assert "secrecy_demand" not in names
    assert len(names) == 20


def test_unknown_rule_name_raises_on_disable() -> None:
    registry = CheckRegistry()
    register_speech_act_rules(registry)

    with pytest.raises(KeyError):
        registry.disable("secrecy_demannd")


def test_every_reported_coordinate_resolves() -> None:
    for text in HIT_SAMPLES:
        req = Request(messages=[Message(text=text)])
        doc = build_document(req.messages)
        for rule in SPEECH_ACT_RULES:
            for result in rule(req, doc):
                for coord in result.evidence:
                    assert doc.index_of(coord) >= 0
                assert result.evidence != []


def test_blank_message_produces_no_results() -> None:
    assert run_all("   ") == []


def test_rules_need_no_expensive_checks() -> None:
    """純規則模式 —— baseline 的定義就是沒有 LLM 時仍完整運作。"""
    with_llm = CheckRegistry()
    register_speech_act_rules(with_llm)
    with_llm.register(FakeLlm())
    rules_only = CheckRegistry()
    register_speech_act_rules(rules_only)

    text = "跟著老師操作保證獲利"
    paired = detect(Request.from_text(text), with_llm).checks
    alone = detect(Request.from_text(text), rules_only).checks

    assert [r for r in paired if r.name != "llm"] == alone
