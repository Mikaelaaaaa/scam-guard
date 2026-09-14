"""規避偵測 —— 資料來源、五個訊號的門檻、輸出形狀與詞表耦合。"""

from scam_guard.check import CheckRegistry, Stage
from scam_guard.normalize import INVISIBLE, build_document
from scam_guard.rules.evasion import (
    EVASION_CHECKS,
    EVASION_WEIGHT,
    HOMOPHONE_VARIANTS,
    INVISIBLE_NAMES,
    MAX_GAP,
    SPLIT_SEPARATORS,
    TARGET_WORDS,
    HomophoneCheck,
    InvisibleCharCheck,
    SplitWordCheck,
    WidthMixCheck,
    ZhuyinCheck,
    register_evasion_checks,
)
from scam_guard.rules.speech_act import SELF_DIRECTED, TIER_A_RULES
from scam_guard.types import CheckResult, Message, Request

ZERO_WIDTH_SPACE = "\u200b"
RIGHT_TO_LEFT_OVERRIDE = "\u202e"
BYTE_ORDER_MARK = "\ufeff"
FAMILY_EMOJI = "\U0001f468\u200d\U0001f469\u200d\U0001f467"


def run(check, *texts: str) -> list[CheckResult]:
    req = Request(messages=[Message(text=text) for text in texts])
    return check(req, build_document(req.messages))


# --- 資料來源 ---------------------------------------------------------------


def test_invisible_characters_exist_only_in_the_raw_fragment() -> None:
    doc = build_document([Message(text=f"請匯{ZERO_WIDTH_SPACE}款到指定帳戶")])

    assert ZERO_WIDTH_SPACE not in doc.text_at((0, 0))
    assert ZERO_WIDTH_SPACE in doc.raw_at((0, 0))


def test_fullwidth_digits_are_folded_in_the_normalized_text() -> None:
    doc = build_document([Message(text="帳號１23")])

    assert doc.text_at((0, 0)) == "帳號123"
    assert "１" in doc.raw_at((0, 0))


def test_spacing_inside_a_word_survives_normalization() -> None:
    """`normalize_text()` 刻意不壓縮連續空白 —— 拆字偵測數的正是那些空白。"""
    doc = build_document([Message(text="請匯到監  管帳戶")])

    assert "監  管" in doc.text_at((0, 0))


def test_zero_width_split_is_covered_by_two_signals_without_a_gap() -> None:
    """零寬字元拆字：規則在 `text` 上照常命中，規避訊號在 `raw` 上命中。"""
    text = f"請把驗證碼傳{ZERO_WIDTH_SPACE}給我"
    doc = build_document([Message(text=text)])
    solicit = next(rule for rule in TIER_A_RULES if rule.name == "solicit_otp")
    req = Request(messages=[Message(text=text)])

    assert solicit(req, doc) != []
    assert run(InvisibleCharCheck(), text) != []


# --- `evasion_invisible` -----------------------------------------------------


def test_zero_width_space_hits() -> None:
    results = run(InvisibleCharCheck(), f"請匯{ZERO_WIDTH_SPACE}款到指定帳戶")

    assert len(results) == 1
    assert "零寬空格" in results[0].detail


def test_family_emoji_does_not_hit() -> None:
    """迴歸：`normalize.INVISIBLE` 含 U+200D，而家庭 emoji 就是用它連接的。"""
    assert run(InvisibleCharCheck(), f"我們一家人 {FAMILY_EMOJI} 出去玩") == []


def test_leading_byte_order_mark_does_not_hit() -> None:
    assert run(InvisibleCharCheck(), f"{BYTE_ORDER_MARK}您好，這是通知") == []


def test_bidi_override_hits() -> None:
    assert run(InvisibleCharCheck(), f"請{RIGHT_TO_LEFT_OVERRIDE}點選連結") != []


def test_soft_hyphen_does_not_hit() -> None:
    assert run(InvisibleCharCheck(), "請匯\u00ad款") == []


def test_invisible_names_cover_the_normalize_constant() -> None:
    """缺項會在使用者的請求上拋 `KeyError`，因此由測試釘住涵蓋關係。"""
    assert set(INVISIBLE_NAMES) == set(INVISIBLE)


# --- `evasion_width_mix` -----------------------------------------------------


def test_width_mix_inside_one_token_hits() -> None:
    results = run(WidthMixCheck(), "帳號１23 請確認")

    assert "帳號１23" in results[0].detail


def test_typographic_mix_across_tokens_does_not_hit() -> None:
    """迴歸：中文排版本來就會句中用全形、網址裡用半形。"""
    assert run(WidthMixCheck(), "請於１０月３１日前至 https://example.com/2026 完成") == []


def test_uniform_fullwidth_does_not_hit() -> None:
    assert run(WidthMixCheck(), "請於１０月３１日完成") == []


# --- `evasion_split_word` ----------------------------------------------------


def _length_then_text(word: str) -> tuple[int, str]:
    return len(word), word


def sample_target() -> str:
    """取目標詞表中最短的一個多字詞 —— 測試不寫死特定詞，Tier-A 擴充時不會壞。"""
    return sorted(TARGET_WORDS, key=_length_then_text)[0]


def test_split_target_word_hits() -> None:
    word = sample_target()

    assert run(SplitWordCheck(), f"請注意{' '.join(word)}的事") != []


def test_intact_target_word_does_not_hit() -> None:
    """原詞由言語行為規則處理，不是規避訊號。"""
    assert run(SplitWordCheck(), f"請注意{sample_target()}的事") == []


def test_separator_must_come_from_the_allowed_set() -> None:
    """迴歸：允許任意字元的話，「匯出款項」裡的「出」會被當成分隔符。"""
    assert run(SplitWordCheck(), "匯出款項已完成") == []


def test_gap_beyond_the_limit_does_not_hit() -> None:
    """迴歸：容錯上限開到 3 以上會讓「匯了一筆款」誤命中。"""
    assert MAX_GAP == 2
    assert run(SplitWordCheck(), "匯了一筆款給他") == []


def test_enumeration_is_not_a_split_word() -> None:
    """頓號與逗號是列舉分隔，不在 `SPLIT_SEPARATORS` 裡。"""
    assert SPLIT_SEPARATORS.isdisjoint({"、", ",", "，"})
    assert run(SplitWordCheck(), "不要把帳號、密碼告訴任何人") == []


def test_spec_example_of_a_split_safe_account() -> None:
    assert "監管帳戶" in TARGET_WORDS
    assert run(SplitWordCheck(), "請匯到監 管 帳 戶") != []


# --- `evasion_homophone` -----------------------------------------------------


def test_variant_table_is_empty_and_the_check_never_hits() -> None:
    """空表是誠實的狀態 —— 樣本還不存在，編變體表是製造假資料。"""
    assert HOMOPHONE_VARIANTS == ()
    assert run(HomophoneCheck(), "請匯款到指定帳戶") == []


def test_mainland_spelling_is_not_a_variant() -> None:
    """「賬號」是中國大陸的標準寫法，收進變體表會在整類轉傳文上系統性誤判。"""
    assert all("賬" not in variant for variant, _ in HOMOPHONE_VARIANTS)
    assert run(HomophoneCheck(), "賬號已開通") == []


def test_variant_table_is_injectable_so_the_match_path_is_tested() -> None:
    check = HomophoneCheck(variants=(("驗証碼", "驗證碼"),))

    results = run(check, "請把驗証碼給我")

    assert "驗証碼" in results[0].detail
    assert "驗證碼" in results[0].detail


# --- `evasion_zhuyin` --------------------------------------------------------


def test_zhuyin_between_han_characters_hits() -> None:
    results = run(ZhuyinCheck(), "請匯ㄊㄞˊ幣五萬到指定帳戶")

    assert "ㄊㄞˊ" in results[0].detail


def test_sentence_final_particle_does_not_hit() -> None:
    assert run(ZhuyinCheck(), "好ㄛ") == []


def test_sentence_initial_particle_does_not_hit() -> None:
    assert run(ZhuyinCheck(), "ㄟ你在嗎") == []


# --- 輸出形狀與計數範圍 ------------------------------------------------------


HIT_SAMPLES: tuple[str, ...] = (
    f"請匯{ZERO_WIDTH_SPACE}款到指定帳戶",
    "帳號１23 請確認",
    "請匯到監 管 帳 戶",
    "請匯ㄊㄞˊ幣五萬到指定帳戶",
)


def test_output_shape_is_uniform() -> None:
    for check in EVASION_CHECKS:
        for text in HIT_SAMPLES:
            for result in run(check, text):
                assert result.hard is False
                assert result.scam_types == []
                assert result.weight == EVASION_WEIGHT == 0.6


def test_evidence_is_never_empty_and_always_resolves() -> None:
    for text in HIT_SAMPLES:
        req = Request(messages=[Message(text=text)])
        doc = build_document(req.messages)
        for check in EVASION_CHECKS:
            for result in check(req, doc):
                assert result.evidence != []
                assert all(doc.index_of(coord) >= 0 for coord in result.evidence)


def test_invisible_character_on_a_sentence_boundary_is_counted() -> None:
    """邊界上的零寬字元被歸入前一句的原文片段，以訊息為範圍計數仍算得到。"""
    doc = build_document([Message(text=f"請匯款。{ZERO_WIDTH_SPACE}請盡快。")])

    assert ZERO_WIDTH_SPACE not in "".join(doc.sentences)
    assert run(InvisibleCharCheck(), f"請匯款。{ZERO_WIDTH_SPACE}請盡快。") != []


def test_evidence_points_at_the_densest_sentence() -> None:
    text = f"請匯{ZERO_WIDTH_SPACE}款。請{ZERO_WIDTH_SPACE}盡{ZERO_WIDTH_SPACE}快。"

    results = run(InvisibleCharCheck(), text)

    assert results[0].evidence == [(0, 1)]
    assert "×3" in results[0].detail


def test_document_is_not_modified() -> None:
    req = Request(messages=[Message(text=HIT_SAMPLES[0])])
    doc = build_document(req.messages)
    before = (doc.sentences, doc.raw_sentences, doc.coords)

    for check in EVASION_CHECKS:
        check(req, doc)

    assert (doc.sentences, doc.raw_sentences, doc.coords) == before


# --- 詞表耦合與註冊 ----------------------------------------------------------


def test_target_words_come_from_the_speech_act_rules() -> None:
    """一份清單勝過兩份 —— 兩份會各自漂移，而漂移不會有任何機制報告。"""
    from_rules = set()
    for rule in TIER_A_RULES:
        from_rules |= rule.predicates | rule.objects | rule.receivers

    assert TARGET_WORDS <= from_rules
    assert TARGET_WORDS.isdisjoint(SELF_DIRECTED)
    assert all(len(word) >= 2 for word in TARGET_WORDS)


def test_five_checks_register_independently() -> None:
    registry = CheckRegistry()
    register_evasion_checks(registry)
    registry.disable("evasion_homophone")

    names = [check.name for check in registry.enabled()]

    assert names == [
        "evasion_invisible",
        "evasion_width_mix",
        "evasion_split_word",
        "evasion_zhuyin",
    ]


def test_all_evasion_checks_are_local() -> None:
    assert [check.name for check in EVASION_CHECKS] == [
        "evasion_invisible",
        "evasion_width_mix",
        "evasion_split_word",
        "evasion_homophone",
        "evasion_zhuyin",
    ]
    assert all(check.stage is Stage.LOCAL for check in EVASION_CHECKS)
