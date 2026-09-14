"""`extract_urls()` 的抽取、正規化、去重與座標歸屬，以及與切句的一致性。"""

import re

import pytest

from scam_guard.normalize import URL_PATTERN, build_document, normalize_text, split_sentences
from scam_guard.types import Message
from scam_guard.url import PublicSuffixList, extract_urls

EXTRA_TLDS = """
// ===BEGIN ICANN DOMAINS===
com
net
tw
com.tw
gov.tw
uk
co.uk
cc
cn
is
me
ly
games
shop
zip
mov
app
dev
sh
py
md
pl
rs
so
// ===END ICANN DOMAINS===
// ===BEGIN PRIVATE DOMAINS===
wixsite.com
// ===END PRIVATE DOMAINS===
"""


@pytest.fixture(name="psl")
def psl_fixture() -> PublicSuffixList:
    return PublicSuffixList.parse(EXTRA_TLDS)


def doc_of(*texts: str):
    return build_document([Message(text=text) for text in texts])


def urls_of(psl: PublicSuffixList, *texts: str):
    return extract_urls(doc_of(*texts), psl)


# --- 基本 -----------------------------------------------------------------


def test_empty_document_returns_empty(psl: PublicSuffixList) -> None:
    assert extract_urls(doc_of(""), psl) == []


def test_fullwidth_url_is_flattened(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "ｈｔｔｐｓ：／／ｅｖｉｌ．ｃｏｍ")
    assert extracted.host == "evil.com"
    assert extracted.scheme_present is True


def test_zero_width_split_host_is_recovered(psl: PublicSuffixList) -> None:
    doc = doc_of("請點 https://ev​il.com/a 領取")
    (extracted,) = extract_urls(doc, psl)
    assert extracted.host == "evil.com"
    # 規避痕跡沒有遺失 —— 它在原文片段裡，呈現層本來就取那個。
    assert "​" in doc.raw_at(extracted.coords[0])


def test_bare_shortener_is_extracted(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "包裹配送失敗 reurl.cc/2xY3z 請更新地址")
    assert extracted.host == "reurl.cc"
    assert extracted.scheme_present is False


def test_extension_like_tlds_are_not_domains(psl: PublicSuffixList) -> None:
    assert urls_of(psl, "請查收 報告.zip 與 readme.md") == []


def test_extension_like_tld_with_scheme_is_extracted(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "https://report.zip/a")
    assert extracted.host == "report.zip"


def test_non_tld_is_not_extracted(psl: PublicSuffixList) -> None:
    assert urls_of(psl, "他住在 U.S.A 已經十年") == []


# --- 尾綴剝除 -------------------------------------------------------------


def test_trailing_question_mark_is_stripped(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "你收到這個了嗎 https://evil.com/a?")
    assert extracted.url == "https://evil.com/a"


def test_query_separators_are_preserved(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "看這裡 https://x.cc/a?id=1;v=2 謝謝")
    assert extracted.url == "https://x.cc/a?id=1;v=2"


def test_unpaired_closing_bracket_is_stripped(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "（詳見 https://evil.com/a）")
    assert extracted.url == "https://evil.com/a"


def test_paired_brackets_are_preserved(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "見 https://x.cc/wiki/A_(B) 一文")
    assert extracted.url == "https://x.cc/wiki/A_(B)"


# --- 主機的解析 -----------------------------------------------------------


def test_userinfo_is_not_the_host(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "https://post.gov.tw@evil.com/login")
    assert extracted.host == "evil.com"
    assert extracted.has_userinfo is True


def test_case_and_port_are_normalized(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "HTTPS://WWW.Evil.COM:8443/a")
    assert extracted.host == "evil.com"


def test_idn_host_becomes_punycode(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "https://中国.com/a")
    assert extracted.host == "xn--fiqs8s.com"
    assert extracted.raw_host == "中国.com"


def test_ip_literal(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "http://192.0.2.1/login")
    assert extracted.is_ip_literal is True
    assert extracted.registrable_domain is None


def test_idna_failure_is_kept_as_a_signal(psl: PublicSuffixList) -> None:
    """編碼失敗不丟棄 —— 丟棄會讓一個刻意構造的主機躲過全部 URL 檢查。"""
    # punycode 化之後超過 63 bytes 的標籤，stdlib 的 IDNA2003 編碼會拒絕。
    long_label = "中" * 60
    (extracted,) = urls_of(psl, f"https://{long_label}.com/a")
    assert extracted.idna_failed is True
    assert extracted.registrable_domain is None


def test_chinese_glued_to_a_pathless_url_is_trimmed(psl: PublicSuffixList) -> None:
    """Cofacts 樣本實測到的形態：主機後直接接中文，且沒有路徑分隔符。

    628 個抽取結果中有 3 個長這樣。不處理的話 `host` 會是一段中文，
    而黑名單比對的鍵就成了那段中文。
    """
    (extracted,) = urls_of(psl, "請至 https://6000.gov.tw會在3月22日8時起開放登記")
    assert extracted.host == "6000.gov.tw"
    assert extracted.url == "https://6000.gov.tw"


def test_idn_host_is_not_trimmed(psl: PublicSuffixList) -> None:
    """上一條的條件寫得很窄，合法的 IDN 主機不受影響。"""
    (extracted,) = urls_of(psl, "https://中国.com")
    assert extracted.host == "xn--fiqs8s.com"


def test_registrable_domain_uses_psl(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "https://www.esunbank.com.tw/login")
    assert extracted.registrable_domain == "esunbank.com.tw"


# --- 去重與順序 -----------------------------------------------------------


def test_same_url_twice_is_one_result_with_two_coords(psl: PublicSuffixList) -> None:
    results = urls_of(psl, "點這裡 https://evil.com/a。或複製 https://evil.com/a 到瀏覽器")
    assert len(results) == 1
    assert len(results[0].coords) == 2


def test_different_paths_are_two_results(psl: PublicSuffixList) -> None:
    results = urls_of(psl, "https://evil.com/a 與 https://evil.com/b")
    assert [r.url for r in results] == ["https://evil.com/a", "https://evil.com/b"]


def test_coords_span_messages(psl: PublicSuffixList) -> None:
    doc = doc_of("你好", "先看這個。https://evil.com/a 很重要")
    (extracted,) = extract_urls(doc, psl)
    assert extracted.coords == ((1, 1),)
    assert doc.raw_at((1, 1)).strip().startswith("https://evil.com/a")


def test_extraction_is_stable(psl: PublicSuffixList) -> None:
    doc = doc_of("https://a.com/1 https://b.com/2 reurl.cc/x")
    assert extract_urls(doc, psl) == extract_urls(doc, psl)


def test_no_risk_fields_on_extracted_url(psl: PublicSuffixList) -> None:
    (extracted,) = urls_of(psl, "https://evil.com/a")
    fields = set(type(extracted).__dataclass_fields__)
    assert fields == {
        "url",
        "host",
        "raw_host",
        "registrable_domain",
        "is_ip_literal",
        "has_userinfo",
        "scheme_present",
        "idna_failed",
        "coords",
    }


# --- 與切句的一致性 -------------------------------------------------------


SAMPLES = (
    "請點 https://evil.com/a?id=1;v=2 領取獎金!",
    "（詳見 https://x.cc/wiki/A_(B)）真的嗎?",
    "www.evil.com/a 與 https://good.com/b。",
    "你收到這個了嗎 https://evil.com/a?",
)


@pytest.mark.parametrize("sample", SAMPLES)
def test_scheme_matches_lie_inside_the_split_mask(sample: str, psl: PublicSuffixList) -> None:
    """帶 scheme 的抽取結果其字元區間 MUST 完全落在切句遮罩區間內。

    守住的是「抽取器不會抓到一段被切過的 URL」。遮罩允許寬鬆（寧可少切
    不可切壞），抽取器必須嚴格，兩者的關係就是這條包含關係。
    """
    text = normalize_text(sample).text
    mask = [(m.start(), m.end()) for m in URL_PATTERN.finditer(text)]
    for match in re.finditer(r"(?i)https?://[^\s<>\"'）」』】]+", text):
        assert any(start <= match.start() and match.end() <= end for start, end in mask), (
            sample,
            match.group(),
        )


def test_bare_url_host_survives_sentence_split(psl: PublicSuffixList) -> None:
    """無 scheme 的寫法不在遮罩保護內，`?` 會切開它 —— 但主機仍完整。"""
    doc = doc_of("reurl.cc/abc?id=1")
    assert len(doc.sentences) == 2
    (extracted,) = extract_urls(doc, psl)
    assert extracted.host == "reurl.cc"


def test_scheme_url_stays_in_one_sentence() -> None:
    pairs = split_sentences(normalize_text("https://x.cc/a?id=1;v=2"))
    assert len(pairs) == 1


def test_url_pattern_is_unchanged() -> None:
    """切句遮罩與抽取器是兩個東西，兩個都要留。"""
    assert URL_PATTERN.pattern == r"https?://\S+|www\.\S+"
