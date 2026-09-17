"""LINE adapter 的四層 registry 組裝 —— 自己一份，比照 `api/app.py`。

規則 + 分類器 + URL 層 + Gemini 語意層。與其他介面共用同一個 `detect()`、
同一張 `weights.toml`，差別只在「掛哪幾層」與「LLM 用哪個 runtime」。這裡的 LLM
是 Gemini（伺服器端，不用裝 GGUF），不是 llama.cpp。
"""

import os
from pathlib import Path

from llm_runtime.gemini import GeminiRuntime, GeminiUnavailable
from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.check import CheckRegistry
from scam_guard.llm.check import LlmCheck
from scam_guard.llm.prompt import DEFAULT_BUDGET
from scam_guard.llm.validate import LlmOutcomeCounter
from scam_guard.ngram import load_model, register_ngram_check
from scam_guard.rules.evasion import register_evasion_checks
from scam_guard.rules.quotation import QuotationCheck
from scam_guard.rules.speech_act import register_speech_act_rules
from scam_guard.url import PublicSuffixList
from scam_guard.url_check import load_tables, register_url_checks
from scam_guard.weights import load_weights

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PSL_DIR = DATA_DIR / "psl"
BLOCKLIST_DIR = DATA_DIR / "blocklist"
ALLOWLIST_DIR = DATA_DIR / "allowlist"
PSL_MAX_AGE_DAYS = 90
BLOCKLIST_MAX_AGE_DAYS = {"176455": 60, "165027": 60}
ALLOWLIST_MAX_AGE_DAYS = 30

TABLE = load_weights()
NGRAM_MODEL = load_model()
LLM_DEADLINE_S = 25.0
"""Gemini 呼叫的上界。LINE 的 reply_token 約 30 秒，留 5 秒給組回覆與網路。"""


def _load_url_layer() -> tuple[
    PublicSuffixList | None, BlocklistStore | None, RankAllowlist | None
]:
    """載 PSL / 黑名單 / 白名單快照。PSL 缺席即整個 URL 層不註冊（不致命、不空 PSL）。

    黑白名單成對：任一缺席即兩者皆 `None`，`url_blocklist` 因此不註冊。缺席是正常
    首次狀態 —— `data/` 是 operator-local 且 gitignored。
    """
    if not (PSL_DIR / "public_suffix_list.dat").exists():
        return None, None, None
    psl = PublicSuffixList.load(PSL_DIR, max_age_days=PSL_MAX_AGE_DAYS)
    store: BlocklistStore | None = None
    allowlist: RankAllowlist | None = None
    if (BLOCKLIST_DIR / "manifest.json").exists() and (ALLOWLIST_DIR / "manifest.json").exists():
        store = BlocklistStore.load(BLOCKLIST_DIR, psl, max_age_days=BLOCKLIST_MAX_AGE_DAYS)
        allowlist = RankAllowlist.load(ALLOWLIST_DIR, max_age_days=ALLOWLIST_MAX_AGE_DAYS)
    return psl, store, allowlist


def _gemini_check() -> LlmCheck | None:
    """`GEMINI_API_KEY` 有設就掛 Gemini 語意層；沒設就不掛（誠實降級為三層）。

    用 `GeminiRuntime` 建構時的 `GeminiUnavailable`（key 缺席即拋）判斷，不是 try 整段
    網路呼叫 —— key 在不在是啟動就知道的事。
    """
    try:
        runtime = GeminiRuntime()
    except GeminiUnavailable:
        return None
    return LlmCheck(
        runtime=runtime,
        counter=LlmOutcomeCounter(),
        table=TABLE,
        budget=DEFAULT_BUDGET,
        deadline_s=LLM_DEADLINE_S,
    )


def build_registry() -> CheckRegistry:
    """LINE adapter 的四層 registry。每落地一項檢查此處多一行，比照 `api/app.py`。"""
    registry = CheckRegistry()
    register_speech_act_rules(registry)
    register_evasion_checks(registry)
    registry.register(QuotationCheck())
    register_ngram_check(registry, TABLE, model=NGRAM_MODEL)
    psl, store, allowlist = _load_url_layer()
    if psl is not None:
        register_url_checks(registry, psl, load_tables(), store=store, allowlist=allowlist)
    llm_check = _gemini_check()
    if llm_check is not None:
        registry.register(llm_check)
    return registry


REGISTRY = build_registry()
LLM_ENABLED = "GEMINI_API_KEY" in os.environ and os.environ.get("GEMINI_API_KEY", "") != ""
