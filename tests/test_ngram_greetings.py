"""重訓後的迴歸夾具：台灣問候與閒聊作為單則訊息 MUST 單獨不命中。

動機（`add-conversation-ham`）：重訓前，分類器把正常對話大量誤判為詐騙 ——
以現行門檻 0.376648 量，八則問候有七則命中（本機實測 2026-09-15）：

    你好 +1.37 命中、哈囉 +0.46 命中、謝謝你的幫忙 +1.14 命中、
    晚點回你 +1.02 命中、在嗎 +0.73 命中、方便講電話嗎 +0.45 命中、
    嗨 +0.47 命中、早安 +0.41 命中；只有「今天天氣真好」−0.04 未中。

根源是訓練集的 ham（Cofacts 宣導文 + 廣告）不含口語問候，`你`/`好 `/`你好`
這些 n-gram 因此全是正權重。本 change 補了一批 PTT 口語對話當 hard-negative 重訓，
讓分類器學過「你好也可能是正常的」。

**夾具為手寫，MUST NOT 取自訓練語料** —— 測在訓練集上等於沒測，且避免授權問題。
`test_fixtures_are_not_in_the_training_corpus` 斷言每則夾具的正規化文字不在
`data/conversation/` 的訓練語料中（該語料 operator-local，缺席時 skip）。

**驗收是每則逐一不命中。** 某則重訓後仍命中 → 這條測試紅，觸發 design Decision 六的
「加量或換 class_weight」，MUST NOT 以調鬆夾具或放寬門檻吸收（那會把大聲的失敗
換成安靜的錯答案）。
"""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from scam_guard.ngram import (
    NGRAM_THRESHOLD,
    NgramClassifierCheck,
    document_text,
    load_model,
    score,
)
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.types import Message, Request
from scam_guard.weights import load_weights

# 手寫台灣問候與閒聊。前九則涵蓋動機中的八則加「今天天氣真好」，其餘為日常口語回覆。
GREETINGS: tuple[str, ...] = (
    "你好",
    "嗨",
    "哈囉",
    "早安",
    "謝謝你的幫忙",
    "晚點回你",
    "今天天氣真好",
    "在嗎",
    "方便講電話嗎",
    "好喔沒問題",
    "我等等打給你",
    "今天要不要一起吃午餐",
    "麻煩你了謝謝",
    "剛剛在忙不好意思",
)

_CONVERSATION_DIR = Path("data/conversation")
_SUBSET_FILE = _CONVERSATION_DIR / "conversation_ham.jsonl"


def _normalized(text: str) -> str:
    return document_text(build_document([Message(text=text)], DEFAULT_LIMITS))


@pytest.mark.parametrize("greeting", GREETINGS)
def test_greeting_does_not_hit_as_a_single_message(greeting: str) -> None:
    model = load_model()
    table = load_weights()
    threshold = table.threshold(NGRAM_THRESHOLD)
    value = score(model, _normalized(greeting)).value
    assert value < threshold, (
        f"問候「{greeting}」分數 {value:.4f} ≥ 門檻 {threshold}：重訓後仍單獨命中。"
        "MUST NOT 調鬆夾具或放寬門檻 —— 觸發 design Decision 六的加量或 class_weight。"
    )


@pytest.mark.parametrize("greeting", GREETINGS)
def test_greeting_check_returns_no_result(greeting: str) -> None:
    """走完整的 `NgramClassifierCheck.__call__`：不命中即回空清單。"""
    model = load_model()
    table = load_weights()
    check = NgramClassifierCheck(model=model, table=table)
    request = Request.from_text(greeting)
    assert check(request, build_document(request.messages, DEFAULT_LIMITS)) == []


@pytest.mark.skipif(
    not _SUBSET_FILE.is_file(),
    reason="data/conversation 未取得（operator-local，CI 上不存在）",
)
def test_fixtures_are_not_in_the_training_corpus() -> None:
    """夾具句 MUST NOT 出現在訓練語料中 —— 由去重守則（正規化文字的 sha256）保證。"""
    ids = {
        json.loads(line)["id"]
        for line in _SUBSET_FILE.read_text("utf-8").splitlines()
        if line.strip()
    }
    for greeting in GREETINGS:
        digest = sha256(_normalized(greeting).encode("utf-8")).hexdigest()
        assert digest not in ids, f"夾具「{greeting}」的正規化文字出現在訓練語料中"
