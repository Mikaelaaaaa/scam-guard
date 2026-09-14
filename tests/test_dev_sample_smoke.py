"""把開發樣本的每一筆送進 `detect()`，確認流程跑得完。

**跑得完，不是跑得對。** 這批資料沒有人工標註，沒有可以對照的答案，因此
**不對判定結果做任何準確率斷言** —— 尤其不碰 `scam_probability` 的數值
（此階段它固定為 `None`，斷言它等於任何東西都只是在測佔位值）。

這個測試回答的是「真實輸入不會讓 pipeline 炸掉」：10,000 字元的單則、
只有一行 URL 的訊息、內含零寬字元與表情符號的訊息 ——
這些都是手寫測試字串寫不出來的形狀。

資料檔不進版控（真實民眾訊息、CC BY-SA 4.0），因此 CI 與別人的機器上本來就
沒有這個檔案，缺檔時 skip。skip 訊息帶產生資料的完整命令，否則它會變成一個
永遠綠燈、永遠沒跑過的測試。
"""

import json
from pathlib import Path

import pytest

from scam_guard.check import CheckRegistry
from scam_guard.pipeline import detect
from scam_guard.types import Request, Verdict

DEV_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "dev_sample.jsonl"

HOW_TO_GET_DATA = f"""找不到開發樣本 {DEV_SAMPLE}。產生方式：

  python -m tools.cofacts_fetch --label scam --out data/cofacts_scam.jsonl
  python -m tools.cofacts_fetch --label hard-negative --out data/cofacts_hard_negative.jsonl
  python -m tools.message_filter data/cofacts_scam.jsonl data/cofacts_hard_negative.jsonl \\
      --out data/dev_sample.jsonl

資料不進版控（真實民眾訊息，且 Cofacts 開放資料為 CC BY-SA 4.0），缺檔是正常的。"""


def test_dev_sample_runs_through_detect() -> None:
    if not DEV_SAMPLE.exists():
        pytest.skip(HOW_TO_GET_DATA)

    registry = CheckRegistry()
    expected_checks = len(registry.enabled())
    count = 0
    # 逐行迭代檔案物件，**不用 `str.splitlines()`** —— 後者也在 U+2028 等
    # Unicode 行邊界上切，而真實訊息裡就有這些字元（`json.dumps` 在
    # `ensure_ascii=False` 下不會轉義它們），切下去就把一行 JSON 切成兩半。
    # 這正是 smoke test 要抓的那種「手寫測試字串寫不出來的形狀」。
    with DEV_SAMPLE.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            verdict = detect(Request.from_text(record["text"]), registry)
            assert isinstance(verdict, Verdict)
            assert len(verdict.checks) == expected_checks
            count += 1
    assert count > 0
