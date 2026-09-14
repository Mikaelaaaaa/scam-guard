"""LINE 聊天記錄匯出檔的解析 —— 承接 `add-message-filter` 的 `line_export` 旗標。

該 change 的交棒寫得很直接：「若 `add-testset` 不接手，這個旗標就白標了」。
本模組是接手的那一半：把一則被標為 `line_export` 的 Cofacts 文章還原成
`list[Message]`，使多則情境有真實素材可測。

格式固定，真實例子：

    [LINE] 與金融正義處理陳先生的聊天記錄
    儲存日期： 2024/09/10 11:35

    2024/08/19（一）
    上午09:53	小麥	 您好
    上午09:54	金融正義處理陳先生	Anya把您的情況大概跟我講了一下⋯

三欄正好對上 `Message(text, sender, sent_at)`：時間、暱稱、內容。

**未涵蓋的格式一律 raise 並指出行號，不猜。** 跨行續行、貼圖與圖片的佔位文字、
暱稱含定位字元都在此列。猜的後果是把一段對話還原成錯的訊息邊界，而那會讓
跨訊息的軌跡規則（`relationship_building`）讀到一個不存在的對話 ——
沒有任何機制會報告這件事。5 至 100 則的規模人工修得完。
"""

import re
from datetime import datetime

from scam_guard.types import Message

HEADER_PATTERN = re.compile(r"\[LINE\][^\n]*聊天記錄")
"""匯出檔標頭。與 `tools.message_filter.LINE_EXPORT_PATTERN` 同一個樣式。"""

SAVED_AT_PREFIX = "儲存日期："

DATE_PATTERN = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})(?:（.）|\(.\))?$")
"""日期分隔行，例：`2024/08/19（一）`。星期欄位有全形與半形兩種寫法。"""

TIME_PATTERN = re.compile(r"^(上午|下午)?(\d{1,2}):(\d{2})$")
"""時間欄，例：`上午09:53`。無 `上午`/`下午` 前綴時視為 24 小時制。"""

NOON = 12


class LineExportError(Exception):
    """匯出檔含本解析器未涵蓋的格式。訊息一律含行號。"""


def _to_time(field: str, line_number: int) -> tuple[int, int]:
    match = TIME_PATTERN.match(field)
    if match is None:
        raise LineExportError(f"第 {line_number} 行的時間欄無法解析：{field!r}")
    meridiem, hour_text, minute_text = match.groups()
    hour = int(hour_text)
    if meridiem == "上午" and hour == NOON:
        hour = 0
    elif meridiem == "下午" and hour != NOON:
        hour += NOON
    if not 0 <= hour < 24:
        raise LineExportError(f"第 {line_number} 行的時數超出範圍：{field!r}")
    return hour, int(minute_text)


def parse(text: str) -> list[Message]:
    """把一則匯出檔解析成多則訊息。標頭缺席或格式未涵蓋時 raise。

    `sender` 保留匯出檔中的暱稱原樣 —— 匿名化是 adapter 的決定，
    這裡若先做了，`add-message-filter` 標出的那些對話就不能用來測發送者切換。
    """
    lines = text.splitlines()
    if not lines or HEADER_PATTERN.search(lines[0]) is None:
        raise LineExportError("第 1 行不是 LINE 匯出檔標頭（`[LINE] …聊天記錄`）")

    messages: list[Message] = []
    current_date: tuple[int, int, int] | None = None
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        if line.lstrip().startswith(SAVED_AT_PREFIX):
            continue
        date_match = DATE_PATTERN.match(line.strip())
        if date_match is not None:
            current_date = (
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            )
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise LineExportError(
                f"第 {line_number} 行不是三欄的訊息行（實際 {len(fields)} 欄）："
                f"跨行續行、貼圖佔位與暱稱含定位字元皆未涵蓋，本解析器不猜。"
                f"該行前 40 字元：{line[:40]!r}"
            )
        if current_date is None:
            raise LineExportError(f"第 {line_number} 行出現訊息，但其前沒有任何日期分隔行")
        hour, minute = _to_time(fields[0].strip(), line_number)
        year, month, day = current_date
        messages.append(
            Message(
                text=fields[2].strip(),
                sender=fields[1].strip(),
                sent_at=datetime(year, month, day, hour, minute),
            )
        )
    if not messages:
        raise LineExportError("匯出檔中沒有任何訊息行")
    return messages
