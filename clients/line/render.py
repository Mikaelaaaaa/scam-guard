"""把 `Verdict` 組成 LINE 文字回覆。

依據直接取自 `verdict.evidence`（呈現層已組裝好的字串），不重新生成 —— 與其他介面
的「依據是組裝不是生成」一致。純文字，因為 LINE 的文字訊息不吃 HTML。
"""

from scam_guard.types import Verdict

DECISION_HEADER = "⚠️ 很可能是詐騙"
ABSTAIN_HEADER = "無法判定"
ABSTAIN_BODY = "系統沒有找到足夠的依據。這不代表它安全，只代表沒有看出足夠訊號。"
HOTLINE = "不確定的時候，撥打 165 反詐騙專線查證。"


def render_verdict(verdict: Verdict) -> str:
    """判定 → 回覆字串。可能性為 `None`（拒答）與有值（判定）兩種收場。

    注意：`DECISION_HEADER` 的 ⚠️ 是給收到詐騙訊息的一般使用者的視覺警示，不是文件；
    文件禁 emoji 的規範不涵蓋 bot 對終端使用者的警示訊息。
    """
    lines: list[str] = []
    if verdict.scam_probability is None:
        lines.append(ABSTAIN_HEADER)
        lines.append(ABSTAIN_BODY)
    else:
        percent = round(verdict.scam_probability * 100)
        lines.append(f"{DECISION_HEADER}（{percent}%）")
        if verdict.scam_type is not None:
            lines.append(f"類型：{verdict.scam_type.value}")
    if verdict.evidence:
        lines.append("")
        lines.append("依據：")
        lines.extend(f"・{line}" for line in verdict.evidence)
    lines.append("")
    lines.append(HOTLINE)
    return "\n".join(lines)
