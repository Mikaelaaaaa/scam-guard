"""檢查的統一介面與註冊機制。

新增檢查不需修改主流程；任一檢查可被停用，供消融實驗使用。
"""

from enum import Enum
from typing import Any, Protocol, TypeAlias

from scam_guard.types import CheckResult, Request

# `Document` 為正規化與切句後的結果，屬 `add-text-normalize`，此時尚未實作。
# 暫以別名佔位（而非建立臨時實作或 import 不存在的模組）。該 change 落地後
# 改為 `if TYPE_CHECKING: from scam_guard.normalize import Document`。
Document: TypeAlias = Any


class Stage(Enum):
    """檢查的成本階段。pipeline 的短路只跳過 `EXPENSIVE`。

    `LOCAL` —— 文字規則、本機黑名單比對，毫秒級，一律執行。本機檢查的成本低到
    不值得為它設計跳過邏輯，且全部執行能得到完整的訊號圖像，對消融實驗有利。

    `EXPENSIVE` —— RDAP、LLM 等外部呼叫，延遲與費用高數個數量級，可被短路。

    階段由檢查自報而非由 registry 指定 —— 檢查自己知道它貴不貴。
    已知風險：標錯就防不住（把 LLM 標成 `LOCAL` 會使它永遠不被短路）。
    """

    LOCAL = "local"
    EXPENSIVE = "expensive"


class Check(Protocol):
    """一個檢查。函式與類別實例皆可滿足 —— 這是結構型別，不需繼承任何基底類別。

    同時接收 `req` 與 `doc` 的理由：多數檢查只需要 `doc`（已正規化、已切句），
    但少數需要 `req`（通道特徵需要 `sender`、軌跡判斷需要 `sent_at`）。

    回傳 `list` 而非單一結果，使一個檢查可產出多個訊號 —— 訊息含三個 URL 時，
    URL 檢查應對每個各產一筆，各有自己的 `detail` 與 `evidence`。

    **未命中時 MUST 回傳空陣列**，不回傳 `hit=False` 的佔位結果：讓「有沒有訊號」
    從陣列長度直接看出。`Verdict.checks` 中的未命中記錄由 pipeline 補上，
    檢查本身不需要製造它。

    **依賴外部服務的檢查 MUST 自行捕捉例外並記錄失敗原因，回傳空陣列**，
    不得讓例外向上傳播 —— 「哪些檢查會失敗」是檢查自己的知識，
    不該讓 pipeline 為每個檢查寫 try/except。回傳空陣列等同「此檢查未提供訊號」，
    下游的信心值計算會自然反映證據不足。

    注意 `Protocol` 不在執行期強制：註冊一個簽章不符的物件不會立刻報錯，
    只在呼叫時才炸。`CheckRegistry.register()` 因此做基本驗證。
    """

    name: str
    stage: Stage

    def __call__(self, req: Request, doc: Document) -> list[CheckResult]: ...


class CheckRegistry:
    """檢查的註冊表，顯式實例而非全域狀態。

    用顯式 registry 而非 `@register` 裝飾器：裝飾器讓註冊發生在 import 時，
    測試要隔離就得操作全域狀態；顯式 registry 可在測試中建新實例，互不干擾。
    消融實驗也需要同時跑兩組不同設定。

    行為約定：

    - `register()` 對**重複名稱拋 `ValueError`**，不覆蓋。同名的兩個檢查
      通常是誤註冊，覆蓋會靜默丟失其中一個。要換掉既有檢查請先建新 registry。
    - `disable()` / `enable()` 對**未註冊的名稱拋 `KeyError`**，不安靜忽略。
      消融實驗常以字串指定要關閉的檢查，拼錯名稱時安靜忽略會讓整組實驗
      悄悄變成對照組。
    """

    def __init__(self) -> None:
        self._checks: dict[str, Check] = {}
        self._disabled: set[str] = set()

    def register(self, check: Check) -> None:
        """加入一個檢查。介面不符時拋 `TypeError`，名稱重複時拋 `ValueError`。"""
        name = getattr(check, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"檢查必須具備非空的字串 name 屬性，{check!r} 不符合 Check 介面"
            )
        if not callable(check):
            raise TypeError(
                f"檢查必須可呼叫，簽章為 (req, doc) -> list[CheckResult]，"
                f"{name!r} 不符合 Check 介面"
            )
        stage = getattr(check, "stage", None)
        if not isinstance(stage, Stage):
            raise TypeError(
                f"檢查必須具備 Stage 型別的 stage 屬性（LOCAL 或 EXPENSIVE），"
                f"{name!r} 的 stage 為 {stage!r}"
            )
        if name in self._checks:
            raise ValueError(f"檢查名稱重複：{name!r} 已註冊於此 registry")
        self._checks[name] = check

    def disable(self, name: str) -> None:
        """停用一個檢查，供消融實驗使用。名稱未註冊時拋 `KeyError`。"""
        if name not in self._checks:
            raise KeyError(f"無此檢查：{name!r}")
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        """重新啟用一個檢查。名稱未註冊時拋 `KeyError`。"""
        if name not in self._checks:
            raise KeyError(f"無此檢查：{name!r}")
        self._disabled.discard(name)

    def enabled(self) -> list[Check]:
        """已啟用的檢查，順序依註冊順序，穩定。"""
        return [check for name, check in self._checks.items() if name not in self._disabled]
