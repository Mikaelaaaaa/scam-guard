"""權重表的載入、驗證與查詢 —— 訊號清冊、分群、角色、門檻與類型優先序。

**權重的尺度是對數似然比：**

    w(訊號) = ln( P(訊號命中 | 詐騙) / P(訊號命中 | 合法) )

這不是任意選的尺度。若訊號條件獨立，後驗對數勝算等於先驗對數勝算加上各訊號的
`w` 之和 —— 也就是 `add-score-compute` 的加權求和。**求和之所以合法，唯一的
理由就是權重是對數似然比。** 若權重是「重要性分數」之類的東西，相加沒有意義，
只是把幾個沒有單位的數字放在一起。同一個定義也說明了「同群組取 max」在補什麼：
相加假設條件獨立，而同群組的訊號明顯不獨立。

**本模組只讀不寫，且 MUST NOT 在 import 時讀檔。** `DEFAULT_WEIGHTS_PATH` 是一個
路徑常數，不是一份已載入的表：`import scam_guard.weights` 失敗會讓**所有**測試
無法收集，而一個沒跑過安裝的環境不該在 import 階段就炸。載入由呼叫端顯式執行。

讀本機 `weights.toml` 不違反「`scam_guard/` 不做 I/O」：那條界線畫在
「誰知道外部格式」（`add-blocklist-store` 的原話），而這份表是我們自己定義的
格式，與 `scam_guard/tables/*.json` 是同一類東西。

**解析只用標準庫。** `tomllib` 自 Python 3.11 進入 stdlib，而
`requires-python = ">=3.11"`。選 TOML 而非 YAML 的唯一理由就是後者需要一個
執行期依賴，而「不新增執行期依賴」在四份已定案的 spec 裡都是明寫的條件。

**載入失敗一律拋例外，不回退為任何內建的預設表。** 一份讀不進來的權重表與一份
悄悄變成預設值的權重表，前者是可以修的錯誤，後者是一份看起來正常的錯誤答案。
"""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from scam_guard.check import CheckRegistry
from scam_guard.types import ScamType

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "tables" / "weights.toml"
"""權重表的預設位置。**這是一個路徑，不是一份已載入的表** —— 見模組 docstring。"""

BASIS_MEASURED = "measured"
BASIS_PLACEHOLDER = "placeholder"
BASIS_VALUES = (BASIS_MEASURED, BASIS_PLACEHOLDER)

PLACEHOLDER_WEIGHTS: frozenset[float] = frozenset({2.5, 0.6, 0.0, -1.5})
"""`basis = "placeholder"` 的權重唯一允許的四個值，每一個都可追溯到既有的 change。

| 值 | 出處 |
|---|---|
| `2.5` | `add-speech-act-rules` 的 Tier-A 佔位值 |
| `0.6` | `add-speech-act-rules` 的 Tier-B 佔位值（亦為自帶碼豁免的降級值） |
| `0.0` | `add-url-check` 的 `url_shortener`：宣告「目的地未知」而非「連結不可信」 |
| `-1.5` | `add-quotation-check` 的引述負權重 |

封閉集合是本設計的重點：想寫一個新數字就必須標成 `measured`，而 `measured`
必須附兩個條件機率，而兩個機率必須算得出那個數字。**沒有一條路可以在沒有量測
的情況下產生一個新數字。** 這比在文件裡寫一句「請不要編數字」強得多 ——
後者不會有任何機制報告它被違反。
"""

MEASURED_TOLERANCE = 0.01
"""`measured` 條目的 `value` 與 `ln(p_scam / p_ham)` 允許的最大差距。"""

PROBABILITY_FIELDS = ("p_hit_given_scam", "p_hit_given_ham")

POSITIVE_THRESHOLDS = ("sigmoid_temperature",)
"""必須為正值的門檻，於載入時擋下。

`sigmoid_temperature` 是除數。0 或負值不是「設定得不好」而是壞掉的設定，
而它的後果會出現在計分深處（除以零，或一條方向相反的機率曲線）。
在邊界擋下使 `scam_guard.scoring` 不需要為它加一個防禦性分支。
"""


@dataclass(frozen=True)
class SignalWeight:
    """一個權重條目。`basis` 決定哪些欄位存在，兩種 basis 要求的欄位不重疊。

    `measured` 攜帶兩個條件機率與量測來源；`placeholder` 攜帶它繼承自哪裡、
    以及哪個 change 會取代它。TOML 沒有 null，「尚無資料」以欄位不存在表達，
    而不存在本身就是被檢查的條件 —— 讀取路徑上因此不需要任何預設值。
    """

    value: float
    basis: str
    inherited_from: str | None = None
    blocked_on: str | None = None
    p_hit_given_scam: float | None = None
    p_hit_given_ham: float | None = None
    measured_on: str | None = None


@dataclass(frozen=True)
class Signal:
    """一個訊號的登錄：分群、證據屬性，以及一到兩個權重條目。

    `hard_capable` 在資訊上是冗餘的（可由 `weight_hard` 在不在推出），保留它的
    理由與 `Document` 驗證 `truncated == (dropped_messages > 0)` 相同：它讓
    「我打算讓這個訊號可以是硬證據」成為一句被寫下來的話，漏寫 `weight_hard`
    時會被指名，而不是安靜地變成另一種形狀。
    """

    name: str
    group: str
    hard_capable: bool
    weight_soft: SignalWeight
    weight_hard: SignalWeight | None = None
    strong: bool = False


@dataclass(frozen=True)
class Threshold:
    """一個門檻。與權重共用 `basis` 規則，但佔位門檻要求的是 `rationale`。

    門檻沒有「繼承自某個既有佔位值」這回事，所以 `placeholder` 的門檻改為
    MUST 附 `rationale`（此值是從什麼推出來的）與 `blocked_on`（誰會取代它）。
    """

    value: float
    basis: str
    rationale: str | None = None
    blocked_on: str | None = None
    p_hit_given_scam: float | None = None
    p_hit_given_ham: float | None = None
    measured_on: str | None = None


@dataclass(frozen=True)
class WeightTable:
    """已載入的權重表。不可變 —— 需要改過的表時由既有實例衍生，不就地修改。

    `groups` 由 `signals` 推導，於建構時算一次：`add-score-compute` 的同群組取
    max 與 `add-confidence` 的群組數必須用同一份分群，兩處各自分群會在分群表
    改動時不同步，出現「分數算一次、信心算三次」。
    """

    path: Path
    signals: Mapping[str, Signal]
    roles: Mapping[str, str]
    thresholds: Mapping[str, Threshold]
    type_priority: tuple[ScamType, ...]
    groups: Mapping[str, tuple[str, ...]] = field(init=False)

    def __post_init__(self) -> None:
        grouped: dict[str, list[str]] = {}
        for signal in self.signals.values():
            grouped.setdefault(signal.group, []).append(signal.name)
        object.__setattr__(self, "signals", MappingProxyType(dict(self.signals)))
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))
        object.__setattr__(
            self,
            "groups",
            MappingProxyType({group: tuple(names) for group, names in grouped.items()}),
        )

    def weight_for(self, name: str, hard: bool) -> float:
        """以 `(CheckResult.name, CheckResult.hard)` 取得權重。

        未登錄的訊號拋 `KeyError`，MUST NOT 以 0 略過 —— 一個沒有權重的訊號
        悄悄不計分，等於少一個證據而沒有任何地方會說。

        `hard_capable = false` 的訊號回報 `hard=True` 時拋 `ValueError`，
        MUST NOT 改用 `weight_soft` 代替：這是表對 `hard` 標錯的交叉檢查，
        改用代替值會把大聲的失敗變回安靜的錯誤。
        """
        if name not in self.signals:
            raise KeyError(f"訊號未登錄於權重表：name={name!r}（表：{self.path}）")
        signal = self.signals[name]
        if not hard:
            return signal.weight_soft.value
        if signal.weight_hard is None:
            raise ValueError(
                f"訊號 {name!r} 宣告 hard_capable=false，卻收到 hard=True 的結果："
                f"表中沒有 weight_hard 可用（表：{self.path}）"
            )
        return signal.weight_hard.value

    def group_of(self, name: str) -> str:
        """訊號所屬的群組。未登錄的訊號拋 `KeyError`。"""
        if name not in self.signals:
            raise KeyError(f"訊號未登錄於權重表：name={name!r}（表：{self.path}）")
        return self.signals[name].group

    def is_strong(self, name: str) -> bool:
        """訊號是否足以單獨支撐判定。未登錄的訊號拋 `KeyError`。"""
        if name not in self.signals:
            raise KeyError(f"訊號未登錄於權重表：name={name!r}（表：{self.path}）")
        return self.signals[name].strong

    def threshold(self, key: str) -> float:
        """取得門檻值。未登錄的 key 拋 `KeyError`，不使用 `dict.get()` 的預設值。"""
        if key not in self.thresholds:
            raise KeyError(f"門檻未登錄於權重表：key={key!r}（表：{self.path}）")
        return self.thresholds[key].value

    def with_overrides(
        self,
        *,
        weights: Mapping[tuple[str, bool], float] = MappingProxyType({}),
        thresholds: Mapping[str, float] = MappingProxyType({}),
    ) -> "WeightTable":
        """衍生一份改過值的表，原實例不變 —— 供 `add-ablation` 掃描。

        `weights` 以 `(訊號名稱, hard)` 為鍵。未登錄的訊號名稱或門檻 key 拋
        `KeyError`；對 `hard_capable = false` 的訊號覆寫 `hard=True` 拋 `ValueError`。

        衍生實例**不**重新套用 `PLACEHOLDER_WEIGHTS` 的封閉集合：那條限制是
        對**檔案**的，防的是有人把一個編出來的數字寫進版控並當成實測結果引用。
        掃描是在記憶體裡跑一遍就丟掉的算術，限制它等於讓掃描沒有東西可掃。
        """
        signals = dict(self.signals)
        for (name, hard), value in weights.items():
            if name not in signals:
                raise KeyError(f"訊號未登錄於權重表：name={name!r}（表：{self.path}）")
            signal = signals[name]
            if hard:
                if signal.weight_hard is None:
                    raise ValueError(
                        f"訊號 {name!r} 宣告 hard_capable=false，無 weight_hard 可覆寫"
                    )
                signals[name] = replace(
                    signal, weight_hard=replace(signal.weight_hard, value=value)
                )
            else:
                signals[name] = replace(
                    signal, weight_soft=replace(signal.weight_soft, value=value)
                )
        overridden = dict(self.thresholds)
        for key, value in thresholds.items():
            if key not in overridden:
                raise KeyError(f"門檻未登錄於權重表：key={key!r}（表：{self.path}）")
            overridden[key] = replace(overridden[key], value=value)
        return WeightTable(
            path=self.path,
            signals=signals,
            roles=dict(self.roles),
            thresholds=overridden,
            type_priority=self.type_priority,
        )

    def validate_against(self, registry: CheckRegistry) -> None:
        """以 registry 驗證表的完整性，MUST 於組裝階段呼叫。

        registry 中存在而表中不存在的訊號名稱拋 `ValueError`，並**一次列出全部**
        缺漏 —— 只指出第一個會讓補表的人來回三次。

        反方向（表有而 registry 沒有）**不拋例外**：消融實驗會停用檢查，
        而停用是正常操作。
        """
        missing = sorted(
            check.name for check in registry.enabled() if check.name not in self.signals
        )
        if missing:
            raise ValueError(
                f"下列已註冊的檢查未登錄於權重表：{'、'.join(missing)}（表：{self.path}）"
            )


def _require(raw: Mapping[str, object], key: str, where: str, path: Path) -> object:
    """取出必填欄位。缺少即拋例外並指出欄位名與它所屬的條目。"""
    if key not in raw:
        raise ValueError(f"{where} 缺少必要欄位 {key!r}（表：{path}）")
    return raw[key]


def _require_text(raw: Mapping[str, object], key: str, where: str, path: Path) -> str:
    value = _require(raw, key, where, path)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} 的 {key} 必須為非空字串，實際為 {value!r}（表：{path}）")
    return value


def _require_number(raw: Mapping[str, object], key: str, where: str, path: Path) -> float:
    value = _require(raw, key, where, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} 的 {key} 必須為數值，實際為 {value!r}（表：{path}）")
    return float(value)


def _check_basis(basis: str, where: str, path: Path) -> None:
    if basis not in BASIS_VALUES:
        raise ValueError(
            f"{where} 的 basis 必須為 {' 或 '.join(repr(v) for v in BASIS_VALUES)}，"
            f"實際為 {basis!r}（表：{path}）"
        )


def _measured_fields(raw: Mapping[str, object], where: str, path: Path) -> tuple[float, float, str]:
    """讀出 `measured` 條目的三個必要欄位，並驗證兩個機率落在開區間 (0, 1)。"""
    probabilities = []
    for key in PROBABILITY_FIELDS:
        value = _require_number(raw, key, where, path)
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{where} 的 {key} 必須落在開區間 (0, 1)，實際為 {value}（表：{path}）"
            )
        probabilities.append(value)
    measured_on = _require_text(raw, "measured_on", where, path)
    return probabilities[0], probabilities[1], measured_on


def _check_no_probabilities(raw: Mapping[str, object], where: str, path: Path) -> None:
    present = [key for key in PROBABILITY_FIELDS if key in raw]
    if present:
        raise ValueError(
            f"{where} 的 basis 為 {BASIS_PLACEHOLDER!r}，MUST NOT 攜帶條件機率，"
            f"卻含 {'、'.join(present)} —— 佔位值沒有實測機率可附（表：{path}）"
        )


def _parse_weight(raw: Mapping[str, object], where: str, path: Path) -> SignalWeight:
    """解析一個權重子表，兩種 `basis` 各自要求一組完整且互斥的欄位。"""
    value = _require_number(raw, "value", where, path)
    basis = _require_text(raw, "basis", where, path)
    _check_basis(basis, where, path)
    if basis == BASIS_MEASURED:
        p_scam, p_ham, measured_on = _measured_fields(raw, where, path)
        expected = math.log(p_scam / p_ham)
        if abs(value - expected) > MEASURED_TOLERANCE:
            raise ValueError(
                f"{where} 的 value 與兩個條件機率不一致：表中為 {value}，"
                f"ln({p_scam} / {p_ham}) 為 {expected:.4f}，"
                f"差距超過容差 {MEASURED_TOLERANCE}（表：{path}）"
            )
        return SignalWeight(
            value=value,
            basis=basis,
            p_hit_given_scam=p_scam,
            p_hit_given_ham=p_ham,
            measured_on=measured_on,
        )
    _check_no_probabilities(raw, where, path)
    if value not in PLACEHOLDER_WEIGHTS:
        raise ValueError(
            f"{where} 的 basis 為 {BASIS_PLACEHOLDER!r}，value 必須屬於允許集合 "
            f"{sorted(PLACEHOLDER_WEIGHTS)}，實際為 {value} —— "
            f"新數值必須標為 {BASIS_MEASURED!r} 並附兩個條件機率（表：{path}）"
        )
    return SignalWeight(
        value=value,
        basis=basis,
        inherited_from=_require_text(raw, "inherited_from", where, path),
        blocked_on=_require_text(raw, "blocked_on", where, path),
    )


def _parse_signal(raw: Mapping[str, object], path: Path) -> Signal:
    name = _require_text(raw, "name", f"[[signal]]（表：{path}）", path)
    where = f"訊號 {name!r}"
    group = _require_text(raw, "group", where, path)
    hard_capable = _require(raw, "hard_capable", where, path)
    if not isinstance(hard_capable, bool):
        raise ValueError(
            f"{where} 的 hard_capable 必須為布林值，實際為 {hard_capable!r}（表：{path}）"
        )
    strong = raw["strong"] if "strong" in raw else False
    if not isinstance(strong, bool):
        raise ValueError(f"{where} 的 strong 必須為布林值，實際為 {strong!r}（表：{path}）")
    if hard_capable and strong:
        raise ValueError(f"{where} 不得同時宣告 hard_capable=true 與 strong=true（表：{path}）")
    if "weight_soft" not in raw:
        raise ValueError(f"{where} 缺少 weight_soft 子表（表：{path}）")
    weight_soft = _parse_weight(raw["weight_soft"], f"{where} 的 weight_soft", path)
    if hard_capable and "weight_hard" not in raw:
        raise ValueError(
            f"{where} 宣告 hard_capable=true，MUST 同時提供 weight_hard 與 weight_soft，"
            f"缺少 weight_hard（表：{path}）"
        )
    if not hard_capable and "weight_hard" in raw:
        raise ValueError(
            f"{where} 宣告 hard_capable=false，MUST NOT 提供 weight_hard（表：{path}）"
        )
    weight_hard = None
    if hard_capable:
        weight_hard = _parse_weight(raw["weight_hard"], f"{where} 的 weight_hard", path)
    return Signal(
        name=name,
        group=group,
        hard_capable=hard_capable,
        weight_soft=weight_soft,
        weight_hard=weight_hard,
        strong=strong,
    )


def _parse_signals(document: Mapping[str, object], path: Path) -> dict[str, Signal]:
    if "signal" not in document:
        raise ValueError(f"權重表缺少 [[signal]] 區段（表：{path}）")
    signals: dict[str, Signal] = {}
    for raw in document["signal"]:
        signal = _parse_signal(raw, path)
        if signal.name in signals:
            raise ValueError(f"訊號名稱重複：{signal.name!r}（表：{path}）")
        signals[signal.name] = signal
    return signals


def _check_group_rationale(
    signals: Mapping[str, Signal], document: Mapping[str, object], path: Path
) -> None:
    """非單元群組 MUST 附 `group_rationale`：合併是例外，例外要說出理由。

    預設每個訊號自成一組。預設合併會讓人為了看起來整齊把不相干的訊號塞進同一群，
    而每一次錯誤的合併都是在丟棄證據 —— 取 max 使群組內非最大者的貢獻永遠是零，
    且沒有任何機制會報告它。
    """
    rationale = document["group_rationale"] if "group_rationale" in document else {}
    members: dict[str, list[str]] = {}
    for signal in signals.values():
        members.setdefault(signal.group, []).append(signal.name)
    for group, names in sorted(members.items()):
        count = len(names)
        if count > 1 and (group not in rationale or not rationale[group]):
            raise ValueError(
                f"群組 {group!r} 含 {count} 個訊號，MUST 於 [group_rationale] 說明"
                f"哪一個單一事實會使群組內的訊號同時命中（表：{path}）"
            )


def _parse_roles(
    document: Mapping[str, object], signals: Mapping[str, Signal], path: Path
) -> dict[str, str]:
    if "roles" not in document:
        raise ValueError(f"權重表缺少 [roles] 區段（表：{path}）")
    roles: dict[str, str] = {}
    for role, name in document["roles"].items():
        if name not in signals:
            raise ValueError(f"[roles] 的 {role!r} 指向表中不存在的訊號名稱 {name!r}（表：{path}）")
        roles[role] = name
    for required in ("quotation", "relationship", "blocklist_exact"):
        if required not in roles:
            raise ValueError(f"[roles] 缺少必要角色 {required!r}（表：{path}）")
    return roles


def _parse_thresholds(document: Mapping[str, object], path: Path) -> dict[str, Threshold]:
    if "thresholds" not in document:
        raise ValueError(f"權重表缺少 [thresholds] 區段（表：{path}）")
    thresholds: dict[str, Threshold] = {}
    for key, raw in document["thresholds"].items():
        where = f"門檻 {key!r}"
        value = _require_number(raw, "value", where, path)
        basis = _require_text(raw, "basis", where, path)
        _check_basis(basis, where, path)
        if basis == BASIS_MEASURED:
            p_scam, p_ham, measured_on = _measured_fields(raw, where, path)
            thresholds[key] = Threshold(
                value=value,
                basis=basis,
                p_hit_given_scam=p_scam,
                p_hit_given_ham=p_ham,
                measured_on=measured_on,
            )
            continue
        _check_no_probabilities(raw, where, path)
        thresholds[key] = Threshold(
            value=value,
            basis=basis,
            rationale=_require_text(raw, "rationale", where, path),
            blocked_on=_require_text(raw, "blocked_on", where, path),
        )
    for key in POSITIVE_THRESHOLDS:
        if key in thresholds and thresholds[key].value <= 0.0:
            raise ValueError(
                f"門檻 {key!r} 必須為正值，實際為 {thresholds[key].value}（表：{path}）"
            )
    return thresholds


def _parse_type_priority(document: Mapping[str, object], path: Path) -> tuple[ScamType, ...]:
    """讀出類型優先序，並驗證它確實是件數降序而非任何假裝有依據的順序。"""
    if "type_priority" not in document:
        raise ValueError(f"權重表缺少 [[type_priority]] 區段（表：{path}）")
    ordered: list[ScamType] = []
    previous_cases = None
    for raw in document["type_priority"]:
        name = _require_text(raw, "name", f"[[type_priority]]（表：{path}）", path)
        if name not in ScamType.__members__:
            raise ValueError(
                f"[[type_priority]] 含不存在的 ScamType 成員名稱：{name!r}（表：{path}）"
            )
        where = f"[[type_priority]] 的 {name}"
        _require_text(raw, "source", where, path)
        cases = _require_number(raw, "cases", where, path)
        if previous_cases is not None and cases > previous_cases:
            raise ValueError(
                f"{where} 的 cases={cases:.0f} 大於前一筆的 {previous_cases:.0f}，"
                f"排序不是件數降序（表：{path}）"
            )
        previous_cases = cases
        ordered.append(ScamType[name])
    listed = [scam_type.name for scam_type in ordered]
    duplicated = sorted({name for name in listed if listed.count(name) > 1})
    if duplicated:
        raise ValueError(f"[[type_priority]] 含重複成員：{'、'.join(duplicated)}（表：{path}）")
    missing = sorted(set(ScamType.__members__) - set(listed))
    if missing:
        raise ValueError(
            f"[[type_priority]] 未涵蓋全部 ScamType 成員，缺少：{'、'.join(missing)}（表：{path}）"
        )
    return tuple(ordered)


def load_weights(path: Path = DEFAULT_WEIGHTS_PATH) -> WeightTable:
    """讀取並驗證權重表。

    全部驗證失敗都拋例外並指出檔案路徑與失敗的欄位，**不回退為任何內建預設表**。
    檔案不存在時 `Path.read_bytes()` 的 `FileNotFoundError` 已含路徑，直接傳播。
    """
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    signals = _parse_signals(document, path)
    _check_group_rationale(signals, document, path)
    return WeightTable(
        path=path,
        signals=signals,
        roles=_parse_roles(document, signals, path),
        thresholds=_parse_thresholds(document, path),
        type_priority=_parse_type_priority(document, path),
    )
