"""權重表的載入、驗證與查詢。

本檔同時是「表與程式碼一致」的綁定處：`[roles]` 的三個角色以**測試**綁定到
`pipeline.QUOTATION_CHECK` 與 `RELATIONSHIP_RULE`，而不是以 import 綁定 ——
import 的代價是一條永久的反向依賴，測試的代價只有一行。
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scam_guard.check import CheckRegistry, Stage
from scam_guard.pipeline import QUOTATION_CHECK
from scam_guard.rules.speech_act import RELATIONSHIP_RULE
from scam_guard.types import ScamType
from scam_guard.weights import (
    DEFAULT_WEIGHTS_PATH,
    PLACEHOLDER_WEIGHTS,
    WeightTable,
    load_weights,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PRIORITY = (
    "FAKE_INVESTMENT",
    "ROMANCE_INVESTMENT",
    "SEXUAL_SERVICE",
    "FAKE_BUYER",
    "ROMANCE_MARRIAGE",
    "PHISHING_LINK",
    "FAKE_LOAN",
    "FAKE_AUTHORITY",
    "BANK_ACCOUNT_HARVEST",
    "GAME_ITEM",
    "FAKE_PRIZE",
    "FAKE_JOB",
    "ACCOUNT_TAKEOVER",
    "INSTALLMENT_CANCEL",
    "GUESS_WHO",
    "ORDER_ANOMALY",
    "FAKE_CHARITY",
    "FAKE_PARCEL",
)
"""165 各 `CaseTitle` 件數降序，兩個合併成員取其涵蓋的兩個 `CaseTitle` 之和。

`INSTALLMENT_CANCEL`（2586）排在 `GUESS_WHO`（2227）之前，而 `ScamType` 的定義
順序恰好相反 —— 這一點是「排序不是 Enum 順序」的可觀察證據。
"""

TYPE_PRIORITY_BLOCK = "".join(
    f'[[type_priority]]\nname = "{name}"\ncases = {1000 - index}\nsource = "測試用"\n\n'
    for index, name in enumerate(EXPECTED_PRIORITY)
)

ROLES_BLOCK = """
[roles]
quotation = "quotation"
relationship = "relationship_building"
blocklist_exact = "url_blocklist"

[thresholds]

"""

SOFT_ONLY = """
[[signal]]
name = "{name}"
group = "{group}"
hard_capable = false
[signal.weight_soft]
value = 0.6
basis = "placeholder"
inherited_from = "測試"
blocked_on = "add-testset"

"""

HARD_CAPABLE = """
[[signal]]
name = "{name}"
group = "{group}"
hard_capable = true
[signal.weight_hard]
value = 2.5
basis = "placeholder"
inherited_from = "測試"
blocked_on = "add-testset"
[signal.weight_soft]
value = 0.6
basis = "placeholder"
inherited_from = "測試"
blocked_on = "add-testset"

"""

MINIMAL_SIGNALS = (
    SOFT_ONLY.format(name="quotation", group="quotation")
    + SOFT_ONLY.format(name="relationship_building", group="relationship_building")
    + HARD_CAPABLE.format(name="url_blocklist", group="url_reputation")
)


def write_table(tmp_path: Path, signals: str, roles: str = ROLES_BLOCK) -> Path:
    """把一份構造的表寫進暫存目錄，供載入驗證的反例使用。"""
    path = tmp_path / "weights.toml"
    path.write_text(roles + signals + TYPE_PRIORITY_BLOCK, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def table() -> WeightTable:
    return load_weights()


# --- 實際的表 -----------------------------------------------------------


def test_real_table_loads(table: WeightTable) -> None:
    assert table.path == DEFAULT_WEIGHTS_PATH


def test_thirty_three_signals_with_unique_names(table: WeightTable) -> None:
    """33 個訊號：規則層 21、引述 1、規避 5、URL 層 5、網域年齡 1。"""
    assert len(table.signals) == 33
    assert all(name == signal.name for name, signal in table.signals.items())


def test_every_signal_declares_a_group(table: WeightTable) -> None:
    assert all(signal.group for signal in table.signals.values())
    assert sum(len(names) for names in table.groups.values()) == 33


def test_non_unit_groups_have_a_rationale() -> None:
    """非單元群組 MUST 有 `group_rationale` —— 由載入驗證，此處確認它真的在表裡。"""
    document = tomllib.loads(DEFAULT_WEIGHTS_PATH.read_text(encoding="utf-8"))
    table = load_weights()
    multi = {group for group, names in table.groups.items() if len(names) > 1}

    assert multi == {
        "authority_script",
        "credential_solicit",
        "advance_fee",
        "too_good_offer",
        "url_reputation",
        "evasion",
    }
    assert all(document["group_rationale"][group] for group in multi)


def test_url_shortener_is_its_own_group(table: WeightTable) -> None:
    """短網址宣告的是「目的地未知」，併入 URL 信譽群組會被取 max 蓋掉。"""
    assert table.groups["url_shortener"] == ("url_shortener",)
    assert table.weight_for("url_shortener", hard=False) == 0.0


def test_every_placeholder_value_is_in_the_allowed_set(table: WeightTable) -> None:
    values = set()
    for signal in table.signals.values():
        values.add(signal.weight_soft.value)
        if signal.weight_hard is not None:
            values.add(signal.weight_hard.value)

    assert values <= PLACEHOLDER_WEIGHTS


def test_allowed_set_is_exactly_four_traceable_values() -> None:
    """四個值各有出處，`add-testset` 之前不得出現第五個。

    - `2.5`  `add-speech-act-rules` 的 Tier-A 佔位值
    - `0.6`  `add-speech-act-rules` 的 Tier-B 佔位值（亦為自帶碼豁免的降級值）
    - `0.0`  `add-url-check` 的 `url_shortener`：宣告資訊不足
    - `-1.5` `add-quotation-check` 的引述負權重
    """
    assert PLACEHOLDER_WEIGHTS == frozenset({2.5, 0.6, 0.0, -1.5})


def test_no_measured_entry_exists_yet(table: WeightTable) -> None:
    """目前一個實測值都沒有。`add-testset` 之後這條測試會被改寫，而改寫本身就是進度。"""
    bases = {signal.weight_soft.basis for signal in table.signals.values()}
    bases |= {
        signal.weight_hard.basis
        for signal in table.signals.values()
        if signal.weight_hard is not None
    }

    assert bases == {"placeholder"}


def test_thresholds_section_exists(table: WeightTable) -> None:
    """本 change 只建立區段與格式；條目由同 PR 的三個 change 填入。"""
    assert isinstance(table.thresholds, type(table.roles))


# --- 查表 ---------------------------------------------------------------


def test_weight_is_keyed_by_name_and_hard(table: WeightTable) -> None:
    assert table.weight_for("solicit_otp", hard=True) == 2.5
    assert table.weight_for("solicit_otp", hard=False) == 0.6


def test_blocklist_two_level_match(table: WeightTable) -> None:
    assert table.weight_for("url_blocklist", hard=True) == 2.5
    assert table.weight_for("url_blocklist", hard=False) == 0.6


def test_hard_on_a_soft_only_signal_raises(table: WeightTable) -> None:
    with pytest.raises(ValueError, match="hard_capable=false"):
        table.weight_for("quotation", hard=True)


def test_unknown_signal_raises_key_error(table: WeightTable) -> None:
    with pytest.raises(KeyError, match="不存在的訊號"):
        table.weight_for("不存在的訊號", hard=False)


def test_group_of_unknown_signal_raises(table: WeightTable) -> None:
    with pytest.raises(KeyError, match="不存在的訊號"):
        table.group_of("不存在的訊號")


def test_threshold_of_unknown_key_raises(table: WeightTable) -> None:
    with pytest.raises(KeyError, match="decision_score"):
        table.threshold("decision_score")


def test_with_overrides_does_not_mutate_the_original(table: WeightTable) -> None:
    derived = table.with_overrides(weights={("solicit_otp", True): 9.9})

    assert derived.weight_for("solicit_otp", hard=True) == 9.9
    assert table.weight_for("solicit_otp", hard=True) == 2.5


def test_with_overrides_rejects_unknown_names(table: WeightTable) -> None:
    with pytest.raises(KeyError, match="不存在的訊號"):
        table.with_overrides(weights={("不存在的訊號", False): 1.0})


# --- 角色 ---------------------------------------------------------------


def test_quotation_role_matches_the_pipeline_constant(table: WeightTable) -> None:
    assert table.roles["quotation"] == QUOTATION_CHECK


def test_relationship_role_matches_the_rule_constant(table: WeightTable) -> None:
    assert table.roles["relationship"] == RELATIONSHIP_RULE


def test_blocklist_exact_role_is_a_registered_signal(table: WeightTable) -> None:
    assert table.roles["blocklist_exact"] in table.signals


# --- 類型優先序 ---------------------------------------------------------


def test_type_priority_covers_every_member(table: WeightTable) -> None:
    assert set(table.type_priority) == set(ScamType)


def test_type_priority_is_case_count_descending(table: WeightTable) -> None:
    assert [scam_type.name for scam_type in table.type_priority] == list(EXPECTED_PRIORITY)


def test_type_priority_is_not_the_enum_order(table: WeightTable) -> None:
    """件數順序與 Enum 定義順序不同 —— 否則「有依據」與「照打字順序」看不出差別。"""
    assert list(table.type_priority) != list(ScamType)


# --- 一致性檢查 ---------------------------------------------------------


class FakeCheck:
    """只提供 `name` 與 `stage` 的假檢查，供 registry 一致性檢查使用。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.stage = Stage.LOCAL

    def __call__(self, req: object, doc: object) -> list:
        return []


def test_registered_but_unlisted_check_is_reported(table: WeightTable) -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("未登錄的檢查"))

    with pytest.raises(ValueError, match="未登錄的檢查"):
        table.validate_against(registry)


def test_all_missing_names_are_listed_at_once(table: WeightTable) -> None:
    registry = CheckRegistry()
    for name in ("缺一", "缺二", "缺三"):
        registry.register(FakeCheck(name))

    with pytest.raises(ValueError) as raised:
        table.validate_against(registry)

    assert all(name in str(raised.value) for name in ("缺一", "缺二", "缺三"))


def test_disabled_check_does_not_trigger_the_error(table: WeightTable) -> None:
    registry = CheckRegistry()
    registry.register(FakeCheck("solicit_otp"))
    registry.register(FakeCheck("未登錄的檢查"))
    registry.disable("未登錄的檢查")

    table.validate_against(registry)


def test_table_entry_without_a_registered_check_is_fine(table: WeightTable) -> None:
    table.validate_against(CheckRegistry())


# --- 載入時驗證：每一條都拋例外，沒有任何預設值回退 -----------------------


def test_duplicate_name(tmp_path: Path) -> None:
    signals = MINIMAL_SIGNALS + SOFT_ONLY.format(name="quotation", group="別的群組")

    with pytest.raises(ValueError, match="訊號名稱重複"):
        load_weights(write_table(tmp_path, signals))


def test_missing_group(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace('group = "quotation"\n', "")

    with pytest.raises(ValueError, match="缺少必要欄位 'group'"):
        load_weights(write_table(tmp_path, broken))


def test_missing_hard_capable(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace("hard_capable = false\n", "", 1)

    with pytest.raises(ValueError, match="缺少必要欄位 'hard_capable'"):
        load_weights(write_table(tmp_path, broken))


def test_hard_capable_without_weight_hard(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace("hard_capable = false\n", "hard_capable = true\n", 1)

    with pytest.raises(ValueError, match="缺少 weight_hard"):
        load_weights(write_table(tmp_path, broken))


def test_soft_only_signal_with_weight_hard(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace(
        HARD_CAPABLE.format(name="url_blocklist", group="url_reputation"),
        HARD_CAPABLE.format(name="url_blocklist", group="url_reputation").replace(
            "hard_capable = true", "hard_capable = false"
        ),
    )

    with pytest.raises(ValueError, match="MUST NOT 提供 weight_hard"):
        load_weights(write_table(tmp_path, broken))


def test_unknown_basis(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace('basis = "placeholder"', 'basis = "猜的"', 1)

    with pytest.raises(ValueError, match="basis 必須為"):
        load_weights(write_table(tmp_path, broken))


def test_measured_without_probabilities(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace(
        'value = 0.6\nbasis = "placeholder"\ninherited_from = "測試"\nblocked_on = "add-testset"',
        'value = 1.83\nbasis = "measured"',
        1,
    )

    with pytest.raises(ValueError, match="缺少必要欄位 'p_hit_given_scam'"):
        load_weights(write_table(tmp_path, broken))


def test_measured_probability_outside_open_interval(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace(
        'value = 0.6\nbasis = "placeholder"\ninherited_from = "測試"\nblocked_on = "add-testset"',
        'value = 1.83\nbasis = "measured"\np_hit_given_scam = 1.0\n'
        'p_hit_given_ham = 0.066\nmeasured_on = "測試"',
        1,
    )

    with pytest.raises(ValueError, match=r"開區間 \(0, 1\)"):
        load_weights(write_table(tmp_path, broken))


def test_measured_value_disagrees_with_the_probabilities(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace(
        'value = 0.6\nbasis = "placeholder"\ninherited_from = "測試"\nblocked_on = "add-testset"',
        'value = 1.90\nbasis = "measured"\np_hit_given_scam = 0.41\n'
        'p_hit_given_ham = 0.066\nmeasured_on = "測試"',
        1,
    )

    with pytest.raises(ValueError) as raised:
        load_weights(write_table(tmp_path, broken))

    assert "1.9" in str(raised.value)
    assert "1.826" in str(raised.value)


def test_measured_value_consistent_with_the_probabilities(tmp_path: Path) -> None:
    """0.41 / 0.066 的對數比為 1.826…，表中寫 1.83 落在容差內。"""
    good = MINIMAL_SIGNALS.replace(
        'value = 0.6\nbasis = "placeholder"\ninherited_from = "測試"\nblocked_on = "add-testset"',
        'value = 1.83\nbasis = "measured"\np_hit_given_scam = 0.41\n'
        'p_hit_given_ham = 0.066\nmeasured_on = "add-testset v1"',
        1,
    )

    loaded = load_weights(write_table(tmp_path, good))

    assert loaded.weight_for("quotation", hard=False) == 1.83


def test_placeholder_carrying_a_probability(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace(
        'blocked_on = "add-testset"',
        'blocked_on = "add-testset"\np_hit_given_scam = 0.41',
        1,
    )

    with pytest.raises(ValueError, match="MUST NOT 攜帶條件機率"):
        load_weights(write_table(tmp_path, broken))


def test_placeholder_without_inherited_from(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace('inherited_from = "測試"\n', "", 1)

    with pytest.raises(ValueError, match="缺少必要欄位 'inherited_from'"):
        load_weights(write_table(tmp_path, broken))


def test_placeholder_without_blocked_on(tmp_path: Path) -> None:
    broken = MINIMAL_SIGNALS.replace('blocked_on = "add-testset"\n', "", 1)

    with pytest.raises(ValueError, match="缺少必要欄位 'blocked_on'"):
        load_weights(write_table(tmp_path, broken))


def test_placeholder_value_outside_the_allowed_set(tmp_path: Path) -> None:
    """編一個看起來像調過的數字，載入就失敗 —— 這是本設計唯一的防線。"""
    broken = MINIMAL_SIGNALS.replace("value = 0.6", "value = 1.9", 1)

    with pytest.raises(ValueError) as raised:
        load_weights(write_table(tmp_path, broken))

    assert "1.9" in str(raised.value)
    assert "[-1.5, 0.0, 0.6, 2.5]" in str(raised.value)


def test_merged_group_without_rationale(tmp_path: Path) -> None:
    signals = MINIMAL_SIGNALS + SOFT_ONLY.format(name="url_brand", group="url_reputation")

    with pytest.raises(ValueError, match="url_reputation"):
        load_weights(write_table(tmp_path, signals))


def test_unit_group_needs_no_rationale(tmp_path: Path) -> None:
    load_weights(write_table(tmp_path, MINIMAL_SIGNALS))


def test_role_pointing_at_a_missing_signal(tmp_path: Path) -> None:
    roles = ROLES_BLOCK.replace('quotation = "quotation"', 'quotation = "拼錯了"')

    with pytest.raises(ValueError, match="'quotation'.*'拼錯了'"):
        load_weights(write_table(tmp_path, MINIMAL_SIGNALS, roles=roles))


def test_threshold_placeholder_without_rationale(tmp_path: Path) -> None:
    roles = ROLES_BLOCK.replace(
        "[thresholds]\n",
        "[thresholds]\n[thresholds.decision_score]\nvalue = 1.5\n"
        'basis = "placeholder"\nblocked_on = "add-metrics"\n',
    )

    with pytest.raises(ValueError, match="缺少必要欄位 'rationale'"):
        load_weights(write_table(tmp_path, MINIMAL_SIGNALS, roles=roles))


def test_threshold_placeholder_without_blocked_on(tmp_path: Path) -> None:
    roles = ROLES_BLOCK.replace(
        "[thresholds]\n",
        "[thresholds]\n[thresholds.decision_score]\nvalue = 1.5\n"
        'basis = "placeholder"\nrationale = "測試"\n',
    )

    with pytest.raises(ValueError, match="缺少必要欄位 'blocked_on'"):
        load_weights(write_table(tmp_path, MINIMAL_SIGNALS, roles=roles))


def test_type_priority_with_a_missing_member(tmp_path: Path) -> None:
    path = tmp_path / "weights.toml"
    truncated = TYPE_PRIORITY_BLOCK.split('[[type_priority]]\nname = "FAKE_PARCEL"')[0]
    path.write_text(ROLES_BLOCK + MINIMAL_SIGNALS + truncated, encoding="utf-8")

    with pytest.raises(ValueError, match="FAKE_PARCEL"):
        load_weights(path)


def test_type_priority_with_a_duplicate_member(tmp_path: Path) -> None:
    path = tmp_path / "weights.toml"
    extra = '[[type_priority]]\nname = "FAKE_PARCEL"\ncases = 1\nsource = "測試用"\n'
    path.write_text(ROLES_BLOCK + MINIMAL_SIGNALS + TYPE_PRIORITY_BLOCK + extra, encoding="utf-8")

    with pytest.raises(ValueError, match="重複成員"):
        load_weights(path)


def test_type_priority_with_an_unknown_member(tmp_path: Path) -> None:
    path = tmp_path / "weights.toml"
    block = TYPE_PRIORITY_BLOCK.replace("FAKE_PARCEL", "不存在的成員")
    path.write_text(ROLES_BLOCK + MINIMAL_SIGNALS + block, encoding="utf-8")

    with pytest.raises(ValueError, match="不存在的成員"):
        load_weights(path)


def test_type_priority_not_in_case_count_order(tmp_path: Path) -> None:
    path = tmp_path / "weights.toml"
    block = TYPE_PRIORITY_BLOCK.replace("cases = 1000\n", "cases = 1\n", 1)
    path.write_text(ROLES_BLOCK + MINIMAL_SIGNALS + block, encoding="utf-8")

    with pytest.raises(ValueError, match="不是件數降序"):
        load_weights(path)


def test_missing_file_raises_with_the_expected_path(tmp_path: Path) -> None:
    missing = tmp_path / "沒有這個檔.toml"

    with pytest.raises(FileNotFoundError) as raised:
        load_weights(missing)

    assert str(missing) in str(raised.value)


# --- 界線 ---------------------------------------------------------------


IMPORT_PROBE = """
import builtins
import pathlib


class Recorder:
    def __init__(self, real):
        self.real = real
        self.seen = []

    def __call__(self, *args, **kwargs):
        self.seen.append(str(args[0]) if args else "")
        return self.real(*args, **kwargs)


builtin_open = Recorder(builtins.open)
builtins.open = builtin_open
path_open = Recorder(pathlib.Path.open)
pathlib.Path.open = path_open
path_read = Recorder(pathlib.Path.read_bytes)
pathlib.Path.read_bytes = path_read

import scam_guard.weights

seen = builtin_open.seen + path_open.seen + path_read.seen
print([item for item in seen if "weights.toml" in item])
"""


def test_import_does_not_read_the_table() -> None:
    """import 時讀檔會讓一個沒裝好的環境在**收集測試**的階段就炸。"""
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )

    assert completed.stdout.strip() == "[]"


def test_module_does_not_import_pipeline_or_rules() -> None:
    source = (REPO_ROOT / "scam_guard" / "weights.py").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert not any("scam_guard.pipeline" in line for line in imports)
    assert not any("scam_guard.rules" in line for line in imports)


def test_table_is_not_named_data_and_is_tracked() -> None:
    """`.gitignore` 有一行 `data/`（無前導斜線、任何深度生效）。"""
    assert DEFAULT_WEIGHTS_PATH.parent.name == "tables"
    assert DEFAULT_WEIGHTS_PATH.exists()


def test_package_data_includes_the_table() -> None:
    """`pip install .`（非 `-e`）之後表必須仍在套件裡，靠的就是這一行宣告。

    不在單元測試裡真的跑一次 `pip install .`：那要建 wheel、建暫時的虛擬環境，
    成本與本檔其餘測試差好幾個數量級。此處斷言的是使它成立的那個機制。
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert "*.toml" in package_data["scam_guard.tables"]


def test_runtime_dependencies_are_still_empty() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["dependencies"] == []
