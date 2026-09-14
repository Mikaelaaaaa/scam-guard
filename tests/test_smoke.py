"""確認套件骨架可被 import。"""

import scam_guard


def test_package_importable() -> None:
    assert scam_guard.__doc__
