"""The package root must stay importable with or without the extra."""

from __future__ import annotations

import importlib.util

import pytest

from echoscribe.errors import ModelError
from echoscribe.model import TORCH_AVAILABLE, require_torch


def test_the_flag_matches_whether_torch_is_installed() -> None:
    assert TORCH_AVAILABLE is (importlib.util.find_spec("torch") is not None)


def test_the_package_root_imports_without_torch_being_loaded() -> None:
    import sys

    # Whatever the flag says, importing echoscribe.model itself must not have
    # dragged torch into the process just to answer the question.
    assert "echoscribe.model" in sys.modules
    if not TORCH_AVAILABLE:
        assert "torch" not in sys.modules


def test_require_torch_agrees_with_the_flag() -> None:
    if TORCH_AVAILABLE:
        require_torch()  # must not raise
    else:
        with pytest.raises(ModelError, match="torch extra"):
            require_torch()
