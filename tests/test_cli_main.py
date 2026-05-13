"""The CLI is tested through main(), the same door the console script uses."""

from __future__ import annotations

import pytest

from echoscribe import __version__
from echoscribe.cli import main


def test_no_command_prints_help_and_exits_with_usage(capsys) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "usage: echoscribe" in captured.err


def test_the_version_flag_prints_the_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_an_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["nonsense"])
    assert exit_info.value.code == 2
