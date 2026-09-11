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


def test_the_declared_console_script_points_at_main() -> None:
    from importlib import metadata

    scripts = [
        entry
        for entry in metadata.entry_points(group="console_scripts")
        if entry.name == "echoscribe"
    ]
    assert len(scripts) == 1
    assert scripts[0].value == "echoscribe.cli:main"
    assert scripts[0].load() is main


def test_the_shared_json_helpers_round_trip(tmp_path) -> None:
    from echoscribe.cli.common import read_json, write_json

    target = write_json(tmp_path / "doc.json", {"b": 1, "a": [1, 2]})
    assert target.read_text() == '{\n  "a": [\n    1,\n    2\n  ],\n  "b": 1\n}\n'
    assert read_json(target) == {"a": [1, 2], "b": 1}


def test_the_jsonl_helpers_round_trip(tmp_path) -> None:
    from echoscribe.cli.common import read_jsonl, write_jsonl

    target = write_jsonl(tmp_path / "rows.jsonl", [{"b": 1}, {"a": [1, 2]}])
    assert target.read_text() == '{"b": 1}\n{"a": [1, 2]}\n'
    assert read_jsonl(target) == [{"b": 1}, {"a": [1, 2]}]


def test_reading_jsonl_skips_blank_lines(tmp_path) -> None:
    from echoscribe.cli.common import read_jsonl

    target = tmp_path / "rows.jsonl"
    target.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl(target) == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize("command", ["decode", "stream", "eval", "bench", "report", "denoise"])
def test_consuming_commands_fail_cleanly_on_an_empty_workdir(tmp_path, capsys, command) -> None:
    assert main([command, "--workdir", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("echoscribe: error:")
    assert "Traceback" not in error
