"""The demo is the whole pipeline in one call, so its test is integration."""

from __future__ import annotations

import pytest

from echoscribe.cli import main

STAGE_FILES = (
    "manifest.jsonl",
    "features.npz",
    "decode.jsonl",
    "session.jsonl",
    "scores-units.json",
    "scores-text.json",
    "bench.json",
    "report.svg",
)


@pytest.mark.slow
def test_the_demo_runs_the_whole_loop(tmp_path, capsys) -> None:
    code = main(["demo", "--workdir", str(tmp_path), "--utterances", "3", "--repeats", "1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "demo summary" in out
    for name in STAGE_FILES:
        assert (tmp_path / name).exists(), name


@pytest.mark.slow
def test_the_demo_prints_the_numbers_it_measured(tmp_path, capsys) -> None:
    main(["demo", "--workdir", str(tmp_path), "--utterances", "3", "--repeats", "1"])
    out = capsys.readouterr().out
    # scripted fixtures decode exactly, so the honest summary reads zero
    assert "unit error rate:   0.0000" in out
    assert "decode RTF:" in out
    assert len(out.split("report digest:")[1].split()[0]) == 64
