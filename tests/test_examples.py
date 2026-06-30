"""Every example is a program; these tests actually run them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run_example(name: str, workdir: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run one example's ``run.py`` as a subprocess, as a user would."""
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name / "run.py"), "--workdir", str(workdir), *extra],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.slow
def test_the_corpus_walkthrough_runs_offline(tmp_path) -> None:
    result = run_example("corpus_and_features", tmp_path, "--utterances", "4")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "manifest.jsonl").exists()
    assert len(list((tmp_path / "wav").glob("*.wav"))) == 4
    assert "planted ground truth" in result.stdout
    assert "reproduced the same manifest digest" in result.stdout


@pytest.mark.slow
def test_the_decoding_comparison_runs_offline(tmp_path) -> None:
    result = run_example("decoding_comparison", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "comparison.json").exists()
    assert "replayed the planted sequence exactly" in result.stdout
    assert "scripted fixture" in result.stdout
