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


@pytest.mark.slow
def test_the_streaming_session_runs_offline(tmp_path) -> None:
    result = run_example("streaming_session", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "session.jsonl").exists()
    assert "identical event trace" in result.stdout


@pytest.mark.slow
def test_the_streaming_session_endpoints_after_the_units(tmp_path) -> None:
    from echoscribe.stream.session import read_session

    result = run_example("streaming_session", tmp_path)
    assert result.returncode == 0, result.stderr
    session = read_session(tmp_path / "session.jsonl")
    assert len(session.utterances) == 1
    record = session.utterances[0]
    assert record.hypothesis == (3, 1, 4, 1, 5)
    assert record.endpointed
    # the last planted unit sits at frame 25 + 48; the endpoint must be later
    assert record.endpoint_frames is not None and record.endpoint_frames > 73


@pytest.mark.slow
def test_the_code_switch_report_runs_offline(tmp_path) -> None:
    result = run_example("codeswitch_report", tmp_path)
    assert result.returncode == 0, result.stderr
    document = (tmp_path / "codeswitch_report.svg").read_text(encoding="utf-8")
    assert document.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "hybrid scores" in document
    assert "hand-written" in result.stdout


@pytest.mark.slow
def test_the_code_switch_scores_match_the_planted_errors(tmp_path) -> None:
    result = run_example("codeswitch_report", tmp_path)
    assert result.returncode == 0, result.stderr
    # u1: one word substituted; u2: one character deleted; u3: one word deleted;
    # u4: exact -- 3 errors over 6 + 6 + 4 + 4 = 20 mixed reference tokens
    assert "corpus totals: 3 errors over 20 mixed tokens" in result.stdout
