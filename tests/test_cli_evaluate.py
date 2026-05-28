"""The eval command is scored by hand before it is trusted with a number."""

from __future__ import annotations

import pytest

from echoscribe.cli import main
from echoscribe.cli.common import read_json


def prepare(tmp_path, utterances: int = 2):
    """A work directory with synth and feats already run over it."""
    assert (
        main(
            [
                "synth",
                "--workdir",
                str(tmp_path),
                "--utterances",
                str(utterances),
                "--max-tokens",
                "2",
                "--seed",
                "5",
            ]
        )
        == 0
    )
    assert main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "40"]) == 0
    return tmp_path


def decode_first(tmp_path) -> None:
    prepare(tmp_path)
    assert main(["decode", "--workdir", str(tmp_path)]) == 0


def test_units_level_scores_the_scripted_decode_as_exact(tmp_path, capsys) -> None:
    decode_first(tmp_path)
    code = main(["eval", "--workdir", str(tmp_path)])
    assert code == 0
    payload = read_json(tmp_path / "scores.json")
    assert payload["level"] == "units"
    assert payload["totals"]["errors"] == 0
    assert payload["totals"]["error_rate"] == 0.0
    assert len(payload["utterances"]) == 2
    assert "UER 0.00%" in capsys.readouterr().out


def test_text_level_wer_is_zero_on_an_exact_decode(tmp_path) -> None:
    decode_first(tmp_path)
    main(["eval", "--workdir", str(tmp_path), "--level", "text", "--metric", "wer"])
    payload = read_json(tmp_path / "scores.json")
    assert payload["metric"] == "wer"
    assert payload["totals"]["errors"] == 0


def test_the_hybrid_metric_runs_over_mixed_text(tmp_path) -> None:
    decode_first(tmp_path)
    main(["eval", "--workdir", str(tmp_path), "--level", "text", "--metric", "hybrid"])
    payload = read_json(tmp_path / "scores.json")
    assert payload["metric"] == "hybrid"
    assert payload["totals"]["errors"] == 0


def test_the_output_names_the_ground_truth(tmp_path, capsys) -> None:
    decode_first(tmp_path)
    main(["eval", "--workdir", str(tmp_path)])
    assert "planted transcripts" in capsys.readouterr().out


def test_eval_without_a_decode_file_fails(tmp_path, capsys) -> None:
    prepare(tmp_path)
    assert main(["eval", "--workdir", str(tmp_path)]) == 1
    assert "run `echoscribe decode` first" in capsys.readouterr().err


def test_an_unknown_metric_is_a_usage_error(tmp_path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["eval", "--workdir", str(tmp_path), "--metric", "magic"])
    assert exit_info.value.code == 2
