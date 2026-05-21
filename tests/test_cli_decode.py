"""The decode command is run end to end over a small prepared corpus."""

from __future__ import annotations

import pytest

from echoscribe.cli import main
from echoscribe.cli.common import read_jsonl


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


def test_decode_writes_one_line_per_utterance(tmp_path) -> None:
    prepare(tmp_path)
    code = main(["decode", "--workdir", str(tmp_path)])
    assert code == 0
    rows = read_jsonl(tmp_path / "decode.jsonl")
    assert len(rows) == 2
    assert all(row["method"] == "greedy" for row in rows)
    assert all(row["exact_match"] is True for row in rows)


def test_the_summary_counts_exact_replays(tmp_path, capsys) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "2 replayed their planted transcript exactly" in out


def test_the_output_says_the_posteriors_are_fixtures(tmp_path, capsys) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    assert "not the output of a trained model" in capsys.readouterr().out


def test_rows_carry_both_units_and_ids(tmp_path) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    row = read_jsonl(tmp_path / "decode.jsonl")[0]
    assert len(row["hypothesis_units"]) == len(row["hypothesis_ids"])
    assert row["hypothesis_units"] == row["reference_units"]


def test_decode_on_an_unprepared_directory_fails(tmp_path, capsys) -> None:
    assert main(["decode", "--workdir", str(tmp_path)]) == 1
    assert "echoscribe synth" in capsys.readouterr().err


def test_an_unknown_method_is_a_usage_error(tmp_path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["decode", "--workdir", str(tmp_path), "--method", "magic"])
    assert exit_info.value.code == 2


def test_the_prefix_beam_replays_the_same_units_as_greedy(tmp_path) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    main(
        [
            "decode",
            "--workdir",
            str(tmp_path),
            "--method",
            "prefix",
            "--beam",
            "4",
            "--out",
            "prefix.jsonl",
        ]
    )
    greedy = {
        row["utterance_id"]: row["hypothesis_ids"] for row in read_jsonl(tmp_path / "decode.jsonl")
    }
    prefix = {
        row["utterance_id"]: row["hypothesis_ids"] for row in read_jsonl(tmp_path / "prefix.jsonl")
    }
    assert greedy == prefix


def test_the_output_name_is_caller_chosen(tmp_path) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path), "--out", "hyps.jsonl"])
    assert (tmp_path / "hyps.jsonl").exists()
    assert not (tmp_path / "decode.jsonl").exists()
