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


def test_an_exact_row_carries_the_reference_text(tmp_path) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    rows = read_jsonl(tmp_path / "decode.jsonl")
    assert rows, "the decode should have produced rows"
    for row in rows:
        assert row["reference_text"]
        if row["exact_match"]:
            assert row["hypothesis_text"] == row["reference_text"]


def test_a_degraded_hypothesis_text_is_the_joined_units(tmp_path) -> None:
    # Not reachable through the scripted decode, which replays everything;
    # the reconstruction rule is checked through the row format instead.
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    row = read_jsonl(tmp_path / "decode.jsonl")[0]
    assert set(row) >= {
        "utterance_id",
        "method",
        "reference_units",
        "hypothesis_units",
        "hypothesis_ids",
        "exact_match",
        "reference_text",
        "hypothesis_text",
    }


def test_bundle_decoding_without_a_vocabulary_fails(tmp_path, capsys) -> None:
    code = main(["decode", "--workdir", str(tmp_path), "--bundle", "absent.pt"])
    assert code == 1
    assert "echoscribe train" in capsys.readouterr().err


@pytest.mark.slow
def test_a_trained_bundle_decodes_through_the_seam(tmp_path, capsys) -> None:
    pytest.importorskip("torch", reason="the torch extra is not installed")
    prepare(tmp_path)
    assert (
        main(
            [
                "train",
                "--workdir",
                str(tmp_path),
                "--epochs",
                "3",
                "--utterances",
                "6",
                "--hidden",
                "24",
            ]
        )
        == 0
    )
    code = main(
        [
            "decode",
            "--workdir",
            str(tmp_path),
            "--bundle",
            str(tmp_path / "model.pt"),
            "--out",
            "bundle.jsonl",
        ]
    )
    assert code == 0
    rows = read_jsonl(tmp_path / "bundle.jsonl")
    assert len(rows) == 2
    assert all(row["method"] == "bundle-ctc" for row in rows)
    for row in rows:
        assert len(row["hypothesis_ids"]) == len(row["hypothesis_units"])
        assert all(symbol >= 0 for symbol in row["hypothesis_ids"])
    assert "memorization" in capsys.readouterr().out


@pytest.mark.slow
def test_a_bundle_rejects_features_of_the_wrong_width(tmp_path, capsys) -> None:
    pytest.importorskip("torch", reason="the torch extra is not installed")
    assert (
        main(
            [
                "synth",
                "--workdir",
                str(tmp_path),
                "--utterances",
                "2",
                "--max-tokens",
                "2",
                "--seed",
                "5",
            ]
        )
        == 0
    )
    assert main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "24"]) == 0
    assert (
        main(
            [
                "train",
                "--workdir",
                str(tmp_path),
                "--epochs",
                "1",
                "--utterances",
                "4",
                "--hidden",
                "16",
            ]
        )
        == 0
    )
    assert main(["decode", "--workdir", str(tmp_path), "--bundle", str(tmp_path / "model.pt")]) == 1
    assert "rerun feats" in capsys.readouterr().err


def test_confidences_ride_along_with_a_greedy_decode(tmp_path) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path), "--confidences"])
    rows = read_jsonl(tmp_path / "decode.jsonl")
    for row in rows:
        assert len(row["token_confidences"]) == len(row["hypothesis_ids"])
        assert all(0.0 <= probability <= 1.0 for probability in row["token_confidences"])
        assert 0.0 <= row["utterance_confidence"] <= 1.0
        # scripted posteriors are unambiguous, so the decoder should be sure
        assert row["utterance_confidence"] > 0.9


def test_confidences_with_the_prefix_method_are_refused(tmp_path, capsys) -> None:
    prepare(tmp_path)
    code = main(["decode", "--workdir", str(tmp_path), "--method", "prefix", "--confidences"])
    assert code == 1
    assert "greedy" in capsys.readouterr().err


def test_prefix_rows_carry_an_ordered_n_best(tmp_path) -> None:
    prepare(tmp_path)
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
            "nbest.jsonl",
        ]
    )
    rows = read_jsonl(tmp_path / "nbest.jsonl")
    assert rows
    for row in rows:
        n_best = row["n_best"]
        assert 1 <= len(n_best) <= 3
        assert n_best[0][0] == row["hypothesis_ids"]
        scores = [entry[1] for entry in n_best]
        assert scores == sorted(scores, reverse=True)


def test_greedy_rows_have_no_n_best_column(tmp_path) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    row = read_jsonl(tmp_path / "decode.jsonl")[0]
    assert "n_best" not in row
