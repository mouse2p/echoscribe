"""The eval command is scored by hand before it is trusted with a number."""

from __future__ import annotations

import pytest

from echoscribe.cli import main
from echoscribe.cli.common import read_json, read_jsonl, write_jsonl


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


def test_dropping_one_unit_scores_one_deletion(tmp_path) -> None:
    decode_first(tmp_path)
    rows = read_jsonl(tmp_path / "decode.jsonl")
    rows[0]["hypothesis_units"] = rows[0]["hypothesis_units"][:-1]
    write_jsonl(tmp_path / "decode.jsonl", rows)
    main(["eval", "--workdir", str(tmp_path)])
    payload = read_json(tmp_path / "scores.json")
    assert payload["totals"]["errors"] == 1
    assert payload["totals"]["deletions"] == 1
    assert payload["utterances"][0]["errors"] == 1
    assert payload["utterances"][1]["errors"] == 0


def test_dropping_one_word_scores_one_deletion(tmp_path) -> None:
    decode_first(tmp_path)
    rows = read_jsonl(tmp_path / "decode.jsonl")
    words = rows[0]["reference_text"].split()
    rows[0]["hypothesis_text"] = " ".join(words[:-1])
    write_jsonl(tmp_path / "decode.jsonl", rows)
    main(["eval", "--workdir", str(tmp_path), "--level", "text", "--metric", "wer"])
    payload = read_json(tmp_path / "scores.json")
    assert payload["totals"]["errors"] == 1
    assert payload["totals"]["deletions"] == 1


def test_swapping_one_character_scores_one_substitution(tmp_path) -> None:
    decode_first(tmp_path)
    rows = read_jsonl(tmp_path / "decode.jsonl")
    text = rows[0]["reference_text"]
    characters = [character for character in text if not character.isspace()]
    characters[-1] = "X" if characters[-1] != "X" else "Y"
    rows[0]["hypothesis_text"] = "".join(characters)
    write_jsonl(tmp_path / "decode.jsonl", rows)
    main(["eval", "--workdir", str(tmp_path), "--level", "text", "--metric", "cer"])
    payload = read_json(tmp_path / "scores.json")
    assert payload["totals"]["errors"] == 1
    assert payload["totals"]["substitutions"] == 1


def test_a_hypothesis_missing_from_the_decode_file_fails(tmp_path, capsys) -> None:
    decode_first(tmp_path)
    rows = read_jsonl(tmp_path / "decode.jsonl")[:1]
    write_jsonl(tmp_path / "decode.jsonl", rows)
    assert main(["eval", "--workdir", str(tmp_path)]) == 1
    assert "no hypothesis" in capsys.readouterr().err


def test_a_chinese_corpus_scores_characters(tmp_path) -> None:
    assert (
        main(
            [
                "synth",
                "--workdir",
                str(tmp_path),
                "--utterances",
                "2",
                "--seed",
                "6",
                "--english-probability",
                "0",
            ]
        )
        == 0
    )
    assert main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "40"]) == 0
    assert main(["decode", "--workdir", str(tmp_path)]) == 0
    assert main(["eval", "--workdir", str(tmp_path), "--level", "text", "--metric", "cer"]) == 0
    payload = read_json(tmp_path / "scores.json")
    assert payload["totals"]["errors"] == 0
    rows = read_jsonl(tmp_path / "decode.jsonl")
    assert any(
        "\u4e00" <= character <= "\u9fff" for row in rows for character in row["reference_text"]
    )
