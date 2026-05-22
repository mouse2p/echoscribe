"""The bias command is checked from both sides of the flip threshold."""

from __future__ import annotations

from echoscribe.cli import main
from echoscribe.cli.common import read_jsonl
from echoscribe.cli.scripted import load_scripted_corpus


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


def hotword_of(tmp_path) -> str:
    utterances, _ = load_scripted_corpus(tmp_path)
    return utterances[0].units[0]


def test_the_hotword_flips_the_ambiguous_choice(tmp_path, capsys) -> None:
    prepare(tmp_path)
    hotword = hotword_of(tmp_path)
    code = main(["bias", "--workdir", str(tmp_path), "--hotword", hotword])
    assert code == 0
    row = read_jsonl(tmp_path / "bias.jsonl")[0]
    assert row["flipped"] is True
    assert row["biased_ids"] != row["unbiased_ids"]
    assert row["hotwords_matched"] == 1
    assert "flipped the decode" in capsys.readouterr().out


def test_the_unbiased_side_still_prefers_its_rival(tmp_path) -> None:
    prepare(tmp_path)
    main(["bias", "--workdir", str(tmp_path), "--hotword", hotword_of(tmp_path)])
    row = read_jsonl(tmp_path / "bias.jsonl")[0]
    assert row["unbiased_ids"] == [row["rival_id"]]
    assert row["biased_ids"] == [row["hotword_id"]]


def test_a_bonus_below_the_margin_changes_nothing(tmp_path, capsys) -> None:
    prepare(tmp_path)
    main(
        [
            "bias",
            "--workdir",
            str(tmp_path),
            "--hotword",
            hotword_of(tmp_path),
            "--bonus",
            "0.1",
        ]
    )
    row = read_jsonl(tmp_path / "bias.jsonl")[0]
    assert row["flipped"] is False
    assert row["biased_ids"] == row["unbiased_ids"]
    assert "left the decode unchanged" in capsys.readouterr().out


def test_an_unknown_hotword_fails_with_a_message(tmp_path, capsys) -> None:
    prepare(tmp_path)
    code = main(["bias", "--workdir", str(tmp_path), "--hotword", "not-a-unit"])
    assert code == 1
    assert "not in the modelling-unit vocabulary" in capsys.readouterr().err
