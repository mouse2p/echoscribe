"""The synth subcommand is run for real, against a temporary directory."""

from __future__ import annotations

import json

from echoscribe.cli import main


def test_synth_fills_the_work_directory(tmp_path, capsys) -> None:
    code = main(["synth", "--workdir", str(tmp_path), "--utterances", "3", "--seed", "7"])
    assert code == 0
    summary = capsys.readouterr().out
    assert "3 utterances" in summary
    manifest = tmp_path / "manifest.jsonl"
    assert manifest.exists()
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 3
    assert len(list((tmp_path / "wav").glob("*.wav"))) == 3


def test_every_record_carries_a_planted_transcript(tmp_path) -> None:
    main(["synth", "--workdir", str(tmp_path), "--utterances", "4", "--seed", "8"])
    records = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    for record in records:
        assert record["text"]
        assert record["tokens"]
        assert len(record["tokens"]) == len(record["languages"])
        assert record["duration"] > 0.0


def test_the_token_bounds_are_honoured(tmp_path) -> None:
    main(
        [
            "synth",
            "--workdir",
            str(tmp_path),
            "--utterances",
            "6",
            "--min-tokens",
            "2",
            "--max-tokens",
            "3",
            "--seed",
            "9",
        ]
    )
    records = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    for record in records:
        assert 2 <= len(record["tokens"]) <= 3
