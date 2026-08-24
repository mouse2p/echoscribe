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


def test_the_same_seed_renders_the_same_manifest(tmp_path) -> None:
    left, right = tmp_path / "a", tmp_path / "b"
    main(["synth", "--workdir", str(left), "--utterances", "3", "--seed", "11"])
    main(["synth", "--workdir", str(right), "--utterances", "3", "--seed", "11"])
    assert (left / "manifest.jsonl").read_bytes() == (right / "manifest.jsonl").read_bytes()


def test_a_different_seed_renders_a_different_manifest(tmp_path) -> None:
    left, right = tmp_path / "a", tmp_path / "b"
    main(["synth", "--workdir", str(left), "--utterances", "3", "--seed", "11"])
    main(["synth", "--workdir", str(right), "--utterances", "3", "--seed", "12"])
    assert (left / "manifest.jsonl").read_bytes() != (right / "manifest.jsonl").read_bytes()


def test_a_negative_utterance_count_fails_with_an_explanation(tmp_path, capsys) -> None:
    code = main(["synth", "--workdir", str(tmp_path), "--utterances", "-1"])
    assert code == 1
    assert "non-negative" in capsys.readouterr().err


def test_an_english_probability_above_one_fails(tmp_path, capsys) -> None:
    code = main(
        ["synth", "--workdir", str(tmp_path), "--utterances", "2", "--english-probability", "1.5"]
    )
    assert code == 1
    assert capsys.readouterr().err.startswith("echoscribe: error:")


def test_a_clean_corpus_records_a_clean_condition(tmp_path) -> None:
    main(["synth", "--workdir", str(tmp_path), "--utterances", "2", "--seed", "5"])
    record = json.loads((tmp_path / "manifest.jsonl").read_text().splitlines()[0])
    assert record["condition"]["noise_kind"] is None


def test_a_planted_mix_measures_close_to_its_requested_snr(tmp_path) -> None:
    import numpy as np

    from echoscribe.audio import read_wav
    from echoscribe.noise.snr import snr_db

    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    main(["synth", "--workdir", str(clean), "--utterances", "2", "--seed", "5"])
    main(
        [
            "synth",
            "--workdir",
            str(noisy),
            "--utterances",
            "2",
            "--seed",
            "5",
            "--noise-kind",
            "white",
            "--snr",
            "5",
        ]
    )
    rows = (noisy / "manifest.jsonl").read_text().splitlines()
    record = json.loads(rows[0])
    assert record["condition"]["noise_kind"] == "white"
    assert record["condition"]["noise_snr_db"] == 5.0
    clean_audio = read_wav(
        clean / json.loads((clean / "manifest.jsonl").read_text().splitlines()[0])["wav_path"]
    )
    noisy_audio = read_wav(noisy / record["wav_path"])
    assert not np.array_equal(clean_audio.samples, noisy_audio.samples)
    measured = snr_db(clean_audio.samples, noisy_audio.samples)
    assert abs(measured - 5.0) < 0.5


def test_an_snr_without_a_noise_kind_fails(tmp_path, capsys) -> None:
    assert main(["synth", "--workdir", str(tmp_path), "--snr", "5"]) == 1
    assert "--noise-kind" in capsys.readouterr().err
