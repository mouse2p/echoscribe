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


@pytest.mark.slow
def test_the_demo_says_the_audio_is_synthetic(tmp_path, capsys) -> None:
    main(["demo", "--workdir", str(tmp_path), "--utterances", "3", "--repeats", "1"])
    out = capsys.readouterr().out
    assert "The audio is synthetic and the transcripts are planted" in out
    assert "not recognizer accuracy" in out


def test_the_demo_reports_which_stage_failed(tmp_path, monkeypatch, capsys) -> None:
    import echoscribe.cli.demo as demo

    monkeypatch.setattr(demo, "run_synth", lambda namespace: 1)
    code = main(["demo", "--workdir", str(tmp_path)])
    assert code == 1
    assert "failed inside its synth stage" in capsys.readouterr().err


def test_every_registered_command_appears_in_the_help(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in (
        "bench",
        "bias",
        "decode",
        "demo",
        "eval",
        "feats",
        "report",
        "stream",
        "synth",
        "train",
    ):
        assert command in out


@pytest.mark.slow
def test_the_demo_can_train_and_decode_through_the_model(tmp_path, capsys) -> None:
    pytest.importorskip("torch", reason="the torch extra is not installed")
    code = main(
        [
            "demo",
            "--workdir",
            str(tmp_path),
            "--utterances",
            "4",
            "--repeats",
            "1",
            "--train",
            "--epochs",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert (tmp_path / "scores-bundle.json").exists()
    assert (tmp_path / "decode-bundle.jsonl").exists()
    assert "trained-bundle UER" in out
    assert "memorization" in out
