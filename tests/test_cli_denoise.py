"""The denoise command is measured against planted clean/noisy pairs."""

from __future__ import annotations

from echoscribe.cli import main


def render_pair(tmp_path, snr: str = "5"):
    """A clean render and a noisy render of one seeded corpus."""
    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    for directory in (clean, noisy):
        arguments = [
            "synth",
            "--workdir",
            str(directory),
            "--utterances",
            "2",
            "--max-tokens",
            "2",
            "--seed",
            "5",
        ]
        if directory == noisy:
            arguments += ["--noise-kind", "white", "--snr", snr]
        assert main(arguments) == 0
    return clean, noisy


def test_denoise_writes_one_wav_per_utterance(tmp_path) -> None:
    clean, noisy = render_pair(tmp_path)
    code = main(["denoise", "--workdir", str(noisy)])
    assert code == 0
    assert len(list((noisy / "denoised").glob("*.wav"))) == 2


def test_a_planted_pair_measures_a_real_improvement(tmp_path, capsys) -> None:
    clean, noisy = render_pair(tmp_path)
    code = main(["denoise", "--workdir", str(noisy), "--clean-workdir", str(clean)])
    assert code == 0
    out = capsys.readouterr().out
    assert "SNR improvement" in out
    gain = float(out.split("SNR improvement: ")[1].split(" dB")[0])
    assert gain > 0.0


def test_the_wiener_method_also_improves_the_pair(tmp_path, capsys) -> None:
    clean, noisy = render_pair(tmp_path)
    main(
        [
            "denoise",
            "--workdir",
            str(noisy),
            "--clean-workdir",
            str(clean),
            "--method",
            "wiener",
        ]
    )
    gain = float(capsys.readouterr().out.split("SNR improvement: ")[1].split(" dB")[0])
    assert gain > 0.0


def test_without_a_reference_no_snr_is_claimed(tmp_path, capsys) -> None:
    _, noisy = render_pair(tmp_path)
    main(["denoise", "--workdir", str(noisy)])
    assert "not an SNR claim" in capsys.readouterr().out


def test_a_missing_manifest_is_explained(tmp_path, capsys) -> None:
    assert main(["denoise", "--workdir", str(tmp_path)]) == 1
    assert "echoscribe synth" in capsys.readouterr().err


def test_a_clean_dir_without_a_manifest_is_explained(tmp_path, capsys) -> None:
    _, noisy = render_pair(tmp_path)
    assert (
        main(["denoise", "--workdir", str(noisy), "--clean-workdir", str(tmp_path / "empty")]) == 1
    )
    assert "clean render" in capsys.readouterr().err
