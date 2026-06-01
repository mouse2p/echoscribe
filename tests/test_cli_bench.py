"""The bench command is checked on the structure of what it records."""

from __future__ import annotations

from echoscribe.cli import main
from echoscribe.cli.common import read_json
from echoscribe.eval.bench import BenchResult


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


def test_bench_writes_a_schema_tagged_result(tmp_path) -> None:
    prepare(tmp_path)
    code = main(["bench", "--workdir", str(tmp_path), "--warmups", "0", "--repeats", "2"])
    assert code == 0
    payload = read_json(tmp_path / "bench.json")
    assert payload["schema"] == 1
    assert payload["task"] == "ctc-greedy-decode"
    assert len(payload["samples"]) == 2
    restored = BenchResult.from_dict(payload)
    assert restored.rtf >= 0.0


def test_the_summary_prints_the_median_and_the_best(tmp_path, capsys) -> None:
    prepare(tmp_path)
    main(["bench", "--workdir", str(tmp_path), "--warmups", "0", "--repeats", "2"])
    out = capsys.readouterr().out
    assert "RTF" in out
    assert "single-process CPU timing" in out


def test_the_prefix_method_is_timed_under_its_own_name(tmp_path) -> None:
    prepare(tmp_path)
    main(
        [
            "bench",
            "--workdir",
            str(tmp_path),
            "--method",
            "prefix",
            "--beam",
            "3",
            "--warmups",
            "0",
            "--repeats",
            "1",
        ]
    )
    assert read_json(tmp_path / "bench.json")["task"] == "ctc-prefix-decode"


def test_zero_repeats_is_refused(tmp_path, capsys) -> None:
    prepare(tmp_path)
    assert main(["bench", "--workdir", str(tmp_path), "--repeats", "0"]) == 1
    assert "at least 1" in capsys.readouterr().err
