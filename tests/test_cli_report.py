"""The report command is checked on the bytes it leaves behind."""

from __future__ import annotations

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


def score_first(tmp_path) -> None:
    """A work directory carried all the way through decode and eval."""
    prepare(tmp_path)
    assert main(["decode", "--workdir", str(tmp_path)]) == 0
    assert main(["eval", "--workdir", str(tmp_path)]) == 0


def test_the_report_renders_after_eval(tmp_path) -> None:
    score_first(tmp_path)
    code = main(["report", "--workdir", str(tmp_path)])
    assert code == 0
    document = (tmp_path / "report.svg").read_text(encoding="utf-8")
    assert document.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<title>echoscribe report</title>" in document
    assert ">scores (units)<" in document


def test_the_report_names_every_utterance(tmp_path) -> None:
    score_first(tmp_path)
    main(["report", "--workdir", str(tmp_path)])
    document = (tmp_path / "report.svg").read_text(encoding="utf-8")
    scores = read_json(tmp_path / "scores.json")
    for row in scores["utterances"]:
        assert row["utterance_id"] in document


def test_the_metadata_stamp_is_inside_the_document(tmp_path) -> None:
    score_first(tmp_path)
    main(["report", "--workdir", str(tmp_path), "--label", "demo-run"])
    document = (tmp_path / "report.svg").read_text(encoding="utf-8")
    assert "run=demo-run seed=0 digest=" in document


def test_report_without_scores_fails_helpfully(tmp_path, capsys) -> None:
    prepare(tmp_path)
    assert main(["report", "--workdir", str(tmp_path)]) == 1
    assert "run `echoscribe eval` first" in capsys.readouterr().err
