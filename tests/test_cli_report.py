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


def test_two_runs_over_the_same_files_agree_byte_for_byte(tmp_path, capsys) -> None:
    score_first(tmp_path)
    main(["report", "--workdir", str(tmp_path), "--out", "one.svg"])
    main(["report", "--workdir", str(tmp_path), "--out", "two.svg"])
    assert (tmp_path / "one.svg").read_bytes() == (tmp_path / "two.svg").read_bytes()
    digests = [
        line.split("digest: ")[1]
        for line in capsys.readouterr().out.splitlines()
        if "digest: " in line
    ]
    assert len(digests) == 2
    assert digests[0] == digests[1]


def test_a_different_label_moves_the_digest(tmp_path, capsys) -> None:
    score_first(tmp_path)
    main(["report", "--workdir", str(tmp_path), "--label", "a", "--out", "a.svg"])
    main(["report", "--workdir", str(tmp_path), "--label", "b", "--out", "b.svg"])
    assert (tmp_path / "a.svg").read_bytes() != (tmp_path / "b.svg").read_bytes()


def test_the_bench_section_appears_when_bench_ran(tmp_path) -> None:
    score_first(tmp_path)
    main(["bench", "--workdir", str(tmp_path), "--warmups", "0", "--repeats", "3"])
    main(["report", "--workdir", str(tmp_path)])
    document = (tmp_path / "report.svg").read_text(encoding="utf-8")
    assert ">decode seconds per repeat<" in document


def test_the_report_works_without_a_bench_file(tmp_path) -> None:
    score_first(tmp_path)
    main(["report", "--workdir", str(tmp_path)])
    document = (tmp_path / "report.svg").read_text(encoding="utf-8")
    assert ">decode seconds per repeat<" not in document


def test_a_corrupt_score_file_is_refused(tmp_path) -> None:
    prepare(tmp_path)
    (tmp_path / "scores.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert main(["report", "--workdir", str(tmp_path)]) == 1


def test_the_alignment_section_names_the_first_utterance(tmp_path) -> None:
    score_first(tmp_path)
    main(["report", "--workdir", str(tmp_path)])
    document = (tmp_path / "report.svg").read_text(encoding="utf-8")
    scores = read_json(tmp_path / "scores.json")
    first_id = scores["utterances"][0]["utterance_id"]
    assert f">alignment of {first_id}<" in document


def test_the_report_still_renders_without_a_decode_file(tmp_path) -> None:
    score_first(tmp_path)
    (tmp_path / "decode.jsonl").unlink()
    assert main(["report", "--workdir", str(tmp_path)]) == 0
    document = (tmp_path / "report.svg").read_text(encoding="utf-8")
    assert ">alignment of" not in document
