"""The stream command is checked against the library's own session reader."""

from __future__ import annotations

from echoscribe.cli import main
from echoscribe.cli.common import read_jsonl
from echoscribe.cli.scripted import load_scripted_corpus
from echoscribe.stream.session import read_session


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


def test_a_streamed_session_replays_every_planted_transcript(tmp_path, capsys) -> None:
    prepare(tmp_path)
    code = main(["stream", "--workdir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "2 replayed their planted transcript exactly" in out
    assert "scripted fixture logits" in out
    session = read_session(tmp_path / "session.jsonl")
    assert len(session.utterances) == 2
    for record in session.utterances:
        assert record.frames > 0
        assert record.events >= 1
        assert record.reference


def test_the_hypotheses_are_the_planted_unit_ids(tmp_path) -> None:
    prepare(tmp_path)
    main(["stream", "--workdir", str(tmp_path)])
    session = read_session(tmp_path / "session.jsonl")
    utterances, _ = load_scripted_corpus(tmp_path)
    planted = {item.utterance_id: item.unit_ids for item in utterances}
    for record in session.utterances:
        assert record.hypothesis == planted[record.utterance_id]


def test_the_chunk_size_does_not_change_the_hypothesis(tmp_path) -> None:
    prepare(tmp_path)
    main(["stream", "--workdir", str(tmp_path), "--frames-per-chunk", "4", "--out", "small.jsonl"])
    main(["stream", "--workdir", str(tmp_path), "--frames-per-chunk", "64", "--out", "big.jsonl"])
    small = read_session(tmp_path / "small.jsonl")
    big = read_session(tmp_path / "big.jsonl")
    assert [record.hypothesis for record in small.utterances] == [
        record.hypothesis for record in big.utterances
    ]


def test_the_session_id_is_a_function_of_the_corpus(tmp_path) -> None:
    prepare(tmp_path)
    main(["stream", "--workdir", str(tmp_path), "--out", "one.jsonl"])
    main(["stream", "--workdir", str(tmp_path), "--out", "two.jsonl"])
    assert (
        read_session(tmp_path / "one.jsonl").session_id
        == read_session(tmp_path / "two.jsonl").session_id
    )


def test_the_session_metadata_records_the_command(tmp_path) -> None:
    prepare(tmp_path)
    main(["stream", "--workdir", str(tmp_path), "--frames-per-chunk", "8"])
    session = read_session(tmp_path / "session.jsonl")
    assert session.metadata["command"] == "stream"
    assert session.metadata["frames_per_chunk"] == 8
    assert session.metadata["posteriors"] == "scripted fixtures"


def test_the_stream_agrees_with_the_batch_decode(tmp_path) -> None:
    prepare(tmp_path)
    main(["decode", "--workdir", str(tmp_path)])
    main(["stream", "--workdir", str(tmp_path)])
    decode_rows = {
        row["utterance_id"]: row["hypothesis_ids"] for row in read_jsonl(tmp_path / "decode.jsonl")
    }
    session = read_session(tmp_path / "session.jsonl")
    for record in session.utterances:
        assert list(record.hypothesis) == decode_rows[record.utterance_id]


def test_the_summary_states_the_latency_budget(tmp_path, capsys) -> None:
    prepare(tmp_path)
    main(["stream", "--workdir", str(tmp_path), "--frames-per-chunk", "8"])
    out = capsys.readouterr().out
    assert "chunks of 80 ms" in out
    assert "right context 40 ms" in out
