"""A session is the record of a run, so every field is pinned."""

from __future__ import annotations

import pytest

from echoscribe.errors import SchemaError, StreamError
from echoscribe.stream.session import (
    SESSION_SCHEMA_VERSION,
    Session,
    UtteranceRecord,
)


def _record(**overrides) -> UtteranceRecord:
    fields = {
        "utterance_id": "utt-00001",
        "hypothesis": (1, 2, 3),
        "reference": "hello 你好",
        "text": "hello 你好",
        "frames": 120,
        "events": 14,
        "endpoint_frames": None,
        "score": -12.5,
    }
    fields.update(overrides)
    return UtteranceRecord(**fields)


def test_a_record_keeps_every_field() -> None:
    record = _record()
    assert record.utterance_id == "utt-00001"
    assert record.hypothesis == (1, 2, 3)
    assert record.frames == 120
    assert not record.endpointed


def test_an_endpoint_frame_marks_the_record_as_endpointed() -> None:
    assert _record(endpoint_frames=90).endpointed


def test_a_record_round_trips_through_a_dict() -> None:
    record = _record(endpoint_frames=90)
    assert UtteranceRecord.from_dict(record.to_dict()) == record


def test_the_serialized_shape_is_fixed() -> None:
    assert set(_record().to_dict()) == {
        "utterance_id",
        "hypothesis",
        "reference",
        "text",
        "frames",
        "events",
        "endpoint_frames",
        "score",
    }


def test_the_score_is_rounded_where_it_is_produced() -> None:
    assert _record(score=1.0 / 3.0).to_dict()["score"] == 0.333333333333


def test_an_endpoint_after_the_last_frame_is_refused() -> None:
    with pytest.raises(StreamError, match="the endpoint fired at frame 200 of 120"):
        _record(endpoint_frames=200)


def test_a_negative_endpoint_is_refused() -> None:
    with pytest.raises(StreamError, match="the endpoint fired at frame -1"):
        _record(endpoint_frames=-1)


def test_a_record_needs_an_id() -> None:
    with pytest.raises(StreamError, match="needs an id"):
        _record(utterance_id="")


def test_negative_counts_are_refused() -> None:
    with pytest.raises(StreamError, match="frames must not be negative"):
        _record(frames=-1)
    with pytest.raises(StreamError, match="events must not be negative"):
        _record(events=-1)


def test_a_negative_symbol_is_refused() -> None:
    with pytest.raises(StreamError, match="negative symbol"):
        _record(hypothesis=(1, -2))


def test_a_missing_field_is_refused() -> None:
    payload = _record().to_dict()
    del payload["frames"]
    with pytest.raises(SchemaError, match="missing 'frames'"):
        UtteranceRecord.from_dict(payload)


def test_the_schema_version_is_declared() -> None:
    assert SESSION_SCHEMA_VERSION == 1


def test_a_record_is_immutable() -> None:
    with pytest.raises(AttributeError):
        _record().frames = 1  # type: ignore[misc]


def _session(count: int = 3, endpoint_every: int = 0) -> Session:
    records = tuple(
        _record(
            utterance_id=f"utt-{index:05d}",
            frames=100 + index,
            endpoint_frames=(90 + index)
            if endpoint_every and index % endpoint_every == 0
            else None,
        )
        for index in range(count)
    )
    return Session("session-1", {"seed": 7}, records)


def test_a_session_counts_its_utterances() -> None:
    session = _session(4)
    assert len(session) == 4
    assert isinstance(session, Session)


def test_total_frames_sums_the_records() -> None:
    assert _session(3).total_frames == 100 + 101 + 102


def test_the_endpoint_rate_counts_only_fired_endpoints() -> None:
    assert _session(4, endpoint_every=2).endpoint_rate == pytest.approx(0.5)
    assert _session(4).endpoint_rate == 0.0


def test_an_empty_session_has_a_zero_endpoint_rate() -> None:
    assert Session("s", {}, ()).endpoint_rate == 0.0


def test_the_mean_endpoint_delay_is_negative_when_it_fires_early() -> None:
    session = _session(3, endpoint_every=1)
    assert session.mean_endpoint_delay() == pytest.approx(-10.0)


def test_the_mean_endpoint_delay_of_nothing_is_zero() -> None:
    assert _session(3).mean_endpoint_delay() == 0.0


def test_a_summary_reports_the_quoted_numbers() -> None:
    summary = _session(3, endpoint_every=1).summary()
    assert summary["utterances"] == 3
    assert summary["frames"] == 303
    assert summary["endpoint_rate"] == 1.0
    assert summary["mean_endpoint_delay"] == -10.0


def test_an_empty_summary_has_no_division_by_zero() -> None:
    summary = Session("s", {}, ()).summary()
    assert summary["utterances"] == 0
    assert summary["mean_score"] == 0.0


def test_a_record_can_be_looked_up_by_id() -> None:
    session = _session(3)
    assert session.record("utt-00001").frames == 101


def test_an_unknown_id_is_refused() -> None:
    with pytest.raises(StreamError, match="is not part of session"):
        _session(2).record("nope")


def test_a_repeated_utterance_id_is_refused() -> None:
    with pytest.raises(StreamError, match="repeated utterance id"):
        Session("s", {}, (_record(), _record()))


def test_a_session_needs_an_id() -> None:
    with pytest.raises(StreamError, match="a session needs an id"):
        Session("", {}, ())


def test_a_session_round_trips(tmp_path) -> None:
    from echoscribe.stream.session import read_session, write_session

    session = _session(3, endpoint_every=2)
    target = tmp_path / "sessions" / "run.jsonl"
    write_session(target, session)
    restored = read_session(target)
    assert restored.session_id == session.session_id
    assert restored.metadata == session.metadata
    assert restored.utterances == session.utterances


def test_a_session_file_has_one_line_per_record_plus_a_header(tmp_path) -> None:
    from echoscribe.stream.session import write_session

    target = tmp_path / "run.jsonl"
    write_session(target, _session(4))
    assert len(target.read_text(encoding="utf-8").splitlines()) == 5


def test_the_header_names_its_schema_and_count(tmp_path) -> None:
    import json

    from echoscribe.stream.session import session_to_lines

    header = json.loads(session_to_lines(_session(2))[0])
    assert header["kind"] == "session"
    assert header["schema"] == SESSION_SCHEMA_VERSION
    assert header["utterances"] == 2


def test_a_foreign_schema_is_refused(tmp_path) -> None:
    import json

    from echoscribe.stream.session import read_session, session_to_lines, write_session

    target = tmp_path / "run.jsonl"
    write_session(target, _session(1))
    lines = session_to_lines(_session(1))
    header = json.loads(lines[0])
    header["schema"] = SESSION_SCHEMA_VERSION + 99
    target.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="is not supported"):
        read_session(target)


def test_a_truncated_file_is_reported(tmp_path) -> None:
    from echoscribe.stream.session import read_session, session_to_lines

    target = tmp_path / "run.jsonl"
    lines = session_to_lines(_session(3))
    target.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    with pytest.raises(StreamError, match="declares 3 utterances but the file holds 1"):
        read_session(target)


def test_a_missing_file_is_reported(tmp_path) -> None:
    from echoscribe.stream.session import read_session

    with pytest.raises(StreamError, match="session file not found"):
        read_session(tmp_path / "absent.jsonl")


def test_an_utterance_before_any_header_is_reported(tmp_path) -> None:
    import json

    from echoscribe.stream.session import read_session

    target = tmp_path / "run.jsonl"
    target.write_text(json.dumps({"kind": "utterance", **_record().to_dict()}), encoding="utf-8")
    with pytest.raises(StreamError, match="before any header"):
        read_session(target)


def test_an_unknown_line_kind_is_reported(tmp_path) -> None:
    import json

    from echoscribe.stream.session import read_session

    target = tmp_path / "run.jsonl"
    target.write_text(json.dumps({"kind": "mystery"}), encoding="utf-8")
    with pytest.raises(StreamError, match="unknown line kind"):
        read_session(target)


def test_non_ascii_transcripts_survive(tmp_path) -> None:
    from echoscribe.stream.session import read_session, write_session

    session = Session("s", {}, (_record(reference="你好 world", text="你好 world"),))
    target = tmp_path / "run.jsonl"
    write_session(target, session)
    assert "你好 world" in target.read_text(encoding="utf-8")
    assert read_session(target).utterances[0].reference == "你好 world"


def test_a_latency_report_converts_frames_to_milliseconds() -> None:
    from echoscribe.stream.session import latency_report

    session = _session(3, endpoint_every=1)
    report = latency_report(session, frame_shift_ms=10.0)
    assert report["endpointed"] == 3
    assert report["mean_ms"] == pytest.approx(100.0)
    assert report["min_ms"] == pytest.approx(100.0)
    assert report["max_ms"] == pytest.approx(100.0)


def test_a_latency_report_scales_with_the_frame_shift() -> None:
    from echoscribe.stream.session import latency_report

    session = _session(2, endpoint_every=1)
    assert latency_report(session, 20.0)["mean_ms"] == pytest.approx(
        2 * latency_report(session, 10.0)["mean_ms"]
    )


def test_a_session_with_no_endpoints_reports_zero_endpoints() -> None:
    from echoscribe.stream.session import latency_report

    report = latency_report(_session(4))
    assert report["endpointed"] == 0
    assert report["utterances"] == 4
    assert report["mean_ms"] == 0.0


def test_a_latency_report_rejects_a_bad_shift() -> None:
    from echoscribe.stream.session import latency_report

    with pytest.raises(StreamError, match="frame_shift_ms must be positive"):
        latency_report(_session(1), 0.0)


def test_the_stream_subpackage_re_exports_the_session_api() -> None:
    from echoscribe import stream

    assert stream.Session is Session
    assert stream.UtteranceRecord is UtteranceRecord
    for name in ("Session", "UtteranceRecord", "read_session", "write_session", "latency_report"):
        assert name in stream.__all__
