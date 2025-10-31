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
