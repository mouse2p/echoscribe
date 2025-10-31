"""A session is the record of a run, so every field is pinned."""

from __future__ import annotations

import pytest

from echoscribe.errors import SchemaError, StreamError
from echoscribe.stream.session import SESSION_SCHEMA_VERSION, UtteranceRecord


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
