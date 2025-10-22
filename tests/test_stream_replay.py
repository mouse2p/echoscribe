"""Replay is the debugging contract, so it is checked for exactness."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import StreamError
from echoscribe.stream.protocol import Hypothesis, StreamEvent
from echoscribe.stream.replay import (
    RECORDING_SCHEMA_VERSION,
    StreamRecording,
    record,
    replay,
    trace_digest,
)


def _recording(frames: int = 12, symbols: int = 4, blocks=(4, 4, 4), seed: int = 0):
    logits = np.random.default_rng(seed).uniform(-2.0, 2.0, size=(frames, symbols))
    return StreamRecording(logits, tuple(blocks), symbols, 4)


def test_a_recording_reports_its_frame_count() -> None:
    assert _recording().num_frames == 12


def test_blocks_reassemble_the_recording_in_order() -> None:
    recording = _recording(frames=10, blocks=(3, 7))
    assert np.array_equal(np.vstack(list(recording.blocks())), recording.logits)
    assert [block.shape[0] for block in recording.blocks()] == [3, 7]


def test_a_digest_is_stable_and_short() -> None:
    left = _recording(seed=1).digest()
    assert left == _recording(seed=1).digest()
    assert len(left) == 16


def test_a_digest_changes_with_the_chunking() -> None:
    logits = np.random.default_rng(2).uniform(-1.0, 1.0, size=(8, 4))
    assert (
        StreamRecording(logits, (8,), 4, 4).digest()
        != StreamRecording(logits, (4, 4), 4, 4).digest()
    )


def test_mismatched_block_sizes_are_refused() -> None:
    logits = np.zeros((6, 4))
    with pytest.raises(StreamError, match="block sizes add to 7"):
        StreamRecording(logits, (3, 4), 4, 4)


def test_a_wrong_symbol_count_is_refused() -> None:
    with pytest.raises(StreamError, match="symbols per frame"):
        StreamRecording(np.zeros((4, 3)), (4,), 5, 4)


def test_a_flat_matrix_is_refused() -> None:
    with pytest.raises(StreamError, match="got shape"):
        StreamRecording(np.zeros(4), (4,), 4, 4)


def test_a_non_positive_chunk_size_is_refused() -> None:
    with pytest.raises(StreamError, match="chunk_frames must be positive"):
        StreamRecording(np.zeros((4, 3)), (4,), 3, 0)


def test_a_trace_digest_covers_kind_frame_and_symbols() -> None:
    first = (StreamEvent("partial", 4, Hypothesis((1, 2))),)
    second = (StreamEvent("partial", 4, Hypothesis((1, 3))),)
    third = (StreamEvent("final", 4, Hypothesis((1, 2))),)
    assert trace_digest(first) != trace_digest(second)
    assert trace_digest(first) != trace_digest(third)
    assert trace_digest(first) == trace_digest(first)


def test_a_trace_digest_of_nothing_is_still_a_hash() -> None:
    assert len(trace_digest(())) == 64


def test_the_schema_version_is_declared() -> None:
    assert RECORDING_SCHEMA_VERSION == 1


def test_a_replay_is_reproducible() -> None:
    recording = _recording(seed=3)
    assert trace_digest(replay(recording)) == trace_digest(replay(recording))


def test_a_replay_ends_with_exactly_one_final_event() -> None:
    events = replay(_recording(seed=4))
    assert [event.kind for event in events].count("final") == 1
    assert events[-1].kind == "final"


def test_the_chunking_does_not_change_the_final_hypothesis() -> None:
    logits = np.random.default_rng(5).uniform(-2.0, 2.0, size=(24, 4))
    finals = []
    for blocks in ((24,), (8, 8, 8), (6, 6, 6, 6), (1,) * 24):
        recording = StreamRecording(logits, blocks, 4, 6)
        finals.append(replay(recording)[-1].hypothesis.symbols)
    assert all(item == finals[0] for item in finals)


def test_a_recording_helper_round_trips_its_inputs() -> None:
    logits = np.random.default_rng(6).uniform(-1.0, 1.0, size=(9, 3))
    recording = record(logits, (4, 5), chunk_frames=3, max_symbols_per_frame=2)
    assert recording.num_symbols == 3
    assert recording.block_sizes == (4, 5)
    assert recording.chunk_frames == 3
    assert recording.max_symbols_per_frame == 2
    assert np.array_equal(recording.logits, logits)


def test_a_replayed_trace_is_non_empty_for_a_non_empty_recording() -> None:
    assert replay(_recording(seed=7)) != ()


def test_replaying_an_empty_recording_still_finishes() -> None:
    recording = StreamRecording(np.zeros((0, 3)), (), 3, 4)
    events = replay(recording)
    assert [event.kind for event in events] == ["final"]
    assert events[0].hypothesis.is_empty
