"""Chunked decoding has to agree with the batch decoder it stands in for."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.chunked import (
    ChunkConfig,
    ChunkResult,
    ChunkState,
    decode_all_chunks,
    decode_chunk,
)
from echoscribe.decode.transducer import TableJoint, TransducerConfig, transducer_greedy
from echoscribe.errors import DecodeError


def _joint(frames: int = 12, labels: int = 40, seed: int = 0) -> TableJoint:
    rng = np.random.default_rng(seed)
    return TableJoint(rng.uniform(-2.0, 2.0, size=(frames, labels + 1, 3)))


def _config(frames_per_chunk: int, cap: int = 3) -> ChunkConfig:
    return ChunkConfig(frames_per_chunk=frames_per_chunk, max_symbols_per_frame=cap)


def _batch(joint: TableJoint, cap: int = 3):
    return transducer_greedy(joint, TransducerConfig(max_symbols_per_frame=cap))


@pytest.mark.parametrize("size", [1, 2, 3, 5, 12, 50])
def test_every_chunk_size_reproduces_the_batch_decode(size: int) -> None:
    joint = _joint(seed=1)
    batch = _batch(joint)
    chunked = decode_all_chunks(joint, _config(size))
    assert chunked.symbols == batch.symbols
    assert chunked.score == pytest.approx(batch.score)


def test_the_frame_bookkeeping_matches_the_batch_decode() -> None:
    joint = _joint(frames=9, seed=2)
    assert decode_all_chunks(joint, _config(3)).frames == _batch(joint).frames


def test_the_chunked_score_accumulates_across_chunks() -> None:
    joint = _joint(frames=9, seed=3)
    state = ChunkState()
    while state.frame < joint.num_frames and not state.finished:
        state = decode_chunk(joint, state, _config(3)).state
    assert state.score == pytest.approx(_batch(joint).score)


def test_a_fresh_state_starts_at_frame_zero() -> None:
    state = ChunkState()
    assert state.frame == 0
    assert state.labels == ()
    assert state.score == 0.0
    assert not state.finished


def test_a_chunk_advances_by_at_most_its_budget() -> None:
    joint = _joint(frames=20, seed=4)
    state = ChunkState()
    for _ in range(4):
        result = decode_chunk(joint, state, _config(3))
        assert result.frames_consumed <= 3
        assert result.state.frame == state.frame + result.frames_consumed
        state = result.state


def test_decoding_past_the_end_is_a_no_op() -> None:
    joint = _joint(frames=4, seed=5)
    state = ChunkState(frame=4)
    result = decode_chunk(joint, state, _config(8))
    assert result.symbols == ()
    assert result.frames_consumed == 0
    assert result.state == state


def test_a_finished_state_stays_finished() -> None:
    state = ChunkState(frame=2, finished=True)
    result = decode_chunk(_joint(frames=6, seed=6), state, _config(4))
    assert isinstance(result, ChunkResult)
    assert result.state is state
    assert result.symbols == ()


def test_a_state_outside_the_audio_is_refused() -> None:
    with pytest.raises(DecodeError, match="is outside"):
        decode_chunk(_joint(frames=3), ChunkState(frame=9), _config(2))


def test_a_blank_outside_the_alphabet_is_refused() -> None:
    with pytest.raises(DecodeError, match="outside the 3-symbol alphabet"):
        decode_chunk(_joint(frames=3), None, ChunkConfig(blank_id=9))


def test_a_config_rejects_a_non_positive_chunk() -> None:
    with pytest.raises(DecodeError, match="frames_per_chunk must be positive"):
        ChunkConfig(frames_per_chunk=0)


def test_labels_carry_across_a_chunk_boundary() -> None:
    # The predictor state is the emitted-label history, so the second chunk has to
    # see what the first one produced rather than starting over.
    logits = np.full((6, 20, 3), -5.0)
    logits[:, 0, 1] = 5.0
    logits[:, 1, 2] = 5.0
    logits[:, 2:, 0] = 5.0
    joint = TableJoint(logits)
    first = decode_chunk(joint, None, ChunkConfig(frames_per_chunk=1, max_symbols_per_frame=1))
    second = decode_chunk(joint, first.state, _config(5))
    assert first.symbols == (1,)
    assert second.symbols == (2,)
    assert second.state.labels == (1, 2)


def test_a_chunk_reports_spans_for_the_labels_it_emitted() -> None:
    logits = np.full((4, 4, 3), -5.0)
    logits[:, 0, 1] = 5.0
    logits[:, 1:, 0] = 5.0
    result = decode_chunk(TableJoint(logits), None, _config(2))
    assert result.symbols == (1,)
    assert result.tokens[0].start == 0
    assert result.tokens[0].end == 2
