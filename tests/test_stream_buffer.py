"""The streaming buffer has to be indistinguishable from batch framing."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import StreamError
from echoscribe.framing import FrameConfig, extract_frames, frame_count
from echoscribe.stream.buffer import FrameBuffer

SNIPPED = FrameConfig(snip_edges=True)


def _signal(num_samples: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, num_samples)


@pytest.mark.parametrize("chunk", [1, 7, 160, 400, 1024, 5000])
def test_chunked_pushes_reproduce_batch_framing(chunk: int) -> None:
    signal = _signal()
    buffer = FrameBuffer(SNIPPED)
    streamed = []
    for start in range(0, len(signal), chunk):
        streamed.append(buffer.push(signal[start : start + chunk]))
    joined = np.vstack(streamed) if streamed else np.zeros((0, 400))
    assert np.array_equal(joined, extract_frames(signal, SNIPPED))


def test_the_frame_count_matches_the_batch_formula() -> None:
    signal = _signal(3200)
    buffer = FrameBuffer(SNIPPED)
    buffer.push(signal)
    assert buffer.frames_emitted == frame_count(3200, SNIPPED)


def test_pending_samples_never_reach_a_full_frame() -> None:
    buffer = FrameBuffer(SNIPPED)
    for chunk in (100, 159, 161, 400, 1):
        buffer.push(np.zeros(chunk))
        assert buffer.pending_samples < SNIPPED.frame_length


def test_pushing_nothing_emits_nothing() -> None:
    buffer = FrameBuffer(SNIPPED)
    assert buffer.push(np.zeros(0)).shape == (0, 400)
    assert buffer.frames_emitted == 0


def test_a_flush_pads_the_tail_into_one_more_frame() -> None:
    buffer = FrameBuffer(SNIPPED)
    buffer.push(np.ones(450))
    assert buffer.frames_emitted == 1
    tail = buffer.flush()
    assert tail.shape == (1, 400)
    assert buffer.frames_emitted == 2
    assert buffer.pending_samples == 0


def test_a_flush_of_an_empty_buffer_is_empty() -> None:
    assert FrameBuffer(SNIPPED).flush().shape == (0, 400)


def test_a_flush_twice_emits_nothing_the_second_time() -> None:
    buffer = FrameBuffer(SNIPPED)
    buffer.push(np.ones(450))
    buffer.flush()
    assert buffer.flush().shape == (0, 400)


def test_a_reset_drops_pending_samples_and_the_count() -> None:
    buffer = FrameBuffer(SNIPPED)
    buffer.push(np.ones(500))
    buffer.reset()
    assert buffer.pending_samples == 0
    assert buffer.frames_emitted == 0


def test_a_padded_config_is_refused() -> None:
    with pytest.raises(StreamError, match="needs snip_edges=True"):
        FrameBuffer(FrameConfig(snip_edges=False))


def test_a_matrix_is_refused() -> None:
    with pytest.raises(StreamError, match="needs 1-D samples"):
        FrameBuffer(SNIPPED).push(np.zeros((4, 2)))


def test_a_negative_frame_count_is_refused() -> None:
    with pytest.raises(StreamError, match="must not be negative"):
        FrameBuffer(SNIPPED, frames_emitted=-1)


def test_the_tail_is_zero_padded_not_replicated() -> None:
    buffer = FrameBuffer(SNIPPED)
    buffer.push(np.full(450, 0.7))
    held = buffer.pending_samples
    tail = buffer.flush()
    assert held > 0
    assert np.allclose(tail[0, :held], 0.7)
    assert np.all(tail[0, held:] == 0.0)


def test_every_split_point_of_a_short_signal_agrees_with_batch() -> None:
    signal = _signal(1200, seed=3)
    expected = extract_frames(signal, SNIPPED)
    for first in range(0, 1200, 97):
        buffer = FrameBuffer(SNIPPED)
        head = buffer.push(signal[:first])
        tail = buffer.push(signal[first:])
        assert np.array_equal(np.vstack([head, tail]), expected), first


def test_a_non_default_frame_config_streams_correctly() -> None:
    config = FrameConfig(sample_rate=8000, frame_length_ms=32.0, frame_shift_ms=8.0)
    signal = _signal(2000, seed=4)
    buffer = FrameBuffer(config)
    streamed = buffer.push(signal[:600])
    streamed = np.vstack([streamed, buffer.push(signal[600:])])
    assert np.array_equal(streamed, extract_frames(signal, config))


def test_single_sample_pushes_still_produce_every_frame() -> None:
    signal = _signal(900, seed=5)
    buffer = FrameBuffer(SNIPPED)
    frames = [buffer.push(signal[index : index + 1]) for index in range(len(signal))]
    joined = np.vstack([block for block in frames if block.size])
    assert np.array_equal(joined, extract_frames(signal, SNIPPED))


def test_the_pending_tail_is_exactly_the_unconsumed_samples() -> None:
    signal = _signal(1000, seed=6)
    buffer = FrameBuffer(SNIPPED)
    buffer.push(signal)
    consumed = buffer.frames_emitted * SNIPPED.frame_shift
    assert np.array_equal(buffer.pending, signal[consumed:])
