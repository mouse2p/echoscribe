"""The stream has to reach the same answer as the batch decoder, chunk by chunk."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.transducer import TransducerConfig, transducer_greedy
from echoscribe.stream.chunks import ChunkSchedule
from echoscribe.stream.endpoint import BlankRatioConfig, BlankRatioEndpointer
from echoscribe.stream.joint import AppendableJoint
from echoscribe.stream.protocol import EVENT_KINDS, Hypothesis, StreamEvent
from echoscribe.stream.runtime import StreamConfig, StreamDecoder


def _logits(frames: int = 30, symbols: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-2.0, 2.0, size=(frames, symbols))


def _decoder(
    symbols: int = 4, size: int = 6, cap: int = 3
) -> tuple[AppendableJoint, StreamDecoder]:
    joint = AppendableJoint(symbols)
    config = StreamConfig(schedule=ChunkSchedule(frames_per_chunk=size), max_symbols_per_frame=cap)
    return joint, StreamDecoder(joint, config)


@pytest.mark.parametrize("size", [1, 2, 5, 6, 11, 30, 64])
def test_a_streamed_decode_reaches_the_batch_answer(size: int) -> None:
    logits = _logits(seed=1)
    batch = AppendableJoint(4)
    batch.append(logits)
    reference = transducer_greedy(batch, TransducerConfig(max_symbols_per_frame=3))

    joint, decoder = _decoder(size=size)
    for frame in range(len(logits)):
        decoder.push_logits(logits[frame][None, :])
    decoder.finish()
    assert decoder.hypothesis.symbols == reference.symbols
    assert decoder.hypothesis.score == pytest.approx(reference.score)


def test_pushing_a_whole_block_at_once_matches_frame_by_frame() -> None:
    logits = _logits(seed=2)
    _, whole = _decoder(size=5)
    _, batched = _decoder(size=5)
    for frame in range(len(logits)):
        batched.push_logits(logits[frame][None, :])
    batched.finish()
    whole.push_logits(logits)
    whole.finish()
    assert whole.hypothesis.symbols == batched.hypothesis.symbols


def test_events_are_partials_followed_by_one_final() -> None:
    logits = _logits(seed=3)
    joint, decoder = _decoder(size=5, cap=2)
    events = []
    for frame in range(len(logits)):
        events.extend(decoder.push_logits(logits[frame][None, :]))
    events.extend(decoder.finish())
    assert all(event.kind in EVENT_KINDS for event in events)
    assert [event.kind for event in events].count("final") == 1
    assert events[-1].kind == "final"
    assert events[-1].hypothesis.is_final
    assert isinstance(events[-1], StreamEvent)


def test_partials_form_a_growing_prefix_chain() -> None:
    logits = _logits(seed=4)
    joint, decoder = _decoder(size=4, cap=2)
    previous = Hypothesis()
    for frame in range(len(logits)):
        for event in decoder.push_logits(logits[frame][None, :]):
            assert event.hypothesis.extends(previous)
            previous = event.hypothesis
    assert decoder.finish()[-1].hypothesis.extends(previous)


def test_no_partial_arrives_before_its_chunk_is_complete() -> None:
    logits = _logits(frames=9, seed=5)
    joint, decoder = _decoder(size=5, cap=2)
    assert decoder.push_logits(logits[:4]) == ()
    events = decoder.push_logits(logits[4:5])
    assert len(events) == 1
    assert events[0].frame == 5


def test_event_frames_never_go_backwards() -> None:
    logits = _logits(seed=6)
    joint, decoder = _decoder(size=3, cap=2)
    seen = 0
    for frame in range(len(logits)):
        for event in decoder.push_logits(logits[frame][None, :]):
            assert event.frame >= seen
            seen = event.frame


def test_the_hypothesis_is_final_only_after_finish() -> None:
    logits = _logits(frames=12, seed=7)
    joint, decoder = _decoder(size=4, cap=2)
    decoder.push_logits(logits)
    assert not decoder.is_closed
    assert not decoder.hypothesis.is_final
    decoder.finish()
    assert decoder.is_closed
    assert decoder.hypothesis.is_final


def test_a_reset_clears_the_hypothesis() -> None:
    logits = _logits(frames=12, seed=8)
    joint, decoder = _decoder(size=3, cap=2)
    decoder.push_logits(logits[:6])
    event = decoder.reset()
    assert event.kind == "reset"
    assert event.is_terminal
    assert decoder.hypothesis.is_empty
    assert joint.num_frames == 0
    assert not decoder.is_closed


def test_a_decode_can_resume_after_a_reset() -> None:
    logits = _logits(frames=12, seed=9)
    joint, decoder = _decoder(size=4, cap=2)
    decoder.push_logits(logits[:6])
    decoder.reset()
    decoder.push_logits(logits[:6])
    decoder.finish()
    assert decoder.is_closed


def test_a_barge_in_discards_the_partial_hypothesis() -> None:
    logits = _logits(frames=16, seed=10)
    joint, decoder = _decoder(size=2, cap=2)
    decoder.push_logits(logits[:10])
    before = decoder.hypothesis.symbols
    decoder.reset()
    assert decoder.hypothesis.symbols == ()
    decoder.push_logits(logits[10:])
    decoder.finish()
    assert decoder.hypothesis.symbols != before or before == ()


def test_finishing_twice_keeps_the_hypothesis() -> None:
    logits = _logits(frames=8, seed=11)
    joint, decoder = _decoder(size=4, cap=2)
    decoder.push_logits(logits)
    first = decoder.finish()[-1]
    second = decoder.finish()[-1]
    assert first.hypothesis.symbols == second.hypothesis.symbols


def test_an_empty_stream_finishes_with_an_empty_hypothesis() -> None:
    joint, decoder = _decoder()
    event = decoder.finish()[-1]
    assert event.kind == "final"
    assert event.hypothesis.is_empty
    assert event.frame == 0


def test_a_reset_returns_a_terminal_event() -> None:
    joint, decoder = _decoder(size=2)
    event = decoder.reset()
    assert isinstance(event, StreamEvent)
    assert event.kind == "reset"
    assert event.is_terminal
    assert event.frame == 0


def test_a_reset_after_a_finish_reopens_the_stream() -> None:
    logits = _logits(frames=8, seed=20)
    joint, decoder = _decoder(size=4, cap=2)
    decoder.push_logits(logits)
    decoder.finish()
    assert decoder.is_closed
    decoder.reset()
    assert not decoder.is_closed
    assert not decoder.hypothesis.is_final
    assert decoder.hypothesis.frame == 0


def test_a_second_utterance_after_a_reset_decodes_on_its_own() -> None:
    logits = _logits(frames=12, seed=21)
    joint, first = _decoder(size=4, cap=2)
    first.push_logits(logits)
    first.finish()
    expected = first.hypothesis.symbols

    first.reset()
    first.push_logits(logits)
    first.finish()
    # The same audio through the same decoder after a reset has to give the same
    # answer, which is only true if the reset really cleared the label history.
    assert first.hypothesis.symbols == expected


def test_a_reset_mid_utterance_does_not_leak_the_old_history() -> None:
    logits = _logits(frames=20, seed=22)
    joint, decoder = _decoder(size=3, cap=2)
    decoder.push_logits(logits[:12])
    decoder.reset()
    fresh, reference = _decoder(size=3, cap=2)
    for frame in range(12):
        reference.push_logits(logits[frame][None, :])
    reference.finish()
    for frame in range(12):
        decoder.push_logits(logits[frame][None, :])
    decoder.finish()
    assert decoder.hypothesis.symbols == reference.hypothesis.symbols


def test_the_stream_subpackage_re_exports_its_surface() -> None:
    from echoscribe import stream

    assert stream.StreamDecoder is StreamDecoder
    assert stream.ChunkSchedule is ChunkSchedule
    for name in ("StreamDecoder", "FrameBuffer", "AppendableJoint", "Hypothesis"):
        assert name in stream.__all__


def _quiet_logits(frames: int, symbols: int = 4) -> np.ndarray:
    """Logits whose softmax is almost entirely on the blank."""
    rows = np.full((frames, symbols), -6.0)
    rows[:, 0] = 6.0
    return rows


def test_an_endpointer_closes_the_stream() -> None:
    logits = np.vstack([_quiet_logits(4), _quiet_logits(20)])
    joint = AppendableJoint(4)
    endpointer = BlankRatioEndpointer(
        BlankRatioConfig(window_frames=6, ratio_threshold=0.9, min_speech_frames=0)
    )
    decoder = StreamDecoder(
        joint,
        StreamConfig(schedule=ChunkSchedule(frames_per_chunk=4), max_symbols_per_frame=2),
        endpointer,
    )
    events = []
    for frame in range(len(logits)):
        events.extend(decoder.push_logits(logits[frame][None, :]))
    kinds = [event.kind for event in events]
    assert "endpoint" in kinds
    assert decoder.is_closed


def test_no_endpoint_event_without_an_endpointer() -> None:
    logits = _quiet_logits(24)
    joint, decoder = _decoder(size=4, cap=2)
    events = []
    for frame in range(len(logits)):
        events.extend(decoder.push_logits(logits[frame][None, :]))
    assert all(event.kind == "partial" for event in events)


def test_a_reset_clears_the_endpointer_too() -> None:
    logits = _quiet_logits(24)
    joint = AppendableJoint(4)
    endpointer = BlankRatioEndpointer(
        BlankRatioConfig(window_frames=4, ratio_threshold=0.9, min_speech_frames=0)
    )
    decoder = StreamDecoder(
        joint, StreamConfig(schedule=ChunkSchedule(frames_per_chunk=4)), endpointer
    )
    decoder.push_logits(logits[:12])
    assert endpointer.has_fired
    decoder.reset()
    assert not endpointer.has_fired
    assert not decoder.is_closed


def test_an_endpoint_stops_further_partials() -> None:
    logits = _quiet_logits(40)
    joint = AppendableJoint(4)
    endpointer = BlankRatioEndpointer(
        BlankRatioConfig(window_frames=4, ratio_threshold=0.9, min_speech_frames=0)
    )
    decoder = StreamDecoder(
        joint, StreamConfig(schedule=ChunkSchedule(frames_per_chunk=2)), endpointer
    )
    events = []
    for frame in range(len(logits)):
        events.extend(decoder.push_logits(logits[frame][None, :]))
    endpoint_index = [event.kind for event in events].index("endpoint")
    assert all(event.kind != "partial" for event in events[endpoint_index:])
