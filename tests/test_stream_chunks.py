"""The latency budget is arithmetic, so it is pinned to the millisecond."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.stream.chunks import ChunkPlan, ChunkSchedule


def test_the_default_schedule_is_ten_frames_with_four_of_lookahead() -> None:
    schedule = ChunkSchedule()
    assert schedule.frames_per_chunk == 10
    assert schedule.right_context_frames == 4
    assert schedule.chunk_ms == pytest.approx(100.0)
    assert schedule.right_context_ms == pytest.approx(40.0)


def test_the_latency_formula_matches_its_documentation() -> None:
    schedule = ChunkSchedule(frames_per_chunk=10, right_context_frames=4)
    expected = (10 + 4 - 1) * 10.0 + 25.0
    assert schedule.algorithmic_latency_ms == pytest.approx(expected)


def test_a_larger_chunk_costs_proportionally_more_latency() -> None:
    small = ChunkSchedule(frames_per_chunk=4).algorithmic_latency_ms
    large = ChunkSchedule(frames_per_chunk=16).algorithmic_latency_ms
    assert large - small == pytest.approx(12 * 10.0)


def test_right_context_is_the_only_part_that_needs_the_future() -> None:
    zero = ChunkSchedule(right_context_frames=0)
    assert zero.algorithmic_latency_ms == pytest.approx(9 * 10.0 + 25.0)


def test_sample_counts_follow_the_sample_rate() -> None:
    schedule = ChunkSchedule(sample_rate=16000)
    assert schedule.frame_shift_samples == 160
    assert schedule.frame_length_samples == 400


def test_a_chunk_count_is_a_ceiling() -> None:
    schedule = ChunkSchedule(frames_per_chunk=10)
    assert schedule.chunk_count(0) == 0
    assert schedule.chunk_count(1) == 1
    assert schedule.chunk_count(10) == 1
    assert schedule.chunk_count(11) == 2


def test_a_plan_covers_every_frame_exactly_once() -> None:
    schedule = ChunkSchedule(frames_per_chunk=4)
    plans = schedule.plan(10)
    assert [plan.num_frames for plan in plans] == [4, 4, 2]
    cursor = 0
    for index, plan in enumerate(plans):
        assert plan.index == index
        assert plan.first_frame == cursor
        cursor = plan.end_frame
    assert cursor == 10


def test_a_plan_of_no_frames_is_empty() -> None:
    assert ChunkSchedule().plan(0) == ()


def test_availability_grows_with_the_frame_index() -> None:
    schedule = ChunkSchedule()
    times = [schedule.available_at_ms(index) for index in range(1, 6)]
    assert times == sorted(times)
    assert times[0] == pytest.approx(4 * 10.0 + 25.0)


def test_samples_required_agrees_with_the_millisecond_budget() -> None:
    schedule = ChunkSchedule(sample_rate=16000)
    for index in (1, 5, 20):
        expected = schedule.available_at_ms(index) / 1000.0 * 16000
        assert schedule.samples_required(index) == pytest.approx(expected, abs=1.0)


def test_a_chunk_plan_rejects_an_inverted_range() -> None:
    with pytest.raises(ConfigError, match="invalid frame range"):
        ChunkPlan(0, 5, 2, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"frames_per_chunk": 0},
        {"right_context_frames": -1},
        {"frame_shift_ms": 0.0},
        {"frame_length_ms": 0.0},
        {"frame_length_ms": 5.0, "frame_shift_ms": 10.0},
    ],
)
def test_invalid_schedules_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        ChunkSchedule(**kwargs)


def test_a_negative_frame_count_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must not be negative"):
        ChunkSchedule().chunk_count(-1)


def test_the_schedule_is_immutable() -> None:
    with pytest.raises(AttributeError):
        ChunkSchedule().frames_per_chunk = 4  # type: ignore[misc]


def test_a_plan_is_reproducible() -> None:
    schedule = ChunkSchedule(frames_per_chunk=7)
    assert schedule.plan(31) == schedule.plan(31)


def test_every_chunk_after_the_first_is_full_except_possibly_the_last() -> None:
    schedule = ChunkSchedule(frames_per_chunk=6)
    plans = schedule.plan(40)
    assert all(plan.num_frames == 6 for plan in plans[:-1])
    assert plans[-1].num_frames == 40 % 6


def test_availability_never_precedes_the_frame_itself() -> None:
    schedule = ChunkSchedule()
    for index in range(1, 30):
        # A frame that starts at (index - 1) * shift cannot be decoded before its
        # own last sample has arrived.
        assert schedule.available_at_ms(index) >= (index - 1) * schedule.frame_shift_ms


def test_a_zero_lookahead_schedule_is_causal() -> None:
    schedule = ChunkSchedule(right_context_frames=0)
    for index in (1, 10, 100):
        assert schedule.available_at_ms(index) <= index * schedule.frame_shift_ms + 25.0


def test_the_last_chunk_wants_more_context_than_the_utterance_has() -> None:
    schedule = ChunkSchedule(frames_per_chunk=5, right_context_frames=3)
    plans = schedule.plan(12)
    # The plan still reports the context it would like, so a caller can see the
    # tail is decoded with a truncated lookahead.
    assert plans[-1].available_at_ms == schedule.available_at_ms(12)
    assert plans[-1].end_frame == 12
