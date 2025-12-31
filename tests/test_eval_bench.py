"""Timing behaviour is pinned with an injected clock, never with sleeping."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.eval.bench import BenchResult, TimingStatistics, timing_statistics


def test_the_statistics_of_a_single_sample_are_that_sample() -> None:
    stats = timing_statistics([0.25])
    assert stats == TimingStatistics(minimum=0.25, median=0.25, mean=0.25)


def test_the_median_of_an_even_count_is_the_mean_of_the_middle_two() -> None:
    stats = timing_statistics([4.0, 1.0, 3.0, 2.0])
    assert stats.minimum == 1.0
    assert stats.median == 2.5
    assert stats.mean == 2.5


def test_statistics_are_bit_identical_whatever_order_samples_arrive_in() -> None:
    import random

    rng = random.Random(7)
    samples = [rng.random() for _ in range(9)]
    shuffled = list(samples)
    rng.shuffle(shuffled)
    assert samples != shuffled  # the shuffle did something
    assert timing_statistics(samples) == timing_statistics(shuffled)


def test_statistics_of_powers_of_two_are_exact() -> None:
    stats = timing_statistics([0.25, 0.5, 1.0, 2.0])
    assert stats.mean == 0.9375
    assert stats.median == 0.75


def test_an_empty_sample_list_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one sample"):
        timing_statistics([])


def test_the_rtf_of_a_two_to_one_slowdown() -> None:
    result = BenchResult(task="decode", audio_seconds=2.0, warmups=0, samples=(1.0, 0.5, 2.0))
    assert result.statistics.median == 1.0
    assert result.rtf == 0.5
    assert result.rtf_best == 0.25


def test_a_pipeline_faster_than_real_time_has_an_rtf_below_one() -> None:
    result = BenchResult(task="decode", audio_seconds=10.0, warmups=1, samples=(2.5, 2.6))
    assert result.rtf == 0.255


def test_the_repeats_are_the_sample_count_and_ignore_warmups() -> None:
    result = BenchResult(task="x", audio_seconds=1.0, warmups=3, samples=(0.1, 0.2))
    assert result.repeats == 2
    assert result.warmups == 3


def test_a_negative_sample_is_refused() -> None:
    with pytest.raises(ConfigError, match="must not be negative"):
        BenchResult(task="x", audio_seconds=1.0, warmups=0, samples=(-0.5,))


def test_a_result_without_samples_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one sample"):
        BenchResult(task="x", audio_seconds=1.0, warmups=0, samples=())


def test_zero_audio_duration_is_refused() -> None:
    with pytest.raises(ConfigError, match="audio_seconds must be positive"):
        BenchResult(task="x", audio_seconds=0.0, warmups=0, samples=(0.1,))


def test_a_result_without_a_task_name_is_refused() -> None:
    with pytest.raises(ConfigError, match="task name"):
        BenchResult(task="", audio_seconds=1.0, warmups=0, samples=(0.1,))
