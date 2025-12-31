"""Timing behaviour is pinned with an injected clock, never with sleeping."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.eval.bench import TimingStatistics, timing_statistics


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
