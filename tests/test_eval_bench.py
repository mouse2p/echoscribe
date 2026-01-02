"""Timing behaviour is pinned with an injected clock, never with sleeping."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.eval.bench import BenchResult, TimingStatistics, measure_rtf, timing_statistics


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


class FakeClock:
    """Hands out prepared readings in order and complains when they run out."""

    def __init__(self, readings: list[float]) -> None:
        self.readings = list(readings)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if not self.readings:
            raise AssertionError("the clock was read more often than prepared")
        return self.readings.pop(0)


def test_warmup_runs_are_executed_but_not_timed() -> None:
    calls: list[str] = []

    def run() -> None:
        calls.append("run")

    clock = FakeClock([10.0, 11.0, 20.0, 21.5])
    result = measure_rtf("decode", run, audio_seconds=3.0, warmups=2, repeats=2, clock=clock)
    assert len(calls) == 4
    assert result.samples == (1.0, 1.5)
    assert result.warmups == 2
    assert clock.calls == 4


def test_the_measured_runs_drive_the_rtf() -> None:
    clock = FakeClock([0.0, 1.0, 0.0, 0.5, 0.0, 2.0])
    result = measure_rtf("x", lambda: None, audio_seconds=2.0, warmups=0, repeats=3, clock=clock)
    assert result.samples == (1.0, 0.5, 2.0)
    assert result.rtf == 0.5


def test_a_failing_run_propagates_and_leaves_no_result() -> None:
    def explode() -> None:
        raise RuntimeError("the task failed")

    with pytest.raises(RuntimeError, match="the task failed"):
        measure_rtf("x", explode, audio_seconds=1.0, warmups=0, repeats=1)


def test_measuring_against_the_real_clock_is_well_formed() -> None:
    result = measure_rtf("noop", lambda: None, audio_seconds=1.0, warmups=1, repeats=2)
    assert result.repeats == 2
    assert all(sample >= 0.0 for sample in result.samples)
    assert result.rtf >= 0.0


def test_a_bench_result_survives_a_dict_roundtrip() -> None:
    result = BenchResult(task="decode", audio_seconds=2.5, warmups=1, samples=(0.5, 0.75))
    assert BenchResult.from_dict(result.to_dict()) == result


def test_the_serialized_form_is_pinned() -> None:
    result = BenchResult(task="decode", audio_seconds=2.5, warmups=1, samples=(0.5, 0.75))
    assert result.to_dict() == {
        "schema": 1,
        "task": "decode",
        "audio_seconds": 2.5,
        "warmups": 1,
        "samples": [0.5, 0.75],
    }


def test_a_foreign_schema_version_is_refused() -> None:
    from echoscribe.errors import SchemaError

    payload = BenchResult("decode", 2.5, 1, (0.5,)).to_dict()
    payload["schema"] = 99
    with pytest.raises(SchemaError, match="not supported"):
        BenchResult.from_dict(payload)


def test_a_missing_field_is_refused() -> None:
    from echoscribe.errors import SchemaError

    payload = BenchResult("decode", 2.5, 1, (0.5,)).to_dict()
    del payload["samples"]
    with pytest.raises(SchemaError, match="missing 'samples'"):
        BenchResult.from_dict(payload)


def test_the_roundtrip_preserves_the_computed_rtf() -> None:
    result = BenchResult(task="x", audio_seconds=4.0, warmups=0, samples=(1.0, 1.5, 3.0))
    restored = BenchResult.from_dict(result.to_dict())
    assert restored.rtf == result.rtf
    assert restored.rtf_best == result.rtf_best


def test_zero_repeats_is_refused_before_running() -> None:
    calls: list[str] = []
    with pytest.raises(ConfigError, match="repeats must be at least 1"):
        measure_rtf("x", lambda: calls.append("run"), audio_seconds=1.0, repeats=0)
    assert calls == []


def test_negative_warmups_are_refused_before_running() -> None:
    calls: list[str] = []
    with pytest.raises(ConfigError, match="warmups must not be negative"):
        measure_rtf("x", lambda: calls.append("run"), audio_seconds=1.0, warmups=-1)
    assert calls == []


def test_a_nonpositive_audio_duration_is_refused_before_running() -> None:
    calls: list[str] = []
    with pytest.raises(ConfigError, match="audio_seconds must be positive"):
        measure_rtf("x", lambda: calls.append("run"), audio_seconds=-2.0)
    assert calls == []
