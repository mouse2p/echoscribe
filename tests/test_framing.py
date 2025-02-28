"""Framing arithmetic is integer arithmetic, so it is pinned exactly."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.framing import FrameConfig


def test_defaults_match_the_standard_25ms_10ms_recipe() -> None:
    config = FrameConfig()
    assert config.sample_rate == 16000
    assert config.frame_length == 400
    assert config.frame_shift == 160
    assert config.fft_size == 512
    assert config.window == "povey"
    assert config.preemphasis == 0.97
    assert config.snip_edges


@pytest.mark.parametrize(
    ("rate", "length_ms", "shift_ms", "expected"),
    [
        (16000, 25.0, 10.0, (400, 160)),
        (8000, 25.0, 10.0, (200, 80)),
        (44100, 25.0, 10.0, (1103, 441)),
        (16000, 32.0, 16.0, (512, 256)),
        (16000, 1.0, 1.0, (16, 16)),
    ],
)
def test_sample_counts_are_rounded_to_the_nearest_sample(
    rate: int, length_ms: float, shift_ms: float, expected: tuple[int, int]
) -> None:
    config = FrameConfig(sample_rate=rate, frame_length_ms=length_ms, frame_shift_ms=shift_ms)
    assert (config.frame_length, config.frame_shift) == expected


def test_half_a_sample_rounds_up_rather_than_to_even() -> None:
    # 25 ms at 44.1 kHz is exactly 1102.5 samples.
    assert FrameConfig(sample_rate=44100).frame_length == 1103
    assert FrameConfig(sample_rate=44100).frame_shift == 441


def test_a_sub_sample_frame_still_yields_at_least_one_sample() -> None:
    config = FrameConfig(sample_rate=16000, frame_length_ms=0.001, frame_shift_ms=0.001)
    assert config.frame_length == 1
    assert config.frame_shift == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": -16000},
        {"sample_rate": 16000.0},
        {"frame_length_ms": 0.0},
        {"frame_length_ms": -25.0},
        {"frame_shift_ms": 0.0},
        {"frame_length_ms": 5.0, "frame_shift_ms": 10.0},
        {"window": "kaiser"},
        {"preemphasis": -0.1},
        {"preemphasis": 1.5},
    ],
)
def test_invalid_configurations_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        FrameConfig(**kwargs)


def test_equal_length_and_shift_is_allowed() -> None:
    config = FrameConfig(frame_length_ms=10.0, frame_shift_ms=10.0)
    assert config.frame_length == config.frame_shift


def test_preemphasis_boundary_values_are_allowed() -> None:
    assert FrameConfig(preemphasis=0.0).preemphasis == 0.0
    assert FrameConfig(preemphasis=1.0).preemphasis == 1.0


def test_config_is_frozen() -> None:
    config = FrameConfig()
    with pytest.raises(AttributeError):
        config.window = "hann"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("rate", "length_ms", "expected"),
    [
        (16000, 25.0, 512),
        (8000, 25.0, 256),
        (16000, 32.0, 512),
        (44100, 25.0, 2048),
        (16000, 16.0, 256),
    ],
)
def test_fft_size_is_the_next_power_of_two(rate: int, length_ms: float, expected: int) -> None:
    config = FrameConfig(sample_rate=rate, frame_length_ms=length_ms)
    assert config.fft_size == expected
    assert config.fft_size >= config.frame_length


def test_fft_size_can_be_left_exact() -> None:
    config = FrameConfig(frame_length_ms=30.0, round_to_power_of_two=False)
    assert config.fft_size == config.frame_length == 480


def test_fft_size_of_a_single_sample_frame_is_one() -> None:
    config = FrameConfig(sample_rate=16000, frame_length_ms=0.001, frame_shift_ms=0.001)
    assert config.frame_length == 1
    assert config.fft_size == 1


def test_fft_size_is_stable_for_an_exact_power_of_two_frame() -> None:
    config = FrameConfig(sample_rate=16000, frame_length_ms=32.0)
    assert config.frame_length == 512
    assert config.fft_size == 512


def test_frame_length_ms_at_least_frame_shift_ms_is_enforced() -> None:
    with pytest.raises(ConfigError, match="must be at least frame_shift_ms"):
        FrameConfig(frame_length_ms=9.0, frame_shift_ms=10.0)
