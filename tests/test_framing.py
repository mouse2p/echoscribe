"""Framing arithmetic is integer arithmetic, so it is pinned exactly."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.framing import FrameConfig, frame_count, frame_starts, preemphasis


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


SNIPPED = FrameConfig(snip_edges=True)
PADDED = FrameConfig(snip_edges=False)


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [(0, 0), (1, 0), (399, 0), (400, 1), (559, 1), (560, 2), (1600, 8), (1601, 8)],
)
def test_snipped_frame_count(num_samples: int, expected: int) -> None:
    assert frame_count(num_samples, SNIPPED) == expected


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [(0, 0), (1, 1), (160, 1), (161, 2), (1600, 10), (1601, 11)],
)
def test_padded_frame_count(num_samples: int, expected: int) -> None:
    assert frame_count(num_samples, PADDED) == expected


def test_padding_never_yields_fewer_frames_than_snipping() -> None:
    for num_samples in range(0, 2000, 37):
        assert frame_count(num_samples, PADDED) >= frame_count(num_samples, SNIPPED)


def test_frame_starts_are_a_regular_grid_when_snipping() -> None:
    starts = frame_starts(1600, SNIPPED)
    assert starts.tolist() == [0, 160, 320, 480, 640, 800, 960, 1120]
    assert np.all(np.diff(starts) == 160)


def test_frame_starts_are_padded_coordinates_when_not_snipping() -> None:
    starts = frame_starts(1600, PADDED)
    # Padded coordinates start at zero; the half-frame left pad is implied.
    assert starts[0] == 0
    assert len(starts) == frame_count(1600, PADDED)
    assert np.all(np.diff(starts) == 160)


def test_the_last_snipped_frame_fits_inside_the_signal() -> None:
    starts = frame_starts(1601, SNIPPED)
    assert int(starts[-1]) + SNIPPED.frame_length <= 1601


def test_frame_count_rejects_a_negative_length() -> None:
    with pytest.raises(ValueError, match="num_samples must not be negative"):
        frame_count(-1, SNIPPED)


def test_frame_starts_are_integers_of_the_index_dtype() -> None:
    assert frame_starts(1600, SNIPPED).dtype == np.intp
    assert frame_starts(1600, PADDED).dtype == np.intp


def test_preemphasis_matches_the_definition_sample_by_sample() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.array([0.5, 1.5, 2.0, 2.5])
    assert np.allclose(preemphasis(x, 0.5), expected.astype(np.float32))


def test_the_first_sample_policy_switches_both_ways() -> None:
    x = np.array([0.8, 0.2, -0.4])
    assert preemphasis(x, 0.97, first="replicate")[0] == pytest.approx(0.8 * 0.03, abs=1e-7)
    assert preemphasis(x, 0.97, first="zero")[0] == pytest.approx(0.8, abs=1e-7)


def test_a_zero_coefficient_is_the_identity() -> None:
    x = np.linspace(-1.0, 1.0, 64)
    assert np.allclose(preemphasis(x, 0.0), x.astype(np.float32))


def test_a_unit_coefficient_is_a_first_difference() -> None:
    x = np.array([1.0, 3.0, 6.0])
    assert np.allclose(preemphasis(x, 1.0), [0.0, 2.0, 3.0])


def test_preemphasis_lifts_high_frequencies_relative_to_low() -> None:
    t = np.arange(4096, dtype=np.float64) / 16000.0

    def band_energy(frequency: float, coefficient: float) -> float:
        tone = np.sin(2 * np.pi * frequency * t)
        filtered = preemphasis(tone, coefficient).astype(np.float64)
        return float(np.mean(filtered[64:-64] ** 2))

    # |H(w)|^2 = 1 - 2c*cos(w) + c^2 grows with frequency, so the high/low energy
    # ratio must rise sharply once pre-emphasis is switched on.
    ratio_off = band_energy(3000.0, 0.0) / band_energy(200.0, 0.0)
    ratio_on = band_energy(3000.0, 0.97) / band_energy(200.0, 0.97)
    assert ratio_off == pytest.approx(1.0, abs=1e-3)
    assert ratio_on > 50.0 * ratio_off


def test_preemphasis_of_an_empty_buffer_is_empty() -> None:
    assert preemphasis(np.zeros(0)).shape == (0,)


def test_preemphasis_rejects_bad_arguments() -> None:
    with pytest.raises(ConfigError, match="coefficient must lie in"):
        preemphasis(np.zeros(4), 1.5)
    with pytest.raises(ConfigError, match="first must be one of"):
        preemphasis(np.zeros(4), 0.97, first="wrap")
    with pytest.raises(ConfigError, match="needs 1-D samples"):
        preemphasis(np.zeros((2, 2)))


def test_preemphasis_is_invertible_by_an_all_pole_filter() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.0, 1.0, 200)
    y = preemphasis(x, 0.97).astype(np.float64)
    recovered = np.empty_like(x)
    recovered[0] = y[0] / 0.03
    for index in range(1, len(x)):
        recovered[index] = y[index] + 0.97 * recovered[index - 1]
    assert np.allclose(recovered, x, atol=1e-4)
