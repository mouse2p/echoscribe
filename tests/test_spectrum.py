"""Spectral conventions are pinned on signals whose transform is known in closed form."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.spectrum import (
    SPECTRAL_FLOOR,
    bin_frequencies,
    log_power_spectrum,
    power_spectrum,
    rfft_bin_count,
)


def test_bin_count_is_half_the_size_plus_one() -> None:
    assert rfft_bin_count(512) == 257
    assert rfft_bin_count(400) == 201
    assert rfft_bin_count(1) == 1


@pytest.mark.parametrize("size", [0, -8])
def test_bin_count_rejects_a_non_positive_size(size: int) -> None:
    with pytest.raises(ConfigError, match="fft_size must be positive"):
        rfft_bin_count(size)


def test_a_constant_frame_concentrates_everything_at_dc() -> None:
    frames = np.ones((1, 16))
    power = power_spectrum(frames)
    assert power.shape == (1, 9)
    assert power[0, 0] == pytest.approx(16.0)
    assert np.allclose(power[0, 1:], 0.0, atol=1e-24)


def test_a_cosine_on_an_exact_bin_peaks_there() -> None:
    size = 64
    bin_index = 5
    t = np.arange(size, dtype=np.float64)
    frames = np.cos(2 * np.pi * bin_index * t / size)[None, :]
    power = power_spectrum(frames)
    assert int(np.argmax(power[0])) == bin_index
    assert power[0, bin_index] == pytest.approx(size / 4.0, rel=1e-12)


def test_zero_padding_does_not_change_the_normalised_power() -> None:
    size = 64
    bin_index = 3
    t = np.arange(size, dtype=np.float64)
    frames = np.cos(2 * np.pi * bin_index * t / size)[None, :]
    padded = power_spectrum(frames, fft_size=256)
    # Energy spreads over interpolated bins, but the total stays the same.
    assert padded.sum() == pytest.approx(power_spectrum(frames).sum(), rel=1e-9)


def test_a_smaller_fft_size_is_rejected() -> None:
    with pytest.raises(ConfigError, match="is smaller than the frame width"):
        power_spectrum(np.zeros((2, 400)), fft_size=256)


def test_power_spectrum_rejects_a_flat_vector() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        power_spectrum(np.zeros(64))


def test_power_spectrum_of_no_frames_has_no_rows() -> None:
    assert power_spectrum(np.zeros((0, 64))).shape == (0, 33)


def test_power_is_non_negative_and_additive_over_frames() -> None:
    rng = np.random.default_rng(3)
    frames = rng.standard_normal((6, 128))
    power = power_spectrum(frames)
    assert np.all(power >= 0.0)
    combined = power_spectrum(np.vstack([frames[:3], frames[3:]]))
    assert np.array_equal(combined, np.vstack([power[:3], power[3:]]))


def test_parseval_relates_the_power_sum_to_the_frame_energy() -> None:
    rng = np.random.default_rng(7)
    frames = rng.standard_normal((4, 128))
    power = power_spectrum(frames)
    # sum_k |X[k]|^2 over the full DFT equals n * sum_t x[t]^2; the one-sided
    # rfft doubles every bin except DC and Nyquist.
    doubled = power.copy()
    doubled[:, 1:-1] *= 2.0
    assert doubled.sum(axis=1) == pytest.approx((frames**2).sum(axis=1), rel=1e-12)


def test_power_of_a_unit_cosine_is_independent_of_padding() -> None:
    size = 128
    t = np.arange(size, dtype=np.float64)
    frames = np.cos(2 * np.pi * 7 * t / size)[None, :]
    totals = [power_spectrum(frames, fft_size=size * factor).sum() for factor in (1, 2, 4)]
    assert totals[0] == pytest.approx(totals[1], rel=1e-9)
    assert totals[1] == pytest.approx(totals[2], rel=1e-9)


def test_scaling_a_frame_scales_its_power_quadratically() -> None:
    rng = np.random.default_rng(11)
    frames = rng.standard_normal((2, 96))
    assert np.allclose(power_spectrum(frames * 3.0), power_spectrum(frames) * 9.0)


def test_output_dtype_is_float64_for_float32_input() -> None:
    assert power_spectrum(np.zeros((2, 32), dtype=np.float32)).dtype == np.float64


def test_bin_frequencies_run_from_dc_to_nyquist() -> None:
    frequencies = bin_frequencies(512, 16000)
    assert frequencies.shape == (257,)
    assert frequencies[0] == 0.0
    assert frequencies[-1] == 8000.0


def test_bin_spacing_is_the_sample_rate_over_the_fft_size() -> None:
    frequencies = bin_frequencies(400, 16000)
    assert np.allclose(np.diff(frequencies), 40.0)


def test_bin_frequencies_scale_with_the_sample_rate() -> None:
    assert np.allclose(bin_frequencies(256, 8000), bin_frequencies(256, 16000) / 2.0)


def test_bin_frequencies_reject_bad_arguments() -> None:
    with pytest.raises(ConfigError, match="fft_size must be positive"):
        bin_frequencies(0, 16000)
    with pytest.raises(ConfigError, match="sample_rate must be positive"):
        bin_frequencies(256, 0)


def test_the_log_spectrum_of_a_silent_frame_is_the_floor() -> None:
    power = log_power_spectrum(np.zeros((1, 64)))
    assert np.all(np.isfinite(power))
    assert np.allclose(power, np.log(SPECTRAL_FLOOR))


def test_the_log_spectrum_matches_the_log_of_the_power_spectrum() -> None:
    rng = np.random.default_rng(13)
    frames = rng.uniform(0.0, 1.0, size=(3, 64))
    expected = np.log(np.maximum(power_spectrum(frames), SPECTRAL_FLOOR))
    assert np.allclose(log_power_spectrum(frames), expected)


def test_the_log_spectrum_floor_can_be_raised() -> None:
    frames = np.zeros((1, 32))
    assert np.allclose(log_power_spectrum(frames, floor=1e-6), np.log(1e-6))
