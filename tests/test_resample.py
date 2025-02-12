"""Rate conversion is checked on tones, where an error shows up as a pitch shift."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.resample import (
    FILTER_WINDOWS,
    design_lowpass_fir,
    linear_resample,
    polyphase_filter_bank,
    polyphase_resample,
    rational_factors,
    window_vector,
)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (16000, 8000, (1, 2)),
        (8000, 16000, (2, 1)),
        (44100, 16000, (160, 441)),
        (16000, 16000, (1, 1)),
        (48000, 16000, (1, 3)),
        (22050, 44100, (2, 1)),
    ],
)
def test_rational_factors_reduces_to_a_coprime_pair(source, target, expected) -> None:
    assert rational_factors(source, target) == expected


def test_rational_factors_reproduces_the_ratio() -> None:
    for source, target in [(44100, 16000), (16000, 22050), (8000, 48000)]:
        up, down = rational_factors(source, target)
        assert up / down == pytest.approx(target / source, rel=1e-12)


@pytest.mark.parametrize("bad", [0, -16000, 16000.0, "16000", None, True])
def test_rational_factors_rejects_a_bad_rate(bad) -> None:
    with pytest.raises(ConfigError):
        rational_factors(16000, bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        rational_factors(bad, 16000)  # type: ignore[arg-type]


@pytest.mark.parametrize("window", list(FILTER_WINDOWS))
def test_lowpass_fir_has_unit_dc_gain(window: str) -> None:
    assert design_lowpass_fir(31, 0.4, window).sum() == pytest.approx(1.0, abs=1e-12)


def test_lowpass_fir_is_symmetric_and_therefore_linear_phase() -> None:
    taps = design_lowpass_fir(33, 0.45, "hamming")
    assert np.allclose(taps, taps[::-1], atol=1e-15)


def test_lowpass_fir_attenuates_above_the_cutoff() -> None:
    taps = design_lowpass_fir(65, 0.25, "hamming")
    rate = 1000.0
    t = np.arange(4096, dtype=np.float64) / rate
    for frequency, limit in [(50.0, 1.05), (300.0, 0.2), (450.0, 0.02)]:
        tone = np.sin(2 * np.pi * frequency * t)
        response = np.convolve(tone, taps, mode="valid")
        middle = response[len(response) // 4 : 3 * len(response) // 4]
        assert np.max(np.abs(middle)) < limit, frequency


def test_longer_filters_transition_more_sharply() -> None:
    t = np.arange(4096, dtype=np.float64) / 1000.0
    tone = np.sin(2 * np.pi * 300.0 * t)
    leakage = []
    for taps_count in (17, 65, 257):
        taps = design_lowpass_fir(taps_count, 0.1, "hamming")
        response = np.convolve(tone, taps, mode="valid")
        leakage.append(float(np.max(np.abs(response[len(response) // 4 :]))))
    assert leakage[0] > leakage[1] > leakage[2]


@pytest.mark.parametrize("bad_cutoff", [0.0, 1.0, -0.5, 1.5])
def test_lowpass_fir_rejects_an_out_of_range_cutoff(bad_cutoff: float) -> None:
    with pytest.raises(ConfigError, match="cutoff must lie in"):
        design_lowpass_fir(31, bad_cutoff)


@pytest.mark.parametrize("bad_taps", [0, -3, 4.5, "16"])
def test_lowpass_fir_rejects_a_bad_tap_count(bad_taps) -> None:
    with pytest.raises(ConfigError, match="num_taps must be a positive integer"):
        design_lowpass_fir(bad_taps, 0.4)


def test_lowpass_fir_accepts_a_single_tap() -> None:
    assert np.allclose(design_lowpass_fir(1, 0.4), [1.0])


def test_window_vector_endpoints_distinguish_hann_from_hamming() -> None:
    assert window_vector("hann", 8)[0] == pytest.approx(0.0)
    assert window_vector("hamming", 8)[0] == pytest.approx(0.08, abs=1e-3)
    assert window_vector("rectangular", 8).tolist() == [1.0] * 8
    assert len(window_vector("blackman", 8)) == 8


def test_window_vector_rejects_an_unknown_kind() -> None:
    with pytest.raises(ConfigError, match="window must be one of"):
        window_vector("kaiser", 8)


def test_window_vector_rejects_a_non_positive_length() -> None:
    with pytest.raises(ConfigError, match="length must be positive"):
        window_vector("hamming", 0)


def test_linear_downsample_halves_the_length() -> None:
    assert linear_resample(np.arange(100, dtype=np.float64), 16000, 8000).shape == (50,)


def test_linear_upsample_doubles_the_length() -> None:
    assert linear_resample(np.arange(100, dtype=np.float64), 8000, 16000).shape == (200,)


def test_linear_resample_keeps_both_endpoints() -> None:
    samples = np.array([-0.75, 0.0, 0.25, 1.0], dtype=np.float64)
    for target in (4000, 8000, 32000):
        result = linear_resample(samples, 8000, target)
        assert result[0] == pytest.approx(-0.75)
        assert result[-1] == pytest.approx(1.0)


def test_linear_resample_at_the_same_rate_is_the_identity() -> None:
    samples = np.linspace(-1.0, 1.0, 37)
    assert np.allclose(linear_resample(samples, 16000, 16000), samples.astype(np.float32))


def test_linear_resample_of_an_empty_buffer_is_empty() -> None:
    assert linear_resample(np.zeros(0), 16000, 8000).shape == (0,)


def test_linear_resample_of_a_single_sample_repeats_it() -> None:
    result = linear_resample(np.array([0.25]), 8000, 16000)
    assert result.shape == (2,)
    assert np.all(result == pytest.approx(0.25))


def test_linear_resample_is_monotonic_on_a_ramp() -> None:
    result = linear_resample(np.arange(64, dtype=np.float64), 16000, 11025)
    assert np.all(np.diff(result) >= 0.0)


def test_linear_resample_returns_float32() -> None:
    assert linear_resample(np.arange(8, dtype=np.float64), 8000, 16000).dtype == np.float32


def test_linear_resample_rejects_a_matrix() -> None:
    with pytest.raises(ConfigError, match="needs 1-D samples"):
        linear_resample(np.zeros((4, 2)), 8000, 16000)


def test_linear_resample_interpolates_between_neighbours_when_upsampling() -> None:
    result = linear_resample(np.array([0.0, 1.0], dtype=np.float64), 8000, 16000)
    # Two samples at twice the rate give four outputs spread over [0, 1].
    assert result.shape == (4,)
    assert list(result) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0], abs=1e-7)


def _tone(frequency: float, rate: int, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return np.sin(2.0 * np.pi * frequency * t)


def _peak_frequency(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    bins = np.fft.rfftfreq(len(samples), 1.0 / rate)
    return float(bins[int(np.argmax(spectrum))])


def test_polyphase_downsample_preserves_a_tone_frequency() -> None:
    converted = polyphase_resample(_tone(440.0, 16000), 16000, 8000)
    assert _peak_frequency(converted, 8000) == pytest.approx(440.0, abs=2.0)


def test_polyphase_upsample_preserves_a_tone_frequency() -> None:
    converted = polyphase_resample(_tone(440.0, 8000), 8000, 16000)
    assert _peak_frequency(converted, 16000) == pytest.approx(440.0, abs=2.0)


def test_polyphase_output_length_is_the_ceiling_of_the_ratio() -> None:
    for count in (1, 7, 100, 1001):
        assert polyphase_resample(np.zeros(count), 16000, 8000).shape == (-(-count // 2),)


def test_polyphase_at_the_same_rate_applies_only_the_prototype_filter() -> None:
    samples = _tone(300.0, 16000, 0.2)
    taps = polyphase_filter_bank(1, 1, 16, "hamming", 0.9)[0]
    expected = np.convolve(samples, taps, mode="full")[: len(samples)]
    result = polyphase_resample(samples, 16000, 16000, taps_per_phase=16)
    assert np.allclose(result, expected.astype(np.float32), atol=1e-5)


def test_polyphase_of_an_empty_buffer_is_empty() -> None:
    assert polyphase_resample(np.zeros(0), 16000, 8000).shape == (0,)


def test_polyphase_suppresses_aliasing_that_linear_lets_through() -> None:
    # 5 kHz at 16 kHz folds to 3 kHz once the target Nyquist is 4 kHz.
    tone = _tone(5000.0, 16000, 0.25)
    filtered = polyphase_resample(tone, 16000, 8000, taps_per_phase=32)
    raw = linear_resample(tone, 16000, 8000)
    middle = slice(64, len(raw) - 64)
    assert np.max(np.abs(filtered[middle])) < 0.25 * np.max(np.abs(raw[middle]))


def test_polyphase_pass_band_keeps_a_tone_amplitude() -> None:
    converted = polyphase_resample(_tone(440.0, 16000, 0.5), 16000, 8000)
    assert np.max(np.abs(converted[64:-64])) == pytest.approx(1.0, abs=0.02)


def test_polyphase_rejects_a_matrix() -> None:
    with pytest.raises(ConfigError, match="needs 1-D samples"):
        polyphase_resample(np.zeros((4, 2)), 8000, 16000)


def test_filter_bank_has_one_row_per_phase() -> None:
    assert polyphase_filter_bank(3, 7, taps_per_phase=5).shape == (3, 5)
    assert polyphase_filter_bank(1, 2, taps_per_phase=5).shape == (1, 5)


def test_filter_bank_rejects_a_bad_tap_count() -> None:
    with pytest.raises(ConfigError, match="taps_per_phase must be a positive integer"):
        polyphase_filter_bank(1, 2, taps_per_phase=0)


def test_filter_bank_rows_sum_to_about_one() -> None:
    # Each phase carries the DC gain, so a constant input stays constant.
    phases = polyphase_filter_bank(2, 1, taps_per_phase=16)
    assert np.allclose(phases.sum(axis=1), 1.0, atol=2e-3)
