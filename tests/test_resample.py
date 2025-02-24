"""Rate conversion is checked on tones, where an error shows up as a pitch shift."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.resample import (
    FILTER_WINDOWS,
    RESAMPLE_METHODS,
    ResampleConfig,
    _cached_filter_bank,
    design_lowpass_fir,
    linear_resample,
    polyphase_filter_bank,
    polyphase_resample,
    rational_factors,
    resample,
    window_vector,
)
from echoscribe.waveform import Audio


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


def test_resample_config_defaults_are_valid() -> None:
    config = ResampleConfig()
    assert config.target_rate == 16000
    assert config.method == "polyphase"
    assert config.taps_per_phase == 16
    assert config.window == "hamming"
    assert config.cutoff == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_rate": 0},
        {"target_rate": -8000},
        {"target_rate": 16000.0},
        {"method": "cubic"},
        {"taps_per_phase": 0},
        {"window": "kaiser"},
        {"cutoff": 0.0},
        {"cutoff": 1.0},
    ],
)
def test_resample_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(ConfigError):
        ResampleConfig(**kwargs)


def test_resample_config_is_frozen() -> None:
    config = ResampleConfig()
    with pytest.raises(AttributeError):
        config.target_rate = 8000  # type: ignore[misc]


def test_every_declared_method_and_window_is_accepted() -> None:
    assert set(RESAMPLE_METHODS) == {"polyphase", "linear"}
    for window in FILTER_WINDOWS:
        assert ResampleConfig(window=window).window == window
    for method in RESAMPLE_METHODS:
        assert ResampleConfig(method=method).method == method


def test_resample_returns_audio_at_the_target_rate() -> None:
    source = Audio(_tone(440.0, 44100, 0.1).astype(np.float32), 44100)
    converted = resample(source, ResampleConfig(target_rate=16000))
    assert converted.sample_rate == 16000
    assert converted.num_frames == pytest.approx(source.num_frames * 16000 / 44100, abs=2)


def test_resample_duration_is_preserved_by_both_methods() -> None:
    source = Audio(_tone(300.0, 22050, 0.25).astype(np.float32), 22050)
    for method in RESAMPLE_METHODS:
        converted = resample(source, ResampleConfig(target_rate=16000, method=method))
        assert converted.duration == pytest.approx(source.duration, abs=2e-3)


def test_resample_defaults_to_polyphase_at_16k() -> None:
    source = Audio(_tone(440.0, 8000, 0.1).astype(np.float32), 8000)
    assert np.array_equal(
        resample(source).samples,
        resample(source, ResampleConfig(target_rate=16000)).samples,
    )


def test_resample_does_not_mutate_the_input() -> None:
    source = Audio(_tone(440.0, 16000, 0.1).astype(np.float32), 16000)
    before = source.samples.copy()
    resample(source, ResampleConfig(target_rate=8000))
    assert np.array_equal(source.samples, before)


def test_both_methods_preserve_a_low_frequency_tone() -> None:
    # 120 Hz sits far below either Nyquist, so the missing anti-alias filter in
    # the linear path cannot matter; amplitude and pitch must both survive.
    source = Audio(_tone(120.0, 16000, 0.5).astype(np.float32), 16000)
    for method in RESAMPLE_METHODS:
        result = resample(source, ResampleConfig(target_rate=8000, method=method))
        assert _peak_frequency(result.samples, 8000) == pytest.approx(120.0, abs=1.0)
        assert np.max(np.abs(result.samples[64:-64])) == pytest.approx(1.0, abs=0.05)


def test_the_cached_bank_equals_a_fresh_design() -> None:
    for up, down in [(1, 2), (2, 1), (160, 441)]:
        cached = _cached_filter_bank(up, down, 16, "hamming", 0.9)
        assert np.array_equal(cached, polyphase_filter_bank(up, down, 16, "hamming", 0.9))


def test_the_cached_bank_is_read_only() -> None:
    bank = _cached_filter_bank(1, 2, 8, "hann", 0.8)
    with pytest.raises(ValueError):
        bank[0, 0] = 0.0


def test_repeated_resampling_of_a_corpus_stays_deterministic() -> None:
    source = _tone(440.0, 22050, 0.2).astype(np.float32)
    first = polyphase_resample(source, 22050, 16000)
    for _ in range(5):
        assert np.array_equal(polyphase_resample(source, 22050, 16000), first)


def test_caching_does_not_leak_across_design_settings() -> None:
    assert not np.array_equal(
        _cached_filter_bank(1, 2, 16, "hamming", 0.9),
        _cached_filter_bank(1, 2, 16, "hann", 0.9),
    )


def test_polyphase_resample_still_matches_the_uncached_bank() -> None:
    samples = _tone(700.0, 32000, 0.1).astype(np.float32)
    up, down = 1, 2
    phases = polyphase_filter_bank(up, down, 12, "blackman", 0.85)
    result = polyphase_resample(samples, 32000, 16000, taps_per_phase=12, window="blackman")
    assert np.array_equal(_cached_filter_bank(1, 2, 12, "blackman", 0.85), phases)
    assert _peak_frequency(result, 16000) == pytest.approx(700.0, abs=2.0)


def test_resample_still_exposes_the_shared_window_names() -> None:
    from echoscribe import resample as module
    from echoscribe import windows

    assert module.FILTER_WINDOWS is windows.FILTER_WINDOWS
    assert module.window_vector is windows.window_vector


def test_the_fir_still_uses_the_shared_coefficients() -> None:
    taps = design_lowpass_fir(21, 0.4, "blackman")
    offsets = np.arange(21, dtype=np.float64) - 10.0
    manual = 0.4 * np.sinc(0.4 * offsets) * window_vector("blackman", 21)
    assert np.allclose(taps, manual / manual.sum(), atol=1e-15)
