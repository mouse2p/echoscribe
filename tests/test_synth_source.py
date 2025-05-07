"""Sources are checked for period, continuity and spectral tilt."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.source import (
    GLOTTAL_OPEN_FRACTION,
    glottal_pulse,
    glottal_source,
    noise_source,
    phase_track,
)

RATE = 16000


def test_phase_advances_by_f0_over_the_sample_rate() -> None:
    phase = phase_track(np.full(64, 1000.0), RATE)
    expected = np.mod(np.arange(1, 65) * 1000.0 / RATE, 1.0)
    assert np.allclose(phase, expected, atol=1e-12)
    assert np.all(phase >= 0.0) and np.all(phase < 1.0)


def test_a_constant_f0_gives_a_linear_phase_between_wraps() -> None:
    phase = phase_track(np.full(40, 1000.0), RATE)
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    assert np.allclose(np.diff(unwrapped), 1000.0 / RATE, atol=1e-12)


def test_phase_track_honours_the_initial_phase() -> None:
    phase = phase_track(np.full(8, 100.0), RATE, initial_phase=0.5)
    assert phase[0] == pytest.approx(0.5 + 100.0 / RATE)


def test_phase_track_rejects_bad_inputs() -> None:
    with pytest.raises(ConfigError, match="initial_phase must lie in"):
        phase_track(np.full(4, 100.0), RATE, initial_phase=1.0)
    with pytest.raises(ConfigError, match="f0 must not be negative"):
        phase_track(np.array([-1.0, 2.0]), RATE)
    with pytest.raises(ConfigError, match="must be 1-D"):
        phase_track(np.zeros((2, 2)), RATE)


def test_the_pulse_starts_at_zero_and_peaks_at_the_open_fraction() -> None:
    assert glottal_pulse(np.array([0.0]))[0] == pytest.approx(0.0)
    assert glottal_pulse(np.array([GLOTTAL_OPEN_FRACTION]))[0] == pytest.approx(1.0)


def test_the_pulse_closes_back_to_zero() -> None:
    assert glottal_pulse(np.array([1.0 - 1e-12]))[0] == pytest.approx(0.0, abs=1e-9)


def test_the_pulse_is_continuous_across_the_open_close_boundary() -> None:
    around = glottal_pulse(np.array([GLOTTAL_OPEN_FRACTION - 1e-6, GLOTTAL_OPEN_FRACTION + 1e-6]))
    assert around[0] == pytest.approx(around[1], abs=1e-5)


def test_the_pulse_never_goes_negative_or_above_one() -> None:
    values = glottal_pulse(np.linspace(0.0, 1.0, 1001))
    assert values.min() >= 0.0
    assert values.max() <= 1.0 + 1e-12


def test_glottal_pulse_rejects_a_bad_open_fraction() -> None:
    with pytest.raises(ConfigError, match="open_fraction must lie in"):
        glottal_pulse(np.array([0.5]), open_fraction=0.0)


def test_a_constant_f0_source_repeats_its_period() -> None:
    source = glottal_source(RATE, RATE, 100.0)
    period = RATE // 100
    left = source[5 * period : RATE - 5 * period]
    right = source[6 * period : RATE - 4 * period]
    # Shifting by exactly one period must reproduce the waveform; only the phase
    # accumulator's rounding can break that.
    assert np.allclose(left, right, atol=1e-8)


def test_the_source_has_no_dc_component() -> None:
    source = glottal_source(4000, RATE, 110.0)
    assert abs(float(source.mean())) < 1e-12


def test_the_source_is_bounded() -> None:
    source = glottal_source(4000, RATE, 110.0)
    assert np.max(np.abs(source)) <= 1.0


def test_the_source_spectrum_falls_with_frequency() -> None:
    # One second at 100 Hz puts the fundamental in bin 100 and its harmonics at
    # every multiple of it, so a 1/f^2 glottal source is far down by 2 kHz.
    spectrum = np.abs(np.fft.rfft(glottal_source(RATE, RATE, 100.0)))
    fundamental = float(spectrum[100])
    harmonic = float(spectrum[200])
    high_band = float(spectrum[1800:2200].mean())
    assert fundamental > harmonic
    assert fundamental > 20.0 * high_band


def test_a_rising_f0_track_shortens_the_period() -> None:
    source = glottal_source(RATE, RATE, np.linspace(100.0, 200.0, RATE))

    def rising_crossings(block: np.ndarray) -> int:
        return int(np.count_nonzero(np.diff(np.sign(block)) > 0))

    assert rising_crossings(source[-RATE // 4 :]) > rising_crossings(source[: RATE // 4])


def test_glottal_source_validates_its_arguments() -> None:
    with pytest.raises(ConfigError, match="num_samples must be a non-negative integer"):
        glottal_source(-1, RATE, 100.0)
    with pytest.raises(ConfigError, match="were requested"):
        glottal_source(10, RATE, np.full(9, 100.0))
    with pytest.raises(ConfigError, match="strictly positive"):
        glottal_source(10, RATE, np.zeros(10))
    with pytest.raises(ConfigError, match="not a voice"):
        glottal_source(10, RATE, np.full(10, 2000.0))


def test_zero_samples_gives_an_empty_source() -> None:
    assert glottal_source(0, RATE, 100.0).shape == (0,)


def test_seeded_noise_replays_identically() -> None:
    left = noise_source(1024, np.random.default_rng(7))
    right = noise_source(1024, np.random.default_rng(7))
    assert np.array_equal(left, right)


def test_different_seeds_give_different_noise() -> None:
    left = noise_source(1024, np.random.default_rng(7))
    right = noise_source(1024, np.random.default_rng(8))
    assert not np.array_equal(left, right)


def test_noise_is_peak_normalised_and_close_to_zero_mean() -> None:
    noise = noise_source(20000, np.random.default_rng(3))
    assert float(np.max(np.abs(noise))) == pytest.approx(1.0)
    assert abs(float(noise.mean())) < 0.05
    # Peak-normalised Gaussian noise has a standard deviation well below one.
    assert 0.2 < float(noise.std()) < 0.4


def test_noise_consumes_the_generator_so_calls_do_not_repeat() -> None:
    rng = np.random.default_rng(2)
    assert not np.array_equal(noise_source(256, rng), noise_source(256, rng))


def test_noise_of_zero_samples_is_empty() -> None:
    assert noise_source(0, np.random.default_rng(1)).shape == (0,)


def test_noise_rejects_bad_arguments() -> None:
    with pytest.raises(ConfigError, match="num_samples must be a non-negative integer"):
        noise_source(-4, np.random.default_rng(1))
    with pytest.raises(ConfigError, match="tilt_hz must be positive"):
        noise_source(16, np.random.default_rng(1), tilt_hz=0.0)
