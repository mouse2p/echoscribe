"""The resonator is checked through its impulse response, which is its whole story."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.formant import apply_resonator, resonator_coefficients

RATE = 16000


def _impulse_response(centre: float, bandwidth: float, length: int = 2048) -> np.ndarray:
    delta = np.zeros(length)
    delta[0] = 1.0
    return apply_resonator(delta, centre, bandwidth, RATE)


def test_the_coefficients_give_unity_dc_gain() -> None:
    a0, b1, b2 = resonator_coefficients(500.0, 100.0, RATE)
    assert a0 / (1.0 - b1 - b2) == pytest.approx(1.0, abs=1e-12)


def test_the_pole_radius_follows_the_bandwidth() -> None:
    for centre in (400.0, 1500.0):
        _, b1, b2 = resonator_coefficients(centre, 100.0, RATE)
        radius = math.sqrt(-b2) if b2 < 0 else 0.0
        assert radius == pytest.approx(np.exp(-np.pi * 100.0 / RATE), rel=1e-12)


def test_the_impulse_response_peaks_at_the_centre_frequency() -> None:
    for centre in (300.0, 900.0, 2400.0):
        response = _impulse_response(centre, 120.0)
        spectrum = np.abs(np.fft.rfft(response))
        peak = float(np.fft.rfftfreq(len(response), 1 / RATE)[int(np.argmax(spectrum))])
        assert peak == pytest.approx(centre, abs=RATE / len(response) + 1.0)


def test_a_narrower_bandwidth_gives_a_sharper_peak() -> None:
    def peak_to_median(bandwidth: float) -> float:
        spectrum = np.abs(np.fft.rfft(_impulse_response(1000.0, bandwidth)))
        peak = spectrum[1000 * len(spectrum) // (RATE // 2)]
        return float(peak) / max(float(np.median(spectrum)), 1e-12)

    assert peak_to_median(60.0) > peak_to_median(400.0)


def test_the_response_decays_faster_with_a_wider_bandwidth() -> None:
    narrow = _impulse_response(800.0, 60.0, 4096)
    wide = _impulse_response(800.0, 600.0, 4096)
    tail = slice(2048, 4096)
    assert np.max(np.abs(narrow[tail])) > np.max(np.abs(wide[tail]))


def test_the_filter_is_linear() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(512)
    y = rng.standard_normal(512)
    combined = apply_resonator(2.0 * x + 3.0 * y, 700.0, 90.0, RATE)
    separate = 2.0 * apply_resonator(x, 700.0, 90.0, RATE) + 3.0 * apply_resonator(
        y, 700.0, 90.0, RATE
    )
    assert np.allclose(combined, separate, atol=1e-10)


def test_the_filter_starts_from_a_zero_state() -> None:
    assert _impulse_response(500.0, 100.0, 64)[0] != 0.0
    assert np.all(apply_resonator(np.zeros(64), 500.0, 100.0, RATE) == 0.0)


def test_the_filter_is_stable() -> None:
    response = _impulse_response(2500.0, 200.0, 8192)
    assert np.all(np.isfinite(response))
    assert np.max(np.abs(response[-1024:])) < np.max(np.abs(response[:1024]))


def test_coefficients_reject_impossible_formants() -> None:
    with pytest.raises(ConfigError, match="at or above Nyquist"):
        resonator_coefficients(9000.0, 100.0, RATE)
    with pytest.raises(ConfigError, match="centre_hz must be positive"):
        resonator_coefficients(0.0, 100.0, RATE)
    with pytest.raises(ConfigError, match="bandwidth_hz must be positive"):
        resonator_coefficients(500.0, 0.0, RATE)


def test_the_resonator_rejects_a_matrix() -> None:
    with pytest.raises(ConfigError, match="must be 1-D"):
        apply_resonator(np.zeros((4, 2)), 500.0, 100.0, RATE)
