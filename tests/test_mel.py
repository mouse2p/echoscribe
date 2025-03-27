"""The two mel scales are pinned at frequencies where their value is exact."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.mel import (
    HTK_REFERENCE_HZ,
    HTK_REFERENCE_MEL,
    MEL_SCALES,
    SLANEY_MIN_LOG_HZ,
    SLANEY_MIN_LOG_MEL,
    hz_to_mel,
    mel_to_hz,
)


def test_both_scales_map_zero_to_zero() -> None:
    for scale in MEL_SCALES:
        assert hz_to_mel(0.0, scale) == pytest.approx(0.0, abs=1e-12)
        assert mel_to_hz(0.0, scale) == pytest.approx(0.0, abs=1e-12)


def test_htk_maps_seven_times_the_reference_to_exactly_one_decade() -> None:
    # 1 + 6300/700 = 10, so the mel value is the reference constant itself.
    assert hz_to_mel(9.0 * HTK_REFERENCE_HZ, "htk") == pytest.approx(HTK_REFERENCE_MEL, rel=1e-12)


def test_slaney_is_linear_below_one_kilohertz() -> None:
    assert hz_to_mel(SLANEY_MIN_LOG_HZ, "slaney") == pytest.approx(SLANEY_MIN_LOG_MEL)
    assert hz_to_mel(500.0, "slaney") == pytest.approx(7.5)
    assert hz_to_mel(250.0, "slaney") == pytest.approx(3.75)


def test_slaney_log_anchor_is_6400_hz_at_mel_42() -> None:
    assert hz_to_mel(6400.0, "slaney") == pytest.approx(42.0, rel=1e-12)


def test_slaney_is_continuous_across_its_break_point() -> None:
    below = hz_to_mel(SLANEY_MIN_LOG_HZ - 1e-6, "slaney")
    above = hz_to_mel(SLANEY_MIN_LOG_HZ + 1e-6, "slaney")
    assert below == pytest.approx(above, abs=1e-6)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_conversions_are_monotonic(scale: str) -> None:
    frequencies = np.linspace(0.0, 8000.0, 512)
    mels = hz_to_mel(frequencies, scale)
    assert np.all(np.diff(mels) > 0.0)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_round_trip_recovers_the_frequency(scale: str) -> None:
    frequencies = np.array([0.0, 20.0, 300.0, 1000.0, 4000.0, 7999.0])
    assert np.allclose(mel_to_hz(hz_to_mel(frequencies, scale), scale), frequencies, atol=1e-9)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_a_scalar_stays_a_scalar_and_an_array_stays_an_array(scale: str) -> None:
    assert isinstance(hz_to_mel(1000.0, scale), float)
    assert isinstance(hz_to_mel(np.array([1000.0]), scale), np.ndarray)
    assert isinstance(mel_to_hz(15.0, scale), float)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_negative_inputs_are_rejected(scale: str) -> None:
    with pytest.raises(ConfigError, match="non-negative"):
        hz_to_mel(-1.0, scale)
    with pytest.raises(ConfigError, match="non-negative"):
        mel_to_hz(-1.0, scale)


def test_an_unknown_scale_is_rejected() -> None:
    with pytest.raises(ConfigError, match="scale must be one of"):
        hz_to_mel(1000.0, "erb")
    with pytest.raises(ConfigError, match="scale must be one of"):
        mel_to_hz(15.0, "erb")


def test_the_two_scales_disagree_above_the_slaney_break_point() -> None:
    htk = hz_to_mel(4000.0, "htk")
    slaney = hz_to_mel(4000.0, "slaney")
    assert htk != pytest.approx(slaney)
    assert slaney < htk
