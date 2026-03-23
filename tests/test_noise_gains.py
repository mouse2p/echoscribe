"""Gains are checked on hand-computed bins before they ever see audio."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.noise.gains import spectral_subtraction_gain


def test_half_noise_leaves_half_the_bin() -> None:
    gain = spectral_subtraction_gain(np.array([[4.0]]), np.array([[2.0]]), floor=0.0)
    assert gain.tolist() == [[0.5]]


def test_oversubtraction_can_drive_a_bin_to_the_floor() -> None:
    gain = spectral_subtraction_gain(
        np.array([[4.0]]), np.array([[2.0]]), oversubtraction=2.0, floor=0.1
    )
    assert gain.tolist() == [[0.1]]


def test_a_silent_bin_with_a_silent_estimate_keeps_gain_one() -> None:
    gain = spectral_subtraction_gain(np.array([[0.0]]), np.array([[0.0]]))
    assert gain.tolist() == [[1.0]]


def test_a_silent_bin_under_loud_noise_drops_to_the_floor() -> None:
    gain = spectral_subtraction_gain(np.array([[0.0]]), np.array([[1.0]]), floor=0.05)
    assert gain.tolist() == [[0.05]]


def test_a_per_bin_profile_broadcasts_across_frames() -> None:
    signal = np.array([[4.0, 1.0], [4.0, 1.0]])
    gain = spectral_subtraction_gain(signal, np.array([2.0, 0.5]), floor=0.0)
    assert gain.shape == (2, 2)
    assert gain.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_gains_never_exceed_one() -> None:
    signal = np.array([[1e12, 0.0, 3.0]])
    noise = np.array([[1.0, 0.0, 1.0]])
    gain = spectral_subtraction_gain(signal, noise, floor=0.0)
    assert np.all(gain <= 1.0)


def test_a_negative_power_is_refused() -> None:
    with pytest.raises(ConfigError, match="non-negative"):
        spectral_subtraction_gain(np.array([[-1.0]]), np.array([[1.0]]))


def test_a_profile_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(ConfigError, match="must hold 3 values"):
        spectral_subtraction_gain(np.zeros((2, 3)), np.zeros(4))


def test_a_shape_mismatch_is_refused() -> None:
    with pytest.raises(ConfigError, match="shape"):
        spectral_subtraction_gain(np.zeros((2, 3)), np.zeros((2, 4)))


def test_a_zero_oversubtraction_is_refused() -> None:
    with pytest.raises(ConfigError, match="positive"):
        spectral_subtraction_gain(np.array([[1.0]]), np.array([[1.0]]), oversubtraction=0.0)


def test_a_floor_of_one_is_refused() -> None:
    with pytest.raises(ConfigError, match="below 1.0"):
        spectral_subtraction_gain(np.array([[1.0]]), np.array([[1.0]]), floor=1.0)
