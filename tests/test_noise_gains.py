"""Gains are checked on hand-computed bins before they ever see audio."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.noise.gains import spectral_subtraction_gain, wiener_gain


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


def test_the_first_frame_is_the_instantaneous_estimate() -> None:
    signal = np.array([[10.0]])
    noise = np.array([[1.0]])
    gain = wiener_gain(signal, noise, floor=0.0)
    # posterior 10 -> instantaneous 9 -> gain 9/10
    assert np.isclose(gain[0, 0], 0.9)


def test_a_stationary_pair_settles_below_its_first_frame() -> None:
    signal = np.full((4, 1), 10.0)
    noise = np.full((4, 1), 1.0)
    gain = wiener_gain(signal, noise, floor=0.0, smoothing=0.9)
    assert gain[1, 0] < gain[0, 0]
    assert gain[2, 0] < gain[1, 0]
    assert gain[3, 0] < gain[2, 0]
    assert np.all(gain > 0.85)


def test_the_hand_computed_second_frame() -> None:
    signal = np.full((2, 1), 10.0)
    noise = np.full((2, 1), 1.0)
    gain = wiener_gain(signal, noise, floor=0.0, smoothing=0.9)
    # frame 0: gain 0.9, clean estimate 0.81 * 10 = 8.1
    # frame 1: directed 0.9 * 8.1 = 7.29, prior 7.29 + 0.1 * 9 = 8.19
    assert np.isclose(gain[1, 0], 8.19 / 9.19)


def test_pure_noise_bins_sit_at_the_floor() -> None:
    signal = np.full((3, 2), 1.0)
    noise = np.full((3, 2), 1.0)
    gain = wiener_gain(signal, noise, floor=0.05)
    assert np.all(gain == 0.05)


def test_wiener_and_subtraction_agree_on_one_frame() -> None:
    signal = np.array([[6.0, 0.5]])
    noise = np.array([[2.0, 1.0]])
    directed = wiener_gain(signal, noise, floor=0.0)
    subtracted = spectral_subtraction_gain(signal, noise, floor=0.0)
    assert np.allclose(directed, subtracted)


def test_a_bad_smoothing_value_is_refused() -> None:
    with pytest.raises(ConfigError, match="smoothing"):
        wiener_gain(np.array([[1.0]]), np.array([[1.0]]), smoothing=1.5)
