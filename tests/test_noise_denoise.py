"""The denoiser is measured on planted pairs from the synthetic laboratory."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.noise.denoise import estimate_noise_profile


def test_the_profile_is_the_lead_in_mean() -> None:
    power = np.arange(60, dtype=np.float64).reshape(12, 5)
    profile = estimate_noise_profile(power, 4)
    # column j holds j, 5+j, 10+j, 15+j over the first four frames
    assert profile.tolist() == [7.5, 8.5, 9.5, 10.5, 11.5]


def test_a_single_frame_profile_is_that_frame() -> None:
    power = np.array([[1.0, 2.0], [9.0, 9.0]])
    assert estimate_noise_profile(power, 1).tolist() == [1.0, 2.0]


def test_asking_for_every_frame_is_allowed() -> None:
    power = np.ones((3, 2))
    assert estimate_noise_profile(power, 3).tolist() == [1.0, 1.0]


def test_more_frames_than_exist_are_refused() -> None:
    with pytest.raises(ConfigError, match="asks for"):
        estimate_noise_profile(np.ones((3, 2)), 4)


def test_zero_frames_are_refused() -> None:
    with pytest.raises(ConfigError, match="integer >= 1"):
        estimate_noise_profile(np.ones((3, 2)), 0)


def test_a_flat_spectrum_is_refused() -> None:
    with pytest.raises(ConfigError, match="matrix"):
        estimate_noise_profile(np.ones(5), 2)
