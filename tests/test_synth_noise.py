"""Noise generators are checked through their spectra and their reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.noise import NOISE_KINDS, pink_noise, white_noise

RATE = 16000


def test_the_declared_kinds_are_the_three_generators() -> None:
    assert NOISE_KINDS == ("white", "pink", "modulated")


def test_white_noise_is_peak_normalised_and_zero_mean() -> None:
    noise = white_noise(20000, np.random.default_rng(1))
    assert float(np.max(np.abs(noise))) == pytest.approx(1.0)
    assert abs(float(noise.mean())) < 0.02


def test_pink_noise_is_peak_normalised() -> None:
    assert float(np.max(np.abs(pink_noise(8192, np.random.default_rng(2))))) == pytest.approx(1.0)


def test_pink_noise_has_no_dc_component() -> None:
    noise = pink_noise(4096, np.random.default_rng(3))
    assert abs(np.fft.rfft(noise)[0]) < 1e-9


def test_pink_noise_puts_more_energy_low_than_white_does() -> None:
    def low_over_high(noise: np.ndarray) -> float:
        # Mean power per bin, so the two bands are comparable despite having very
        # different widths.
        power = np.abs(np.fft.rfft(noise)) ** 2
        return float(power[1:64].mean() / power[256:1024].mean())

    white = low_over_high(white_noise(8192, np.random.default_rng(4)))
    pink = low_over_high(pink_noise(8192, np.random.default_rng(4)))
    assert white == pytest.approx(1.0, rel=0.2)
    assert pink > 10.0 * white


def test_generators_replay_from_a_seed() -> None:
    assert np.array_equal(
        white_noise(512, np.random.default_rng(7)), white_noise(512, np.random.default_rng(7))
    )
    assert np.array_equal(
        pink_noise(512, np.random.default_rng(7)), pink_noise(512, np.random.default_rng(7))
    )


def test_different_seeds_differ() -> None:
    assert not np.array_equal(
        white_noise(512, np.random.default_rng(7)), white_noise(512, np.random.default_rng(8))
    )


@pytest.mark.parametrize("generator", [white_noise, pink_noise])
def test_zero_length_is_allowed(generator) -> None:
    assert generator(0, np.random.default_rng(1)).shape == (0,)


@pytest.mark.parametrize("generator", [white_noise, pink_noise])
def test_a_negative_length_is_rejected(generator) -> None:
    with pytest.raises(ConfigError, match="non-negative integer"):
        generator(-1, np.random.default_rng(1))


def test_odd_lengths_work_for_pink_noise() -> None:
    assert pink_noise(1001, np.random.default_rng(5)).shape == (1001,)
