"""Noise generators are checked through their spectra and their reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.noise import (
    NOISE_KINDS,
    mix_at_snr,
    pink_noise,
    signal_to_noise_db,
    white_noise,
)
from echoscribe.waveform import Audio

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


def _tone(seconds: float = 0.5) -> Audio:
    t = np.arange(int(RATE * seconds), dtype=np.float64) / RATE
    return Audio((0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), RATE)


@pytest.mark.parametrize("target", [30.0, 20.0, 10.0, 5.0, 0.0, -5.0])
def test_mixing_reaches_the_requested_snr(target: float) -> None:
    clean = _tone()
    noise = white_noise(clean.num_frames, np.random.default_rng(1))
    mixed = mix_at_snr(clean, noise, target)
    assert signal_to_noise_db(clean.samples, mixed.samples) == pytest.approx(target, abs=0.05)


def test_mixing_preserves_the_clean_level() -> None:
    clean = _tone()
    mixed = mix_at_snr(clean, white_noise(clean.num_frames, np.random.default_rng(2)), 20.0)
    # The signal part is untouched, so the mixed level is only slightly higher.
    assert float(np.mean(mixed.samples.astype(np.float64) ** 2)) > float(
        np.mean(clean.samples.astype(np.float64) ** 2)
    )


def test_mixing_keeps_the_sample_rate_and_length() -> None:
    clean = _tone()
    mixed = mix_at_snr(clean, white_noise(64, np.random.default_rng(3)), 10.0)
    assert mixed.sample_rate == RATE
    assert mixed.num_frames == clean.num_frames


def test_a_short_noise_track_is_tiled() -> None:
    clean = _tone(0.2)
    mixed = mix_at_snr(clean, white_noise(100, np.random.default_rng(4)), 10.0)
    assert signal_to_noise_db(clean.samples, mixed.samples) == pytest.approx(10.0, abs=0.05)


def test_an_identical_signal_has_infinite_snr() -> None:
    clean = _tone()
    assert signal_to_noise_db(clean.samples, clean.samples) == float("inf")


def test_a_silent_reference_against_pure_noise_has_minus_infinite_snr() -> None:
    tone = _tone()
    silence = np.zeros(tone.num_frames, dtype=np.float32)
    assert signal_to_noise_db(silence, tone.samples) == float("-inf")
    # The mirror case: an untouched reference against itself is a perfect signal.
    assert signal_to_noise_db(tone.samples, tone.samples) == float("inf")


def test_snr_rejects_mismatched_shapes_and_empty_input() -> None:
    with pytest.raises(ConfigError, match="cannot compare shapes"):
        signal_to_noise_db(np.zeros(4), np.zeros(5))
    with pytest.raises(ConfigError, match="empty signal is undefined"):
        signal_to_noise_db(np.zeros(0), np.zeros(0))


def test_mixing_rejects_a_silent_noise_track() -> None:
    clean = _tone()
    with pytest.raises(ConfigError, match="the noise track is silent"):
        mix_at_snr(clean, np.zeros(64), 10.0)


def test_mixing_rejects_an_empty_signal() -> None:
    with pytest.raises(ConfigError, match="cannot add noise to an empty signal"):
        mix_at_snr(Audio(np.zeros(0, dtype=np.float32), RATE), np.ones(8), 10.0)


def test_a_higher_snr_leaves_the_signal_cleaner() -> None:
    clean = _tone()
    noise = white_noise(clean.num_frames, np.random.default_rng(6))
    quiet = mix_at_snr(clean, noise, 25.0)
    loud = mix_at_snr(clean, noise, 5.0)
    assert np.max(np.abs(quiet.samples - clean.samples)) < np.max(
        np.abs(loud.samples - clean.samples)
    )
