"""SNR measurement is checked against mixes whose SNR was planted."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.noise.snr import snr_db
from echoscribe.synth.noise import generate_noise, mix_at_snr
from echoscribe.waveform import Audio


def tone_audio(seconds: float = 0.5, rate: int = 16000, frequency: float = 220.0) -> Audio:
    """A deterministic clean signal: one windowed tone."""
    count = int(seconds * rate)
    position = np.arange(count) / rate
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * position / seconds))
    return Audio((0.5 * envelope * np.sin(2 * np.pi * frequency * position)), rate)


@pytest.mark.parametrize("requested", [0.0, 5.0, 20.0])
def test_a_planted_mix_measures_its_requested_snr(requested: float) -> None:
    clean = tone_audio()
    rng = np.random.default_rng(3)
    noise = generate_noise("white", clean.num_frames, rng)
    mixed = mix_at_snr(clean, noise, requested)
    measured = snr_db(clean.samples, mixed.samples)
    assert abs(measured - requested) < 0.05


def test_identical_signals_measure_infinite_snr() -> None:
    clean = tone_audio()
    assert snr_db(clean.samples, clean.samples.copy()) == float("inf")


def test_a_silent_reference_is_refused() -> None:
    with pytest.raises(ConfigError, match="silent"):
        snr_db(np.zeros(100), np.ones(100))


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ConfigError, match="differently sized"):
        snr_db(np.ones(100), np.ones(99))


def test_an_empty_signal_is_refused() -> None:
    with pytest.raises(ConfigError, match="empty"):
        snr_db(np.zeros(0), np.zeros(0))


def test_more_noise_measures_lower() -> None:
    clean = tone_audio()
    rng = np.random.default_rng(4)
    noise = generate_noise("white", clean.num_frames, rng)
    quiet = mix_at_snr(clean, noise, 15.0)
    loud = mix_at_snr(clean, noise, 3.0)
    assert snr_db(clean.samples, loud.samples) < snr_db(clean.samples, quiet.samples)
