"""Feature configuration is validated once so extraction never has to be."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.features import (
    FeatureConfig,
    extract_log_mel,
    filterbank_matrix,
    mel_energies,
)
from echoscribe.framing import FrameConfig, frame_count
from echoscribe.mel import mel_centre_frequencies
from echoscribe.waveform import Audio


def test_defaults_are_an_80_bin_slaney_bank_at_16k() -> None:
    config = FeatureConfig()
    assert config.num_mel_bins == 80
    assert config.sample_rate == 16000
    assert config.fft_size == 512
    assert config.mel_scale == "slaney"
    assert config.low_frequency == 20.0
    assert config.resolved_high_frequency == 8000.0
    assert not config.normalize_filterbank


def test_the_sample_rate_follows_the_frame_config() -> None:
    config = FeatureConfig(frame=FrameConfig(sample_rate=8000))
    assert config.sample_rate == 8000
    assert config.fft_size == 256
    assert config.resolved_high_frequency == 4000.0


def test_an_explicit_high_frequency_is_kept() -> None:
    config = FeatureConfig(high_frequency=7500.0)
    assert config.resolved_high_frequency == 7500.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_mel_bins": 0},
        {"num_mel_bins": -4},
        {"num_mel_bins": 23.0},
        {"mel_scale": "erb"},
        {"log_floor": 0.0},
        {"low_frequency": -1.0},
        {"high_frequency": 0.0},
        {"high_frequency": 9000.0},
        {"low_frequency": 5000.0, "high_frequency": 4000.0},
    ],
)
def test_invalid_configurations_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        FeatureConfig(**kwargs)


def test_config_is_frozen_and_hashable() -> None:
    config = FeatureConfig(num_mel_bins=23)
    with pytest.raises(AttributeError):
        config.num_mel_bins = 40  # type: ignore[misc]
    assert hash(config) == hash(FeatureConfig(num_mel_bins=23))


def test_the_fft_size_never_shrinks_below_the_frame() -> None:
    config = FeatureConfig(frame=FrameConfig(frame_length_ms=30.0))
    assert config.fft_size >= config.frame.frame_length


def _sweep(seconds: float = 0.5, rate: int = 16000) -> Audio:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    tone = np.sin(2 * np.pi * (200.0 + 900.0 * t) * t)
    return Audio(tone.astype(np.float32), rate)


def test_the_filterbank_matrix_matches_the_config() -> None:
    config = FeatureConfig(num_mel_bins=23)
    assert filterbank_matrix(config).shape == (23, config.fft_size // 2 + 1)


def test_log_mel_shape_follows_the_frame_count_and_bin_count() -> None:
    audio = _sweep()
    config = FeatureConfig(num_mel_bins=40)
    features = extract_log_mel(audio, config)
    assert features.shape[1] == 40
    assert features.shape[0] == frame_count(audio.num_frames, config.frame)


def test_log_mel_is_finite_and_below_the_floor_log_for_silence() -> None:
    silence = Audio(np.zeros(1600, dtype=np.float32), 16000)
    features = extract_log_mel(silence, FeatureConfig(num_mel_bins=23))
    assert np.all(np.isfinite(features))
    assert np.allclose(features, np.log(1e-10))


def test_log_mel_equals_the_log_of_the_linear_mel_energies() -> None:
    audio = _sweep(0.2)
    config = FeatureConfig(num_mel_bins=23, log_floor=1e-30)
    expected = np.log(np.maximum(mel_energies(audio, config), 1e-30))
    assert np.allclose(extract_log_mel(audio, config), expected)


def test_log_mel_is_deterministic_for_the_same_input() -> None:
    audio = _sweep(0.3)
    config = FeatureConfig(num_mel_bins=23)
    first = extract_log_mel(audio, config)
    assert np.array_equal(extract_log_mel(audio, config), first)


def test_mel_energies_are_non_negative() -> None:
    assert np.all(mel_energies(_sweep(0.2), FeatureConfig(num_mel_bins=23)) >= 0.0)


def test_a_tone_puts_its_energy_in_the_band_that_contains_it() -> None:
    rate = 16000
    t = np.arange(rate, dtype=np.float64) / rate
    tone = Audio(np.sin(2 * np.pi * 440.0 * t).astype(np.float32), rate)
    config = FeatureConfig(num_mel_bins=23, low_frequency=0.0, mel_scale="htk")
    features = extract_log_mel(tone, config)
    peak = int(np.argmax(features.mean(axis=0)))
    centres = mel_centre_frequencies(23, rate, 0.0, None, "htk")
    assert centres[peak - 1] < 440.0 < centres[peak + 1]
