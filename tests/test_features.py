"""Feature configuration is validated once so extraction never has to be."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.features import FeatureConfig
from echoscribe.framing import FrameConfig


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
