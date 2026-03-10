"""The transform pair is checked against signals whose spectra are known."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.noise.stft import StftConfig


def test_the_default_config_describes_a_standard_pair() -> None:
    config = StftConfig()
    assert (config.frame_length, config.hop_length, config.fft_size) == (400, 160, 512)
    assert config.window == "hann"
    assert config.bin_count == 257


def test_the_bin_count_follows_the_fft_size() -> None:
    assert StftConfig(frame_length=256, fft_size=256).bin_count == 129
    assert StftConfig(fft_size=400, frame_length=400).bin_count == 201


def test_an_fft_smaller_than_the_frame_is_refused() -> None:
    with pytest.raises(ConfigError, match="must cover frame_length"):
        StftConfig(frame_length=512, fft_size=256)


def test_a_hop_longer_than_the_frame_is_refused() -> None:
    with pytest.raises(ConfigError, match="must not exceed"):
        StftConfig(frame_length=200, hop_length=400)


def test_a_zero_hop_is_refused() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        StftConfig(hop_length=0)


def test_a_window_the_filters_do_not_know_is_refused() -> None:
    with pytest.raises(ConfigError, match="window"):
        StftConfig(window="kaiser")
