"""The transform pair is checked against signals whose spectra are known."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError, FeatureError
from echoscribe.noise.stft import StftConfig, stft


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


def test_the_shape_counts_frames_and_bins() -> None:
    config = StftConfig(frame_length=400, hop_length=160, fft_size=512)
    spectrum = stft(np.zeros(1000), config)
    assert spectrum.shape == (4, 257)


def test_a_constant_signal_concentrates_in_the_first_bin() -> None:
    config = StftConfig(frame_length=400, hop_length=200, fft_size=400, window="rectangular")
    spectrum = stft(np.ones(800), config)
    magnitudes = np.abs(spectrum)
    assert np.allclose(magnitudes[:, 0], 400.0)
    assert magnitudes[:, 1:].max() < 1e-8


def test_a_pure_tone_lands_on_its_exact_bin() -> None:
    # cos(2*pi*10*n/512) over exactly one 512-sample frame: all energy in bin 10
    n = np.arange(512)
    tone = np.cos(2 * np.pi * 10 * n / 512)
    config = StftConfig(frame_length=512, hop_length=256, fft_size=512, window="rectangular")
    spectrum = stft(tone, config)
    magnitudes = np.abs(spectrum[0])
    assert np.isclose(magnitudes[10], 256.0)
    assert magnitudes[[*range(0, 10), *range(11, 257)]].max() < 1e-8


def test_a_signal_shorter_than_one_frame_is_refused() -> None:
    with pytest.raises(ConfigError, match="fewer than one"):
        stft(np.zeros(10), StftConfig())


def test_a_matrix_instead_of_a_signal_is_refused() -> None:
    with pytest.raises(FeatureError, match="must be 1-D"):
        stft(np.zeros((4, 400)), StftConfig())


def test_the_transform_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    signal = rng.standard_normal(1000)
    assert np.array_equal(stft(signal), stft(signal))
