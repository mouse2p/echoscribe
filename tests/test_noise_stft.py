"""The transform pair is checked against signals whose spectra are known."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError, FeatureError
from echoscribe.noise.stft import StftConfig, istft, power_spectrogram, stft


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


def test_a_seeded_signal_survives_a_hamming_round_trip_exactly() -> None:
    rng = np.random.default_rng(11)
    signal = rng.standard_normal(1000)
    config = StftConfig(frame_length=400, hop_length=200, fft_size=512, window="hamming")
    rebuilt = istft(stft(signal, config), config)
    assert rebuilt.shape == (1000,)
    assert np.allclose(rebuilt, signal, atol=1e-8)


def test_a_hann_round_trip_loses_only_the_zero_weight_edges() -> None:
    # hann vanishes at its endpoints, so the first and last sample are covered
    # by a frame whose weight there is zero; everything in between is exact.
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(800)
    config = StftConfig(frame_length=400, hop_length=100, fft_size=512, window="hann")
    rebuilt = istft(stft(signal, config), config)
    assert rebuilt[0] == 0.0
    assert np.allclose(rebuilt[1:-1], signal[1:-1], atol=1e-8)


def test_the_round_trip_holds_across_several_hops() -> None:
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(1200)
    for hop in (100, 160, 200, 400):
        config = StftConfig(frame_length=400, hop_length=hop, fft_size=512, window="hamming")
        rebuilt = istft(stft(signal, config), config, length=1200)
        assert np.allclose(rebuilt, signal, atol=1e-8), hop


def test_a_spectrogram_with_the_wrong_bin_count_is_refused() -> None:
    config = StftConfig(fft_size=512)
    with pytest.raises(ConfigError, match="bins"):
        istft(np.zeros((3, 129), dtype=complex), config)


def test_a_spectrogram_without_frames_is_refused() -> None:
    config = StftConfig(fft_size=512)
    with pytest.raises(ConfigError, match="at least one frame"):
        istft(np.zeros((0, 257), dtype=complex), config)


def test_the_length_argument_truncates_the_rebuild() -> None:
    rng = np.random.default_rng(13)
    signal = rng.standard_normal(1000)
    config = StftConfig(frame_length=400, hop_length=200, fft_size=512, window="hamming")
    rebuilt = istft(stft(signal, config), config, length=997)
    assert rebuilt.shape == (997,)
    assert np.allclose(rebuilt, signal[:997], atol=1e-8)


def test_a_length_the_frames_cannot_cover_is_refused() -> None:
    config = StftConfig(frame_length=400, hop_length=160, fft_size=512)
    spectrum = stft(np.zeros(1000), config)
    with pytest.raises(ConfigError, match="exceeds"):
        istft(spectrum, config, length=1001)


def test_a_zero_length_rebuilds_nothing() -> None:
    config = StftConfig(frame_length=400, hop_length=160, fft_size=512)
    spectrum = stft(np.zeros(1000), config)
    assert istft(spectrum, config, length=0).shape == (0,)


def test_a_negative_length_is_refused() -> None:
    config = StftConfig(frame_length=400, hop_length=160, fft_size=512)
    spectrum = stft(np.zeros(1000), config)
    with pytest.raises(ConfigError, match="non-negative integer"):
        istft(spectrum, config, length=-1)


def test_a_hand_computed_bin() -> None:
    power = power_spectrogram(np.array([[3 + 4j, 0 + 0j, -1 + 0j]]))
    assert power.tolist() == [[25.0, 0.0, 1.0]]


def test_a_real_spectrogram_squares_cleanly() -> None:
    power = power_spectrogram(np.array([[-2.0, 0.5]]))
    assert power.tolist() == [[4.0, 0.25]]


def test_power_of_a_transformed_signal_is_nonnegative() -> None:
    rng = np.random.default_rng(17)
    spectrum = stft(rng.standard_normal(800), StftConfig(frame_length=400, hop_length=200))
    power = power_spectrogram(spectrum)
    assert power.shape == spectrum.shape
    assert np.all(power >= 0.0)


def test_a_rank_one_spectrum_is_refused() -> None:
    with pytest.raises(FeatureError, match="must be 2-D"):
        power_spectrogram(np.zeros(5))


def test_the_noise_subpackage_re_exports_the_transform_pair() -> None:
    from echoscribe import noise

    assert noise.stft is stft
    assert noise.istft is istft
    assert noise.StftConfig is StftConfig
    for name in ("stft", "istft", "power_spectrogram", "StftConfig"):
        assert name in noise.__all__
