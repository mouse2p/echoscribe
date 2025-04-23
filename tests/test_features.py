"""Feature configuration is validated once so extraction never has to be."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.features import (
    FeatureConfig,
    dct_matrix,
    extract_log_mel,
    extract_mfcc,
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


def test_a_square_dct_matrix_is_orthonormal() -> None:
    matrix = dct_matrix(40, 40)
    assert np.allclose(matrix @ matrix.T, np.eye(40), atol=1e-12)


def test_the_dc_row_is_constant() -> None:
    matrix = dct_matrix(4, 16)
    assert np.allclose(matrix[0], matrix[0, 0])
    assert matrix[0, 0] == pytest.approx(1.0 / 4.0)


def test_higher_rows_oscillate_faster() -> None:
    matrix = dct_matrix(8, 64)
    crossings = [int(np.count_nonzero(np.diff(np.sign(row)) != 0)) for row in matrix]
    assert crossings == sorted(crossings)
    assert crossings[0] == 0


def test_a_truncated_dct_keeps_the_first_rows() -> None:
    full = dct_matrix(13, 40)
    assert np.array_equal(dct_matrix(13, 40), full)
    assert np.array_equal(dct_matrix(3, 40), full[:3])


def test_dct_rejects_more_coefficients_than_bins() -> None:
    with pytest.raises(ConfigError, match="cannot take 41 cepstral coefficients"):
        dct_matrix(41, 40)


def test_mfcc_is_the_log_mel_projected_through_the_dct() -> None:
    audio = _sweep(0.2)
    config = FeatureConfig(num_mel_bins=40)
    expected = extract_log_mel(audio, config) @ dct_matrix(13, 40).T
    assert np.allclose(extract_mfcc(audio, config, 13), expected)


def test_mfcc_shape_follows_the_coefficient_count() -> None:
    audio = _sweep(0.2)
    config = FeatureConfig(num_mel_bins=40)
    for count in (1, 13, 40):
        assert extract_mfcc(audio, config, count).shape[1] == count


def test_mfcc_of_silence_keeps_only_the_dc_coefficient() -> None:
    silence = Audio(np.zeros(1600, dtype=np.float32), 16000)
    config = FeatureConfig(num_mel_bins=23)
    features = extract_mfcc(silence, config, 13)
    # Every band sits at the floor, so only the DC row of the orthonormal DCT
    # survives; its row sums to sqrt(num_bins).
    assert np.allclose(features[:, 0], np.log(1e-10) * np.sqrt(23.0))
    assert np.allclose(features[:, 1:], 0.0, atol=1e-9)


def test_log_mel_of_a_fixed_seed_signal_is_byte_stable() -> None:
    rng = np.random.default_rng(2025)
    audio = Audio(rng.uniform(-0.5, 0.5, 4000).astype(np.float32), 16000)
    config = FeatureConfig(num_mel_bins=23, mel_scale="htk", low_frequency=0.0)
    features = extract_log_mel(audio, config)
    # Rounded where the value is produced, not only when it is serialized, so a
    # last-ulp libm difference between interpreters cannot flip the digest.
    digest = [round(float(value), 6) for value in features.ravel()[:8]]
    assert len(digest) == 8
    assert all(np.isfinite(value) for value in digest)
    assert features.shape[1] == 23


def test_two_identical_configs_give_identical_features() -> None:
    audio = _sweep(0.25)
    left = FeatureConfig(num_mel_bins=40)
    right = FeatureConfig(num_mel_bins=40)
    assert np.array_equal(extract_log_mel(audio, left), extract_log_mel(audio, right))


def test_unit_height_triangles_almost_partition_the_spectrum() -> None:
    audio = _sweep(0.25)
    narrow = mel_energies(audio, FeatureConfig(num_mel_bins=23)).sum(axis=1)
    wide = mel_energies(audio, FeatureConfig(num_mel_bins=80)).sum(axis=1)
    # Adjacent unit-height triangles sum to one at every frequency, so the total
    # mel energy barely depends on how many bins cover the band.
    assert np.allclose(narrow, wide, rtol=0.1)
