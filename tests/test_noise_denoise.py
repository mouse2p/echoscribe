"""The denoiser is measured on planted pairs from the synthetic laboratory."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.noise.denoise import (
    DenoiseConfig,
    denoise,
    estimate_noise_profile,
)
from echoscribe.noise.snr import snr_improvement_db
from echoscribe.synth.noise import generate_noise, mix_at_snr
from echoscribe.waveform import Audio


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


def lead_in_tone(seconds: float = 1.0, rate: int = 16000, lead: float = 0.2) -> Audio:
    """A clean signal with a silent lead-in: tones under a slow envelope."""
    count = int(seconds * rate)
    lead_samples = int(lead * rate)
    position = np.arange(count - lead_samples) / rate
    body = 0.6 * np.sin(2 * np.pi * 220.0 * position) + 0.3 * np.sin(2 * np.pi * 640.0 * position)
    envelope = 0.4 + 0.6 * 0.5 * (1.0 - np.cos(2 * np.pi * position / position[-1] * 3))
    samples = np.zeros(count)
    samples[lead_samples:] = body * envelope
    return Audio(samples, rate)


def planted_pair(seed: int = 5, snr: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    clean = lead_in_tone()
    rng = np.random.default_rng(seed)
    noise = generate_noise("white", clean.num_frames, rng)
    mixed = mix_at_snr(clean, noise, snr)
    return clean.samples.astype(np.float64), mixed.samples.astype(np.float64)


def test_subtraction_improves_the_measured_snr() -> None:
    clean, noisy = planted_pair()
    result = denoise(noisy, DenoiseConfig(method="subtraction"))
    before, after, improvement = snr_improvement_db(clean, noisy, result.samples)
    assert before == pytest.approx(5.0, abs=0.05)
    assert after > before
    assert improvement > 2.0


def test_the_wiener_method_improves_the_measured_snr_too() -> None:
    clean, noisy = planted_pair()
    result = denoise(noisy, DenoiseConfig(method="wiener"))
    _, _, improvement = snr_improvement_db(clean, noisy, result.samples)
    assert improvement > 2.0


def test_a_louder_mix_leaves_less_to_recover() -> None:
    clean_quiet, noisy_quiet = planted_pair(seed=6, snr=0.0)
    clean_loud, noisy_loud = planted_pair(seed=6, snr=20.0)
    low = denoise(noisy_quiet, DenoiseConfig()).samples
    high = denoise(noisy_loud, DenoiseConfig()).samples
    gain_at_zero = snr_improvement_db(clean_quiet, noisy_quiet, low)[2]
    gain_at_twenty = snr_improvement_db(clean_loud, noisy_loud, high)[2]
    assert gain_at_zero > gain_at_twenty


def test_the_result_carries_the_gains_and_profile_that_explain_it() -> None:
    _, noisy = planted_pair()
    result = denoise(noisy)
    assert result.gains.shape[1] == 257
    assert result.noise_profile.shape == (257,)
    assert np.all(result.gains >= 0.05)
    assert np.all(result.gains <= 1.0)


def test_an_explicit_zero_profile_changes_nothing() -> None:
    _, noisy = planted_pair()
    result = denoise(noisy, noise_profile=np.zeros(257))
    assert np.all(result.gains == 1.0)
    assert np.allclose(result.samples, noisy, atol=1e-8)


def test_a_profile_of_the_wrong_width_is_refused() -> None:
    _, noisy = planted_pair()
    with pytest.raises(ConfigError, match="one value per bin"):
        denoise(noisy, noise_profile=np.zeros(7))


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ConfigError, match="method"):
        DenoiseConfig(method="magic")


def test_a_clean_recording_with_a_lead_in_survives_almost_untouched() -> None:
    clean = lead_in_tone().samples.astype(np.float64)
    result = denoise(clean)
    lead = int(0.2 * 16000)
    interior = result.samples[lead:]
    reference = clean[lead:]
    relative = np.linalg.norm(interior - reference) / np.linalg.norm(reference)
    assert relative < 1e-4


def test_a_stationary_tone_without_a_lead_in_pays_for_the_assumption() -> None:
    # The profile estimator believes the lead-in is noise.  A tone that starts
    # at sample zero *is* the lead-in, so the denoiser subtracts the tone
    # itself -- the documented failure mode, asserted rather than hidden.
    rate = 16000
    position = np.arange(rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 300.0 * position)
    result = denoise(tone)
    kept = float(np.sum(result.samples[400:-400] ** 2) / np.sum(tone[400:-400] ** 2))
    assert kept < 0.25


def test_the_output_always_has_exactly_the_input_length() -> None:
    rng = np.random.default_rng(9)
    for size in (400, 401, 1005, 16000):
        result = denoise(rng.standard_normal(size))
        assert result.samples.shape == (size,)
        assert np.all(np.isfinite(result.samples))


def test_an_empty_signal_is_refused() -> None:
    with pytest.raises(ConfigError, match="empty"):
        denoise(np.zeros(0))


def test_denoising_is_deterministic() -> None:
    _, noisy = planted_pair()
    first = denoise(noisy)
    second = denoise(noisy)
    assert np.array_equal(first.samples, second.samples)
    assert np.array_equal(first.gains, second.gains)


def test_the_noise_subpackage_re_exports_the_denoiser_surface() -> None:
    from echoscribe import noise

    assert noise.denoise is denoise
    assert noise.snr_db is not None
    for name in (
        "DenoiseConfig",
        "DenoiseResult",
        "estimate_noise_profile",
        "spectral_subtraction_gain",
        "wiener_gain",
    ):
        assert name in noise.__all__
