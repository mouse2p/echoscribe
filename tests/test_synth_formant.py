"""The resonator is checked through its impulse response, which is its whole story."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.formant import (
    apply_resonator,
    normalize_peak,
    render_phone,
    render_sequence,
    resonator_coefficients,
    samples_for_duration,
)
from echoscribe.synth.phones import ENGLISH_PHONES, FINALS, INITIALS
from echoscribe.synth.tone import TONES

RATE = 16000


def _impulse_response(centre: float, bandwidth: float, length: int = 2048) -> np.ndarray:
    delta = np.zeros(length)
    delta[0] = 1.0
    return apply_resonator(delta, centre, bandwidth, RATE)


def test_the_coefficients_give_unity_dc_gain() -> None:
    a0, b1, b2 = resonator_coefficients(500.0, 100.0, RATE)
    assert a0 / (1.0 - b1 - b2) == pytest.approx(1.0, abs=1e-12)


def test_the_pole_radius_follows_the_bandwidth() -> None:
    for centre in (400.0, 1500.0):
        _, b1, b2 = resonator_coefficients(centre, 100.0, RATE)
        radius = math.sqrt(-b2) if b2 < 0 else 0.0
        assert radius == pytest.approx(np.exp(-np.pi * 100.0 / RATE), rel=1e-12)


def test_the_impulse_response_peaks_at_the_centre_frequency() -> None:
    for centre in (300.0, 900.0, 2400.0):
        response = _impulse_response(centre, 120.0)
        spectrum = np.abs(np.fft.rfft(response))
        peak = float(np.fft.rfftfreq(len(response), 1 / RATE)[int(np.argmax(spectrum))])
        assert peak == pytest.approx(centre, abs=RATE / len(response) + 1.0)


def test_a_narrower_bandwidth_gives_a_sharper_peak() -> None:
    def peak_to_median(bandwidth: float) -> float:
        spectrum = np.abs(np.fft.rfft(_impulse_response(1000.0, bandwidth)))
        peak = spectrum[1000 * len(spectrum) // (RATE // 2)]
        return float(peak) / max(float(np.median(spectrum)), 1e-12)

    assert peak_to_median(60.0) > peak_to_median(400.0)


def test_the_response_decays_faster_with_a_wider_bandwidth() -> None:
    narrow = _impulse_response(800.0, 60.0, 4096)
    wide = _impulse_response(800.0, 600.0, 4096)
    tail = slice(2048, 4096)
    assert np.max(np.abs(narrow[tail])) > np.max(np.abs(wide[tail]))


def test_the_filter_is_linear() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(512)
    y = rng.standard_normal(512)
    combined = apply_resonator(2.0 * x + 3.0 * y, 700.0, 90.0, RATE)
    separate = 2.0 * apply_resonator(x, 700.0, 90.0, RATE) + 3.0 * apply_resonator(
        y, 700.0, 90.0, RATE
    )
    assert np.allclose(combined, separate, atol=1e-10)


def test_the_filter_starts_from_a_zero_state() -> None:
    assert _impulse_response(500.0, 100.0, 64)[0] != 0.0
    assert np.all(apply_resonator(np.zeros(64), 500.0, 100.0, RATE) == 0.0)


def test_the_filter_is_stable() -> None:
    response = _impulse_response(2500.0, 200.0, 8192)
    assert np.all(np.isfinite(response))
    assert np.max(np.abs(response[-1024:])) < np.max(np.abs(response[:1024]))


def test_coefficients_reject_impossible_formants() -> None:
    with pytest.raises(ConfigError, match="at or above Nyquist"):
        resonator_coefficients(9000.0, 100.0, RATE)
    with pytest.raises(ConfigError, match="centre_hz must be positive"):
        resonator_coefficients(0.0, 100.0, RATE)
    with pytest.raises(ConfigError, match="bandwidth_hz must be positive"):
        resonator_coefficients(500.0, 0.0, RATE)


def test_the_resonator_rejects_a_matrix() -> None:
    with pytest.raises(ConfigError, match="must be 1-D"):
        apply_resonator(np.zeros((4, 2)), 500.0, 100.0, RATE)


def test_the_loop_matches_a_reference_recursion() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal(600)
    a0, b1, b2 = resonator_coefficients(1200.0, 130.0, RATE)
    expected = np.empty(600)
    y1 = y2 = 0.0
    for index, value in enumerate(x):
        y = a0 * value + b1 * y1 + b2 * y2
        expected[index] = y
        y2 = y1
        y1 = y
    assert np.allclose(apply_resonator(x, 1200.0, 130.0, RATE), expected, rtol=1e-12, atol=0.0)


def test_an_empty_input_produces_an_empty_response() -> None:
    assert apply_resonator(np.zeros(0), 500.0, 100.0, RATE).shape == (0,)


def test_the_output_dtype_is_float64_for_float32_input() -> None:
    result = apply_resonator(np.zeros(8, dtype=np.float32), 500.0, 100.0, RATE)
    assert result.dtype == np.float64


def test_cascading_resonators_sharpens_the_peak() -> None:
    delta = np.zeros(4096)
    delta[0] = 1.0
    single = apply_resonator(delta, 500.0, 90.0, RATE)
    cascade = apply_resonator(single, 500.0, 90.0, RATE)

    def half_power_width(response: np.ndarray) -> int:
        spectrum = np.abs(np.fft.rfft(response))
        return int(np.count_nonzero(spectrum > spectrum.max() / 2.0))

    assert half_power_width(cascade) < half_power_width(single)


def test_a_cascade_keeps_the_peak_at_the_shared_centre() -> None:
    delta = np.zeros(4096)
    delta[0] = 1.0
    cascade = apply_resonator(apply_resonator(delta, 1400.0, 120.0, RATE), 1400.0, 120.0, RATE)
    spectrum = np.abs(np.fft.rfft(cascade))
    peak = float(np.fft.rfftfreq(len(cascade), 1 / RATE)[int(np.argmax(spectrum))])
    assert peak == pytest.approx(1400.0, abs=8.0)


def test_samples_for_duration_rounds_half_up() -> None:
    assert samples_for_duration(25.0, RATE) == 400
    assert samples_for_duration(10.0, RATE) == 160
    assert samples_for_duration(0.03125, RATE) == 1
    assert samples_for_duration(0.0, RATE) == 0


@pytest.mark.parametrize("symbol", list(ENGLISH_PHONES))
def test_every_english_phone_renders_its_declared_length(symbol: str) -> None:
    phone = ENGLISH_PHONES[symbol]
    rendered = render_phone(phone, RATE, rng=np.random.default_rng(1))
    assert rendered.shape == (samples_for_duration(phone.duration_ms, RATE),)
    assert np.all(np.isfinite(rendered))


def test_silence_renders_as_exact_zeros() -> None:
    rendered = render_phone(ENGLISH_PHONES["sil"], RATE)
    assert np.all(rendered == 0.0)
    assert rendered.size > 0


def test_a_turbulent_phone_needs_a_generator() -> None:
    with pytest.raises(ConfigError, match="needs a seeded generator"):
        render_phone(ENGLISH_PHONES["s"], RATE)


def test_a_voiced_phone_does_not_need_a_generator() -> None:
    assert render_phone(ENGLISH_PHONES["iy"], RATE).size > 0


def test_rendering_is_deterministic_for_a_fixed_seed() -> None:
    phone = ENGLISH_PHONES["sh"]
    left = render_phone(phone, RATE, rng=np.random.default_rng(9))
    right = render_phone(phone, RATE, rng=np.random.default_rng(9))
    assert np.array_equal(left, right)


def test_a_different_seed_changes_a_turbulent_phone() -> None:
    phone = ENGLISH_PHONES["sh"]
    left = render_phone(phone, RATE, rng=np.random.default_rng(9))
    right = render_phone(phone, RATE, rng=np.random.default_rng(10))
    assert not np.array_equal(left, right)


def test_a_different_seed_leaves_a_pure_vowel_alone() -> None:
    phone = ENGLISH_PHONES["aa"]
    left = render_phone(phone, RATE, rng=np.random.default_rng(9))
    right = render_phone(phone, RATE, rng=np.random.default_rng(10))
    assert np.array_equal(left, right)


def test_the_envelope_tapers_a_rendered_vowel_to_zero() -> None:
    rendered = render_phone(ENGLISH_PHONES["aa"], RATE)
    assert rendered[0] == 0.0
    assert rendered[-1] == pytest.approx(0.0, abs=1e-9)


def test_a_voiced_vowel_is_periodic_at_the_given_f0() -> None:
    rendered = render_phone(ENGLISH_PHONES["aa"], RATE, f0=100.0)
    period = RATE // 100
    middle = rendered[5 * period : len(rendered) - 5 * period]
    shifted = rendered[6 * period : len(rendered) - 4 * period]
    assert np.corrcoef(middle, shifted)[0, 1] > 0.98


def test_an_f0_track_must_match_the_phone_length() -> None:
    with pytest.raises(ConfigError, match="the phone needs"):
        render_phone(ENGLISH_PHONES["iy"], RATE, f0=np.full(10, 110.0))


def test_rendered_amplitudes_stay_in_a_sane_range() -> None:
    worst = 0.0
    for phone in ENGLISH_PHONES.values():
        rendered = render_phone(phone, RATE, rng=np.random.default_rng(2))
        worst = max(worst, float(np.max(np.abs(rendered))) if rendered.size else 0.0)
    # A three-formant cascade has resonant gain, so the peak exceeds one; four is
    # the bound beyond which the mix would clip on a normal utterance.
    assert worst < 4.0


@pytest.mark.parametrize("symbol", list(INITIALS)[:8])
def test_tone_language_initials_render(symbol: str) -> None:
    phone = INITIALS[symbol]
    rendered = render_phone(phone, RATE, rng=np.random.default_rng(3))
    assert rendered.size == samples_for_duration(phone.duration_ms, RATE)
    assert np.all(np.isfinite(rendered))


@pytest.mark.parametrize("symbol", list(FINALS)[:8])
def test_tone_language_finals_render(symbol: str) -> None:
    phone = FINALS[symbol]
    rendered = render_phone(phone, RATE, f0=110.0)
    assert rendered.size == samples_for_duration(phone.duration_ms, RATE)
    assert np.all(np.isfinite(rendered))


def test_a_sequence_is_the_concatenation_of_its_phones() -> None:
    phones = [ENGLISH_PHONES["t"], ENGLISH_PHONES["ay"], ENGLISH_PHONES["m"]]
    rendered = render_sequence(phones, RATE, rng=np.random.default_rng(2))
    expected = sum(samples_for_duration(p.duration_ms, RATE) for p in phones)
    assert rendered.shape == (expected,)


def test_rendering_a_sequence_twice_with_one_seed_is_identical() -> None:
    phones = [ENGLISH_PHONES["s"], ENGLISH_PHONES["ih"], ENGLISH_PHONES["t"]]
    left = render_sequence(phones, RATE, rng=np.random.default_rng(6))
    right = render_sequence(phones, RATE, rng=np.random.default_rng(6))
    assert np.array_equal(left, right)


def test_a_sequence_without_a_tone_matches_phone_by_phone_rendering() -> None:
    phones = [ENGLISH_PHONES["m"], ENGLISH_PHONES["ow"]]
    rng_left = np.random.default_rng(5)
    rng_right = np.random.default_rng(5)
    manual = np.concatenate([render_phone(p, RATE, rng=rng_left) for p in phones])
    assert np.array_equal(render_sequence(phones, RATE, rng=rng_right), manual)


def test_silence_contributes_zeros_to_a_sequence() -> None:
    phones = [ENGLISH_PHONES["sil"], ENGLISH_PHONES["aa"]]
    rendered = render_sequence(phones, RATE)
    silence = samples_for_duration(ENGLISH_PHONES["sil"].duration_ms, RATE)
    assert np.all(rendered[:silence] == 0.0)


def test_a_tone_changes_the_rendered_nucleus() -> None:
    rising = render_sequence([FINALS["a"]], RATE, base_f0=110.0, tone=TONES[1])
    falling = render_sequence([FINALS["a"]], RATE, base_f0=110.0, tone=TONES[3])
    assert rising.shape == falling.shape
    assert not np.allclose(rising, falling)


def test_a_tone_preserves_the_total_duration() -> None:
    phones = [INITIALS["m"], FINALS["ang"]]
    assert render_sequence(phones, RATE).shape == render_sequence(phones, RATE, tone=TONES[2]).shape


def test_an_empty_sequence_is_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot render an empty phone sequence"):
        render_sequence([], RATE)


def test_a_rendered_sequence_stays_in_a_sane_amplitude_range() -> None:
    phones = [ENGLISH_PHONES[s] for s in ("sh", "iy", "sil", "b", "aa")]
    rendered = render_sequence(phones, RATE, rng=np.random.default_rng(12))
    assert np.max(np.abs(rendered)) < 4.0
    assert np.all(np.isfinite(rendered))


def test_normalisation_puts_the_peak_at_the_target() -> None:
    signal = np.array([0.2, -4.0, 1.0])
    assert float(np.max(np.abs(normalize_peak(signal, 0.9)))) == pytest.approx(0.9)


def test_normalisation_preserves_the_shape() -> None:
    rng = np.random.default_rng(8)
    signal = rng.standard_normal(256) * 3.0
    scaled = normalize_peak(signal, 0.5)
    ratio = float(np.max(np.abs(signal))) / 0.5
    assert np.allclose(scaled * ratio, signal, atol=1e-12)


def test_normalisation_of_silence_is_silence() -> None:
    assert np.all(normalize_peak(np.zeros(64)) == 0.0)


def test_normalisation_of_an_empty_signal_is_empty() -> None:
    assert normalize_peak(np.zeros(0)).shape == (0,)


def test_the_default_target_leaves_headroom() -> None:
    scaled = normalize_peak(np.array([0.0, 8.0, -8.0]))
    assert float(np.max(np.abs(scaled))) == pytest.approx(0.9)


@pytest.mark.parametrize("target", [-0.1, 1.5])
def test_normalisation_rejects_an_out_of_range_target(target: float) -> None:
    with pytest.raises(ConfigError, match="target must lie in"):
        normalize_peak(np.ones(4), target)


def test_normalisation_does_not_mutate_its_input() -> None:
    signal = np.array([0.5, -2.0])
    normalize_peak(signal, 0.8)
    assert np.array_equal(signal, np.array([0.5, -2.0]))


def test_a_normalised_render_survives_a_16bit_round_trip() -> None:
    from echoscribe.audio import float32_to_pcm16, pcm16_to_float32

    phones = [ENGLISH_PHONES[s] for s in ("sh", "iy", "sil", "b", "aa")]
    rendered = normalize_peak(render_sequence(phones, RATE, rng=np.random.default_rng(13)))
    recovered = pcm16_to_float32(float32_to_pcm16(rendered.astype(np.float32)))
    assert np.max(np.abs(recovered.astype(np.float64) - rendered)) < 1e-3
