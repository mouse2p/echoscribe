"""The room model is pinned through its decay law, which is the only claim it makes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.room import (
    RT60_DECAY_CONSTANT,
    RoomConfig,
    apply_room,
    decay_envelope,
    fft_convolve,
    impulse_response,
)
from echoscribe.waveform import Audio

RATE = 16000


def test_the_decay_constant_is_ln_of_a_thousand() -> None:
    assert RT60_DECAY_CONSTANT == pytest.approx(math.log(1000.0))


def test_the_envelope_starts_at_one() -> None:
    assert decay_envelope(200.0, RATE, 8)[0] == pytest.approx(1.0)


def test_the_envelope_reaches_minus_sixty_db_at_rt60() -> None:
    envelope = decay_envelope(200.0, RATE, RATE)
    assert envelope[int(0.2 * RATE)] == pytest.approx(1e-3, rel=1e-6)


def test_a_longer_rt60_decays_more_slowly() -> None:
    short = decay_envelope(100.0, RATE, RATE)
    long = decay_envelope(600.0, RATE, RATE)
    assert long[RATE // 2] > short[RATE // 2]


def test_the_envelope_is_strictly_decreasing() -> None:
    envelope = decay_envelope(300.0, RATE, 4096)
    assert np.all(np.diff(envelope) < 0.0)


def test_the_envelope_of_no_samples_is_empty() -> None:
    assert decay_envelope(200.0, RATE, 0).shape == (0,)


def test_the_envelope_rejects_bad_arguments() -> None:
    with pytest.raises(ConfigError, match="rt60_ms must be positive"):
        decay_envelope(0.0, RATE, 16)
    with pytest.raises(ConfigError, match="num_samples must be a non-negative integer"):
        decay_envelope(200.0, RATE, -1)


def test_config_defaults_describe_a_small_room() -> None:
    config = RoomConfig()
    assert config.rt60_ms == 200.0
    assert config.direct_gain == 1.0
    assert config.tail_gain == 0.3
    assert config.early_reflections == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rt60_ms": 0.0},
        {"direct_gain": 0.0},
        {"tail_gain": -1.0},
        {"early_reflections": -1},
        {"seed": 7},
    ],
)
def test_invalid_room_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        RoomConfig(**kwargs)


def _burst(seconds: float = 0.1) -> Audio:
    samples = np.zeros(int(RATE * seconds), dtype=np.float32)
    samples[0] = 1.0
    return Audio(samples, RATE)


def test_the_impulse_response_starts_with_the_direct_path() -> None:
    config = RoomConfig(tail_gain=1e-9, early_reflections=0)
    response = impulse_response(config, RATE)
    assert response[0] == pytest.approx(config.direct_gain, abs=1e-6)


def test_the_impulse_response_length_defaults_to_a_little_over_one_rt60() -> None:
    config = RoomConfig(rt60_ms=250.0)
    response = impulse_response(config, RATE)
    expected = int(math.ceil(0.25 * RATE * 1.2)) + 1
    assert response.shape == (expected,)


def test_an_explicit_length_is_honoured() -> None:
    assert impulse_response(RoomConfig(), RATE, 64).shape == (64,)


def test_early_reflections_add_energy_after_the_direct_path() -> None:
    plain = impulse_response(RoomConfig(early_reflections=0, tail_gain=1e-9), RATE, 512)
    with_early = impulse_response(RoomConfig(early_reflections=5, tail_gain=1e-9), RATE, 512)
    assert float(np.sum(np.abs(with_early[1:]))) > float(np.sum(np.abs(plain[1:])))


def test_the_response_is_reproducible_from_the_seed() -> None:
    config = RoomConfig()
    assert np.array_equal(impulse_response(config, RATE), impulse_response(config, RATE))


def test_a_different_seed_gives_a_different_tail() -> None:
    from echoscribe.seeds import Seed

    left = impulse_response(RoomConfig(seed=Seed(1, "a")), RATE)
    right = impulse_response(RoomConfig(seed=Seed(2, "a")), RATE)
    assert not np.allclose(left[16:], right[16:])


def test_the_tail_energy_follows_the_decay_envelope() -> None:
    config = RoomConfig(rt60_ms=300.0, direct_gain=1e-9, early_reflections=0, tail_gain=1.0)
    response = impulse_response(config, RATE, RATE)

    def block_energy(start: int, width: int = 160) -> float:
        return float(np.mean(response[start : start + width] ** 2))

    ratio = block_energy(int(0.3 * RATE)) / block_energy(0)
    # Amplitude falls by 1e-3 over one RT60, so power falls by about 1e-6.
    assert 1e-7 < ratio < 1e-4


def test_fft_convolve_matches_the_direct_convolution() -> None:
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(200)
    kernel = rng.standard_normal(37)
    assert np.allclose(fft_convolve(signal, kernel), np.convolve(signal, kernel), atol=1e-10)


def test_fft_convolve_output_length() -> None:
    assert fft_convolve(np.ones(10), np.ones(4)).shape == (13,)


def test_fft_convolve_of_an_empty_input_is_empty() -> None:
    assert fft_convolve(np.zeros(0), np.ones(4)).shape == (0,)


def test_fft_convolve_with_a_unit_impulse_is_the_identity() -> None:
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(128)
    kernel = np.zeros(5)
    kernel[0] = 1.0
    assert np.allclose(fft_convolve(signal, kernel)[:128], signal, atol=1e-12)


def test_fft_convolve_rejects_a_matrix() -> None:
    with pytest.raises(ConfigError, match="two 1-D inputs"):
        fft_convolve(np.zeros((4, 2)), np.ones(3))


def test_applying_a_room_lengthens_the_waveform_by_the_tail() -> None:
    audio = _burst(0.1)
    config = RoomConfig(rt60_ms=200.0)
    reverberant = apply_room(audio, config)
    response_length = impulse_response(config, RATE).size
    assert reverberant.num_frames == audio.num_frames + response_length - 1


def test_a_dry_room_leaves_the_waveform_almost_unchanged() -> None:
    audio = _burst(0.05)
    config = RoomConfig(tail_gain=1e-9, early_reflections=0)
    reverberant = apply_room(audio, config)
    assert reverberant.samples[0] == pytest.approx(1.0, abs=1e-5)
    assert np.max(np.abs(reverberant.samples[1 : audio.num_frames])) < 1e-5


def test_reverberation_spreads_a_delta_in_time() -> None:
    reverberant = apply_room(_burst(0.05), RoomConfig(rt60_ms=300.0))
    head = float(np.sum(reverberant.samples[:160] ** 2))
    tail = float(np.sum(reverberant.samples[160:] ** 2))
    assert tail > head


def test_applying_a_room_rejects_an_empty_waveform() -> None:
    with pytest.raises(ConfigError, match="cannot reverberate an empty waveform"):
        apply_room(Audio(np.zeros(0, dtype=np.float32), RATE), RoomConfig())
