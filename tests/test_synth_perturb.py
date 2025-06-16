"""Perturbation is checked on the quantity it claims to change, and no other."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.perturb import VolumeResult, speed_perturb, volume_perturb
from echoscribe.waveform import Audio

RATE = 16000


def _tone(seconds: float = 0.2, amplitude: float = 0.3) -> Audio:
    t = np.arange(int(RATE * seconds), dtype=np.float64) / RATE
    return Audio((amplitude * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32), RATE)


def test_a_zero_db_gain_is_the_identity() -> None:
    audio = _tone()
    result = volume_perturb(audio, 0.0)
    assert isinstance(result, VolumeResult)
    assert np.allclose(result.audio.samples, audio.samples, atol=1e-7)
    assert result.clipped == 0


def test_a_six_db_gain_doubles_the_amplitude() -> None:
    audio = _tone(amplitude=0.2)
    result = volume_perturb(audio, 6.0206)
    assert float(np.max(np.abs(result.audio.samples))) == pytest.approx(0.4, abs=1e-3)


def test_a_negative_gain_attenuates() -> None:
    audio = _tone(amplitude=0.5)
    result = volume_perturb(audio, -6.0206)
    assert float(np.max(np.abs(result.audio.samples))) == pytest.approx(0.25, abs=1e-3)


def test_clipping_is_counted_and_bounded() -> None:
    audio = _tone(amplitude=0.9)
    result = volume_perturb(audio, 12.0)
    assert result.clipped > 0
    assert float(np.max(np.abs(result.audio.samples))) <= 1.0
    assert 0.0 < result.clipped_ratio < 1.0


def test_clipping_can_be_switched_off() -> None:
    audio = _tone(amplitude=0.9)
    result = volume_perturb(audio, 12.0, clip=False)
    assert result.clipped == 0
    assert float(np.max(np.abs(result.audio.samples))) > 1.0


def test_the_gain_does_not_change_length_or_rate() -> None:
    audio = _tone()
    result = volume_perturb(audio, 9.0)
    assert result.audio.num_frames == audio.num_frames
    assert result.audio.sample_rate == RATE


def test_volume_perturb_does_not_mutate_its_input() -> None:
    audio = _tone()
    before = audio.samples.copy()
    volume_perturb(audio, 20.0)
    assert np.array_equal(audio.samples, before)


def test_stretching_lengthens_the_waveform() -> None:
    audio = _tone(0.2)
    slower = speed_perturb(audio, 1.5)
    assert slower.num_frames == pytest.approx(audio.num_frames * 1.5, abs=2)
    assert slower.sample_rate == RATE


def test_compressing_shortens_the_waveform() -> None:
    audio = _tone(0.2)
    faster = speed_perturb(audio, 0.75)
    assert faster.num_frames == pytest.approx(audio.num_frames * 0.75, abs=2)


def test_a_unit_factor_keeps_the_length() -> None:
    audio = _tone(0.2)
    assert speed_perturb(audio, 1.0).num_frames == pytest.approx(audio.num_frames, abs=1)


def test_stretching_lowers_the_dominant_frequency() -> None:
    def dominant(audio: Audio) -> float:
        padded = np.zeros(1 << 15)
        padded[: audio.num_frames] = audio.samples * np.hanning(audio.num_frames)
        spectrum = np.abs(np.fft.rfft(padded))
        bins = np.fft.rfftfreq(len(padded), 1 / RATE)
        return float(bins[int(np.argmax(spectrum))])

    audio = _tone(0.5)
    assert dominant(speed_perturb(audio, 1.25)) == pytest.approx(300.0 / 1.25, abs=2.0)


def test_speed_perturb_rejects_bad_arguments() -> None:
    with pytest.raises(ConfigError, match="factor must be positive"):
        speed_perturb(_tone(), 0.0)
    with pytest.raises(ConfigError, match="cannot time-scale an empty waveform"):
        speed_perturb(Audio(np.zeros(0, dtype=np.float32), RATE), 1.1)
