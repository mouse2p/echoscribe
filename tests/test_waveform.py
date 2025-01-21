"""Audio is the contract every other module relies on, so it is pinned tightly."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import AudioError, FeatureError
from echoscribe.waveform import Audio


def test_audio_keeps_its_samples_and_sample_rate() -> None:
    audio = Audio([0.0, 0.5, -0.5], 16000)
    assert list(audio.samples) == [0.0, 0.5, -0.5]
    assert audio.sample_rate == 16000


def test_audio_coerces_samples_to_float32() -> None:
    audio = Audio(np.array([0, 1, 2], dtype=np.int16), 8000)
    assert audio.samples.dtype == np.float32


def test_audio_is_immutable() -> None:
    audio = Audio([0.0], 16000)
    with pytest.raises(AttributeError):
        audio.sample_rate = 8000  # type: ignore[misc]


@pytest.mark.parametrize("rate", [0, -1, -16000])
def test_audio_rejects_a_non_positive_sample_rate(rate: int) -> None:
    with pytest.raises(AudioError, match="positive integer"):
        Audio([0.0], rate)


@pytest.mark.parametrize("rate", [1.0, "16000", None, True])
def test_audio_rejects_a_sample_rate_that_is_not_an_int(rate) -> None:
    with pytest.raises(AudioError):
        Audio([0.0], rate)  # type: ignore[arg-type]


def test_audio_rejects_stereo_input() -> None:
    with pytest.raises(FeatureError, match="must be 1-D"):
        Audio(np.zeros((4, 2), dtype=np.float32), 16000)


def test_audio_rejects_non_finite_samples() -> None:
    with pytest.raises(FeatureError, match="must be finite"):
        Audio([0.0, float("nan")], 16000)


def test_audio_accepts_an_empty_buffer() -> None:
    audio = Audio([], 16000)
    assert audio.samples.shape == (0,)
