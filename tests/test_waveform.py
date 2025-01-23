"""Audio is the contract every other module relies on, so it is pinned tightly."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import AudioError, FeatureError
from echoscribe.waveform import Audio, concatenate, silence


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


def test_frame_count_and_duration_agree() -> None:
    audio = Audio(np.zeros(1600, dtype=np.float32), 16000)
    assert audio.num_frames == 1600
    assert audio.duration == pytest.approx(0.1)


def test_duration_of_an_empty_buffer_is_zero() -> None:
    assert Audio([], 16000).duration == 0.0


def test_peak_ignores_sign() -> None:
    assert Audio([0.2, -0.9, 0.4], 16000).peak() == pytest.approx(0.9)


def test_peak_of_an_empty_buffer_is_zero() -> None:
    assert Audio([], 8000).peak() == 0.0


def test_rms_of_a_full_scale_square_wave_is_one() -> None:
    samples = np.tile([-1.0, 1.0], 32).astype(np.float32)
    assert Audio(samples, 16000).rms() == pytest.approx(1.0)


def test_rms_of_a_sine_is_one_over_root_two() -> None:
    t = np.arange(16000, dtype=np.float64) / 16000.0
    samples = np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)
    assert Audio(samples, 16000).rms() == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-3)


def test_rms_never_exceeds_peak() -> None:
    rng = np.random.default_rng(1)
    audio = Audio(rng.uniform(-0.7, 0.7, 512).astype(np.float32), 16000)
    assert audio.rms() <= audio.peak() + 1e-9


def test_is_silent_holds_for_zeros_and_fails_for_a_spike() -> None:
    assert Audio(np.zeros(64, dtype=np.float32), 16000).is_silent()
    assert not Audio([0.0] * 63 + [0.5], 16000).is_silent()


def test_is_silent_threshold_is_inclusive() -> None:
    audio = Audio([0.01], 16000)
    assert audio.is_silent(threshold=0.01)
    assert not audio.is_silent(threshold=0.009)


def test_is_silent_rejects_a_negative_threshold() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        Audio([0.0], 16000).is_silent(threshold=-0.1)


def test_peak_and_rms_are_computed_in_float64() -> None:
    # 0.1 squared summed 10000 times in float32 drifts visibly; float64 does not.
    samples = np.full(10000, 0.1, dtype=np.float32)
    audio = Audio(samples, 16000)
    assert audio.rms() == pytest.approx(0.1, abs=1e-7)
    assert audio.peak() == pytest.approx(float(np.float32(0.1)), abs=0.0)


def test_audio_does_not_alias_the_caller_array() -> None:
    source = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    audio = Audio(source, 16000)
    source[0] = -9.0
    assert audio.samples[0] == pytest.approx(0.1)


def test_empty_audio_reports_a_zero_duration_at_any_rate() -> None:
    for rate in (8000, 16000, 44100):
        assert Audio([], rate).duration == 0.0
        assert Audio([], rate).rms() == 0.0


def test_slice_is_half_open() -> None:
    audio = Audio([0.0, 1.0, 2.0, 3.0], 16000)
    assert list(audio.slice(1, 3).samples) == [1.0, 2.0]


def test_slice_keeps_the_sample_rate() -> None:
    assert Audio([0.0, 1.0], 44100).slice(0, 1).sample_rate == 44100


def test_slice_boundaries_are_inclusive_of_both_ends_of_the_buffer() -> None:
    audio = Audio([0.0, 1.0, 2.0], 16000)
    assert audio.slice(0, 3).num_frames == 3
    assert audio.slice(3, 3).num_frames == 0
    assert audio.slice(0, 0).num_frames == 0


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 2), (0, 4), (2, 1), (5, 5)],
)
def test_slice_rejects_out_of_range_and_inverted_bounds(start: int, end: int) -> None:
    with pytest.raises(AudioError, match="is outside"):
        Audio([0.0, 1.0, 2.0], 16000).slice(start, end)


def test_slice_returns_an_independent_copy() -> None:
    audio = Audio([0.0, 1.0, 2.0, 3.0], 16000)
    part = audio.slice(0, 2)
    part.samples[0] = -5.0
    assert audio.samples[0] == 0.0


def test_silence_length_follows_the_sample_rate() -> None:
    assert silence(0.01, 16000).num_frames == 160
    assert silence(0.01, 8000).num_frames == 80
    assert silence(0.0, 16000).num_frames == 0


def test_silence_rounds_instead_of_truncating() -> None:
    # 3 samples at 16 kHz is 0.1875 ms; truncation would give 2 frames.
    assert silence(3 / 16000, 16000).num_frames == 3


def test_silence_is_silent() -> None:
    assert silence(0.5, 16000).is_silent(threshold=0.0)


def test_silence_rejects_a_negative_duration() -> None:
    with pytest.raises(AudioError, match="non-negative"):
        silence(-0.1, 16000)


def test_concatenate_joins_in_order() -> None:
    joined = concatenate([Audio([0.0], 16000), Audio([1.0, 2.0], 16000)])
    assert list(joined.samples) == [0.0, 1.0, 2.0]
    assert joined.sample_rate == 16000


def test_concatenate_of_a_single_part_is_a_copy() -> None:
    original = Audio([0.5], 16000)
    joined = concatenate([original])
    joined.samples[0] = -1.0
    assert original.samples[0] == pytest.approx(0.5)


def test_concatenate_rejects_mixed_sample_rates() -> None:
    with pytest.raises(AudioError, match="part 1 is at 8000 Hz"):
        concatenate([Audio([0.0], 16000), Audio([0.0], 8000)])


def test_concatenate_rejects_an_empty_sequence() -> None:
    with pytest.raises(AudioError, match="at least one waveform"):
        concatenate([])


def test_silence_then_concatenate_preserves_total_duration() -> None:
    gap = silence(0.01, 16000)
    joined = concatenate([gap, Audio(np.ones(160, dtype=np.float32), 16000), gap])
    assert joined.duration == pytest.approx(0.03)
