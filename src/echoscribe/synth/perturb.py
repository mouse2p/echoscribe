"""Speed and volume perturbation for corpus augmentation.

Both are deterministic given a seed, and both are honest about what they change:

*speed*
    time-scaling without pitch correction.  Stretching a waveform to 1.1x its
    length lowers every formant *and* F0 by the same factor, which is not what a
    speaker talking more slowly does.  It is still the standard augmentation because
    it widens the acoustic condition a decoder has to survive, not because it is a
    model of speaking rate.
*volume*
    a plain gain in dB, applied before clipping.  The result is clipped to
    ``[-1.0, 1.0]`` and the number of clipped samples is reported, because a
    distorted waveform is a different acoustic condition and a caller should be
    able to see when it has created one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.resample import ResampleConfig, resample
from echoscribe.types import FEATURE_DTYPE
from echoscribe.waveform import Audio


@dataclass(frozen=True)
class VolumeResult:
    """A gained waveform plus how much of it had to be clipped."""

    audio: Audio
    clipped: int

    @property
    def clipped_ratio(self) -> float:
        """Fraction of samples that hit the ``[-1, 1]`` boundary."""
        return self.clipped / self.audio.num_frames if self.audio.num_frames else 0.0


def volume_perturb(audio: Audio, gain_db: float, clip: bool = True) -> VolumeResult:
    """Apply a gain in dB, optionally clipping to the representable range."""
    gain = 10.0 ** (gain_db / 20.0)
    scaled = audio.samples.astype(FEATURE_DTYPE) * gain
    if not clip:
        return VolumeResult(Audio(scaled.astype(np.float32), audio.sample_rate), 0)
    outside = int(np.count_nonzero(np.abs(scaled) > 1.0))
    limited = np.clip(scaled, -1.0, 1.0)
    return VolumeResult(Audio(limited.astype(np.float32), audio.sample_rate), outside)


def speed_perturb(audio: Audio, factor: float, method: str = "polyphase") -> Audio:
    """Time-scale a waveform by ``factor`` without touching its sample rate.

    ``factor > 1`` makes the utterance *longer* and lowers its pitch, matching the
    convention that the factor is a duration multiplier rather than a rate
    multiplier.  The implementation resamples to ``rate * factor`` -- which produces
    ``factor`` times as many samples of the same waveform -- and then relabels the
    result at the original rate.  That is the cheapest way to get a correct time
    scale with a band-limited result.
    """
    require_positive("factor", factor, ConfigError)
    require(audio.num_frames > 0, "cannot time-scale an empty waveform", ConfigError)
    intermediate = int(round(audio.sample_rate * factor))
    require_positive("intermediate rate", intermediate, ConfigError)
    converted = resample(audio, ResampleConfig(target_rate=intermediate, method=method))
    return Audio(converted.samples, audio.sample_rate)
