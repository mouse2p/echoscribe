"""Noise-profile estimation and the end-to-end waveform denoiser.

The estimator here makes one assumption and says so: the recording begins
with noise before it begins with speech, and the first frames are therefore
a sample of the noise alone.  Recordings from the synthetic laboratory are
mixed that way, and so are most real capture pipelines that keep a moment of
room tone before the first word.  A tracker that followed a moving noise
floor (minimum statistics) would drop the assumption; it would also stop
being a dozen testable lines, and everything downstream only needs *a*
profile, not the best possible one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from echoscribe.errors import ConfigError, require, require_in, require_positive
from echoscribe.noise.gains import (
    GAIN_METHODS,
    spectral_subtraction_gain,
    wiener_gain,
)
from echoscribe.noise.stft import StftConfig, istft, power_spectrogram, stft
from echoscribe.types import FEATURE_DTYPE, FloatArray, require_rank


def estimate_noise_profile(power: FloatArray, frames: int) -> FloatArray:
    """The per-bin mean power over the first ``frames`` frames.

    Averaging rather than taking a minimum biases the profile up when the
    lead-in contains any speech, which costs a little signal and buys
    stability: a minimum over few frames underestimates the noise floor and
    then leaves exactly the residual the gain was supposed to remove.
    """
    matrix = np.asarray(power, dtype=FEATURE_DTYPE)
    require(
        matrix.ndim == 2,
        f"power must be a (frames, bins) matrix, got {matrix.ndim}-D",
        ConfigError,
    )
    require(
        isinstance(frames, int) and not isinstance(frames, bool) and frames >= 1,
        f"frames must be an integer >= 1, got {frames!r}",
        ConfigError,
    )
    require(
        frames <= matrix.shape[0],
        f"the profile asks for {frames} frames but the spectrogram has {matrix.shape[0]}",
        ConfigError,
    )
    return matrix[:frames].mean(axis=0)


@dataclass(frozen=True)
class DenoiseConfig:
    """Which gain, with what knobs, over which transform."""

    method: str = "subtraction"
    oversubtraction: float = 1.5
    floor: float = 0.05
    smoothing: float = 0.9
    profile_frames: int = 10
    transform: StftConfig = field(default_factory=StftConfig)

    def __post_init__(self) -> None:
        require_in("method", self.method, GAIN_METHODS, ConfigError)
        require_positive("oversubtraction", self.oversubtraction, ConfigError)
        require(
            0.0 <= self.floor < 1.0,
            f"floor must stay in [0.0, 1.0), got {self.floor!r}",
            ConfigError,
        )
        require(
            0.0 <= self.smoothing <= 1.0,
            f"smoothing must stay in [0.0, 1.0], got {self.smoothing!r}",
            ConfigError,
        )
        require(
            isinstance(self.profile_frames, int)
            and not isinstance(self.profile_frames, bool)
            and self.profile_frames >= 1,
            f"profile_frames must be an integer >= 1, got {self.profile_frames!r}",
            ConfigError,
        )


@dataclass(frozen=True)
class DenoiseResult:
    """The enhanced waveform plus the two arrays that explain it."""

    samples: FloatArray
    gains: FloatArray
    noise_profile: FloatArray


def denoise(
    signal: np.ndarray,
    config: DenoiseConfig | None = None,
    noise_profile: np.ndarray | None = None,
) -> DenoiseResult:
    """Denoise one waveform: pad, transform, gain every bin, transform back.

    The signal is reflect-padded by one frame at each end before the
    transform and the padding is trimmed afterwards.  That is not cosmetic:
    the outermost samples of a reconstruction hang on synthesis-window
    weights near zero, and where a gain has modified the frame content, the
    normalized overlap-add divides by those weights -- a quiet edge sample
    divided by a near-zero weight is a click.  Padding moves every original
    sample into the well-conditioned interior.

    The returned record carries the gains and the noise profile that were
    actually applied, so a result can be inspected -- or refuted -- without
    running the denoiser again.
    """
    settings = DenoiseConfig() if config is None else config
    samples = np.asarray(signal, dtype=FEATURE_DTYPE)
    require_rank("signal", samples, 1)
    require(samples.size > 0, "cannot denoise an empty signal", ConfigError)
    pad = settings.transform.frame_length
    padded = np.concatenate([samples[:pad][::-1], samples, samples[-pad:][::-1]])
    spectrum = stft(padded, settings.transform)
    power = power_spectrogram(spectrum)
    if noise_profile is None:
        # A very short signal may simply not contain the requested lead-in;
        # estimate from every frame it has rather than refusing to run.
        lead_in = min(settings.profile_frames, power.shape[0])
        profile = estimate_noise_profile(power, lead_in)
    else:
        profile = np.asarray(noise_profile, dtype=FEATURE_DTYPE)
        require(
            profile.ndim == 1 and profile.size == power.shape[1],
            f"the noise profile must hold one value per bin ({power.shape[1]}), "
            f"got shape {tuple(profile.shape)}",
            ConfigError,
        )
        require(
            np.all(np.isfinite(profile)) and np.all(profile >= 0),
            "the noise profile must be finite and non-negative",
            ConfigError,
        )
    if settings.method == "subtraction":
        gains = spectral_subtraction_gain(power, profile, settings.oversubtraction, settings.floor)
    else:
        gains = wiener_gain(power, profile, settings.floor, settings.smoothing)
    rebuilt = istft(spectrum * gains, settings.transform)
    return DenoiseResult(
        samples=rebuilt[pad : pad + samples.size], gains=gains, noise_profile=profile
    )
