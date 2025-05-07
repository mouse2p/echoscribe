"""Excitation sources for the formant synthesizer.

A source-filter synthesizer needs two excitations: a quasi-periodic glottal flow
for voiced sounds, and turbulence for fricatives, plosive bursts and affricates.
Both are generated here from a caller-supplied :class:`numpy.random.Generator`, so
the whole laboratory stays reproducible from a seed.

The glottal pulse is a two-part shape rather than an impulse: a raised-cosine
opening over the first ``open_fraction`` of the period and a squared decay over
the rest.  That gives a source spectrum falling with frequency, which is what
makes the formant peaks of the filter stage visible instead of being buried under
a flat excitation.

The train is built from a *phase accumulator* rather than by tiling one period, so
an F0 that moves -- as it does under a lexical tone -- changes the period smoothly
instead of restarting the waveform.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.types import FEATURE_DTYPE, FloatArray

#: Fraction of one glottal period spent opening.
GLOTTAL_OPEN_FRACTION = 0.6

#: Default base F0 in Hz for a rendered utterance.
DEFAULT_BASE_F0 = 120.0

#: Anything above this is not a plausible voice fundamental.
MAX_F0 = 1000.0


def phase_track(f0: np.ndarray, sample_rate: int, initial_phase: float = 0.0) -> FloatArray:
    """Integrate an F0 trajectory into a wrapped phase in ``[0, 1)``."""
    require_positive("sample_rate", sample_rate, ConfigError)
    require(
        0.0 <= initial_phase < 1.0,
        f"initial_phase must lie in [0, 1), got {initial_phase!r}",
        ConfigError,
    )
    track = np.asarray(f0, dtype=FEATURE_DTYPE)
    require(track.ndim == 1, f"f0 track must be 1-D, got shape {track.shape}", ConfigError)
    require(bool(np.all(track >= 0.0)), "f0 must not be negative", ConfigError)
    return np.mod(initial_phase + np.cumsum(track / sample_rate), 1.0)


def glottal_pulse(phase: np.ndarray, open_fraction: float = GLOTTAL_OPEN_FRACTION) -> FloatArray:
    """One period of the glottal flow shape, indexed by phase in ``[0, 1)``."""
    require(
        0.0 < open_fraction < 1.0,
        f"open_fraction must lie in (0, 1), got {open_fraction!r}",
        ConfigError,
    )
    values = np.asarray(phase, dtype=FEATURE_DTYPE)
    opening = 0.5 * (1.0 - np.cos(np.pi * np.minimum(values / open_fraction, 1.0)))
    closing_ratio = np.clip((values - open_fraction) / (1.0 - open_fraction), 0.0, 1.0)
    return np.where(values < open_fraction, opening, (1.0 - closing_ratio) ** 2)


def check_f0_track(track: np.ndarray, num_samples: int) -> None:
    """Reject an F0 track a voice could not produce or that has the wrong length."""
    require(
        track.shape == (num_samples,),
        f"f0 track has {track.shape[0]} samples but {num_samples} were requested",
        ConfigError,
    )
    require(bool(np.all(track > 0.0)), "f0 must be strictly positive everywhere", ConfigError)
    require(
        float(track.max()) < MAX_F0,
        f"f0 above {MAX_F0} Hz is not a voice, got {float(track.max()):.1f}",
        ConfigError,
    )


def glottal_source(
    num_samples: int,
    sample_rate: int,
    f0: np.ndarray | float,
    initial_phase: float = 0.0,
) -> FloatArray:
    """Render ``num_samples`` of voiced excitation following the ``f0`` track.

    A scalar ``f0`` is held constant; an array must have exactly ``num_samples``
    entries.  The mean is removed afterwards, because a source with a DC component
    would put a spike at bin 0 of every frame's spectrum.
    """
    require(
        isinstance(num_samples, int) and num_samples >= 0,
        f"num_samples must be a non-negative integer, got {num_samples!r}",
        ConfigError,
    )
    if num_samples == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    if np.ndim(f0) == 0:
        track = np.full(num_samples, float(f0))
    else:
        track = np.asarray(f0, dtype=FEATURE_DTYPE)
    check_f0_track(track, num_samples)
    pulse = glottal_pulse(phase_track(track, sample_rate, initial_phase))
    return pulse - pulse.mean()


def noise_source(
    num_samples: int,
    rng: np.random.Generator,
    tilt_hz: float | None = None,
    sample_rate: int = 16000,
) -> FloatArray:
    """Render ``num_samples`` of Gaussian turbulence, optionally low-pass tilted.

    The output is scaled so its largest absolute sample is exactly one, which
    guarantees it cannot push a mixed waveform out of range.  That makes the
    standard deviation about 0.29 for white noise rather than 1.0 -- the peak, not
    the variance, is what a renderer has to control.

    ``tilt_hz`` applies a single-pole low-pass so a fricative can be given a darker
    spectrum than a plosive burst without a second code path.  ``None`` leaves the
    noise white.
    """
    require(
        isinstance(num_samples, int) and num_samples >= 0,
        f"num_samples must be a non-negative integer, got {num_samples!r}",
        ConfigError,
    )
    if num_samples == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    raw = rng.standard_normal(num_samples)
    if tilt_hz is not None:
        require_positive("tilt_hz", tilt_hz, ConfigError)
        raw = _one_pole_lowpass(raw, tilt_hz, sample_rate)
    peak = float(np.max(np.abs(raw)))
    return raw / peak if peak > 1e-12 else raw


def _one_pole_lowpass(values: np.ndarray, corner_hz: float, sample_rate: int) -> np.ndarray:
    """A causal single-pole low-pass, the cheapest useful spectral tilt."""
    coefficient = float(np.exp(-2.0 * np.pi * corner_hz / sample_rate))
    output = []
    previous = 0.0
    for value in values.tolist():
        previous = value + coefficient * previous
        output.append(previous)
    return np.asarray(output, dtype=FEATURE_DTYPE)
