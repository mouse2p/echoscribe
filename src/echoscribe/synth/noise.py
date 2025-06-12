"""Background noise, SNR control and the mixing that puts them together.

The laboratory needs noise for two reasons: to check that a feature pipeline
degrades the way the theory says it should, and to give an endpointer something
to reject between utterances.  Both need noise whose *statistics* are known, which
is why the generators here are specified by their spectrum rather than by a
recording.

Three kinds are available:

``white``
    flat power spectral density.  The reference case.
``pink``
    power falling at 3 dB per octave (amplitude at ``1 / sqrt(f)``), which is much
    closer to ambient room and microphone noise than white is.
``modulated``
    white noise under a slow random envelope, standing in for a busy background.
    This is *not* babble: there is no speech in it, only amplitude fluctuation,
    and calling it babble would overstate what an endpointer is being tested
    against.

Every generator is peak-normalised to one and driven by a caller-supplied
generator, so a noise condition is reproducible from a seed.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require
from echoscribe.types import FEATURE_DTYPE, FloatArray

#: Background noise kinds the laboratory can generate.
NOISE_KINDS = ("white", "pink", "modulated")


def white_noise(num_samples: int, rng: np.random.Generator) -> FloatArray:
    """Peak-normalised Gaussian white noise."""
    require_non_negative_count(num_samples)
    if num_samples == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    return _peak_normalise(rng.standard_normal(num_samples))


def pink_noise(num_samples: int, rng: np.random.Generator) -> FloatArray:
    """Peak-normalised noise with a ``1 / sqrt(f)`` amplitude spectrum.

    Shaped in the frequency domain rather than by cascading filters, because the
    target spectrum is then exact and the result does not depend on a warm-up
    transient at the start of the buffer.
    """
    require_non_negative_count(num_samples)
    if num_samples == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    bins = num_samples // 2 + 1
    shaped = rng.standard_normal(bins) + 1j * rng.standard_normal(bins)
    index = np.arange(bins, dtype=FEATURE_DTYPE)
    shaped = shaped / np.sqrt(np.maximum(index, 1.0))
    shaped[0] = 0.0  # no DC: a pink floor should not offset the waveform
    return _peak_normalise(np.fft.irfft(shaped, n=num_samples))


def require_non_negative_count(num_samples: int) -> None:
    """Shared guard for every generator's length argument."""
    require(
        isinstance(num_samples, int) and not isinstance(num_samples, bool) and num_samples >= 0,
        f"num_samples must be a non-negative integer, got {num_samples!r}",
        ConfigError,
    )


def _peak_normalise(values: np.ndarray) -> FloatArray:
    """Scale so the largest absolute sample is exactly one."""
    peak = float(np.max(np.abs(values)))
    return values / peak if peak > 1e-12 else np.asarray(values, dtype=FEATURE_DTYPE)
