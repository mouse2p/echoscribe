"""Spectral transforms of framed signals.

The convention here is a real FFT with ``fft_size`` points per frame and power
normalised by ``fft_size``:

    ``power[k] = (re[k]**2 + im[k]**2) / fft_size``

Dividing by ``fft_size`` is what makes results comparable across configurations.
Without it, zero-padding a 400-sample frame to 512 points would inflate every
peak by 512/400 and a mel filterbank tuned on one setting would silently change
scale under another.  With it, a unit cosine landing exactly on bin ``k`` has
power ``fft_size / 4`` at that bin regardless of padding.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.numeric import safe_log
from echoscribe.types import FEATURE_DTYPE, FloatArray, require_rank


def rfft_bin_count(fft_size: int) -> int:
    """Number of bins a real FFT of ``fft_size`` points produces."""
    require(
        isinstance(fft_size, int) and not isinstance(fft_size, bool) and fft_size >= 1,
        f"fft_size must be a positive integer, got {fft_size!r}",
        ConfigError,
    )
    return fft_size // 2 + 1


def power_spectrum(frames: np.ndarray, fft_size: int | None = None) -> FloatArray:
    """Return the normalised power spectrum of every frame.

    ``fft_size=None`` uses the frame width itself.  A larger ``fft_size``
    zero-pads, which interpolates the spectrum without adding information; a
    smaller one is rejected rather than truncating the frame.
    """
    matrix = require_rank("frames", np.asarray(frames, dtype=FEATURE_DTYPE), 2)
    size = _resolve_fft_size(matrix, fft_size)
    transform = np.fft.rfft(matrix, n=size, axis=1)
    return (transform.real**2 + transform.imag**2) / size


#: Floor applied before taking the logarithm of a power spectrum.  Powers can be
#: exactly zero -- a silent frame, or a bin a pure tone leaves empty -- and the
#: log of zero would poison every later sum.
SPECTRAL_FLOOR = 1e-20


def bin_frequencies(fft_size: int, sample_rate: int) -> FloatArray:
    """Centre frequency in Hz of every bin of a real FFT of ``fft_size`` points.

    The last bin is the Nyquist frequency, ``sample_rate / 2``, and the spacing is
    ``sample_rate / fft_size``, which is the resolution the frame length buys.
    """
    require_positive("sample_rate", sample_rate, ConfigError)
    require_positive("fft_size", fft_size, ConfigError)
    return np.asarray(np.fft.rfftfreq(fft_size, d=1.0 / sample_rate), dtype=np.float64)


def log_power_spectrum(
    frames: np.ndarray,
    fft_size: int | None = None,
    floor: float = SPECTRAL_FLOOR,
) -> FloatArray:
    """Return ``log`` of the power spectrum, floored so silence stays finite."""
    return safe_log(power_spectrum(frames, fft_size), floor)


def complex_spectrum(frames: np.ndarray, fft_size: int | None = None) -> np.ndarray:
    """Return the raw one-sided complex transform of every frame."""
    matrix = require_rank("frames", np.asarray(frames, dtype=FEATURE_DTYPE), 2)
    size = _resolve_fft_size(matrix, fft_size)
    return np.fft.rfft(matrix, n=size, axis=1)


def magnitude_spectrum(frames: np.ndarray, fft_size: int | None = None) -> FloatArray:
    """Return ``|X[k]| / sqrt(fft_size)`` so magnitude squared is the power."""
    matrix = require_rank("frames", np.asarray(frames, dtype=FEATURE_DTYPE), 2)
    size = _resolve_fft_size(matrix, fft_size)
    return np.abs(np.fft.rfft(matrix, n=size, axis=1)) / np.sqrt(float(size))


def phase_spectrum(frames: np.ndarray, fft_size: int | None = None) -> FloatArray:
    """Return the unwrapped-free principal phase of every bin, in radians."""
    return np.angle(complex_spectrum(frames, fft_size))


def _resolve_fft_size(matrix: np.ndarray, fft_size: int | None) -> int:
    """Share the padding rules between the power, magnitude and phase paths."""
    size = matrix.shape[1] if fft_size is None else fft_size
    require_positive("fft_size", size, ConfigError)
    require(
        size >= matrix.shape[1],
        f"fft_size {size} is smaller than the frame width {matrix.shape[1]}",
        ConfigError,
    )
    return int(size)
