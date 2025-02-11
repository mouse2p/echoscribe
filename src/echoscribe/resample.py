"""Sample-rate conversion by polyphase filtering and by linear interpolation.

Two methods are offered because they have genuinely different trade-offs, and
the synthetic laboratory needs both:

``polyphase``
    A windowed-sinc FIR applied through the polyphase decomposition of the exact
    rational ratio ``target / source``.  Band-limited, so a tone keeps its
    frequency and no aliasing is introduced, at the cost of designing a filter
    and accepting pass-band ripple that depends on the window.

``linear``
    Interpolation between the two nearest input samples.  Cheap, endpoint
    preserving and free of ringing, but it applies no anti-alias filter, so
    energy above the target Nyquist folds back into the result.  It exists so a
    caller can measure what is lost by not filtering.

Both are pure NumPy and both are deterministic for a given input.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from echoscribe.errors import ConfigError, require, require_in, require_positive
from echoscribe.types import SAMPLE_DTYPE, SampleArray


def rational_factors(source_rate: int, target_rate: int) -> tuple[int, int]:
    """Reduce ``target / source`` to the coprime pair ``(up, down)``.

    ``44100 -> 16000`` gives ``(160, 441)``; ``16000 -> 8000`` gives ``(1, 2)``.
    The polyphase decomposition indexes by this pair, so it has to be exact
    rather than a float ratio.
    """
    for name, rate in (("source_rate", source_rate), ("target_rate", target_rate)):
        require(
            isinstance(rate, int) and not isinstance(rate, bool) and rate > 0,
            f"{name} must be a positive integer, got {rate!r}",
            ConfigError,
        )
    divisor = gcd(int(source_rate), int(target_rate))
    return int(target_rate) // divisor, int(source_rate) // divisor


#: Window functions accepted by :func:`design_lowpass_fir`.
FILTER_WINDOWS = ("rectangular", "hamming", "hann", "blackman")


def window_vector(kind: str, length: int) -> np.ndarray:
    """Return a symmetric window of ``length`` samples.

    ``hann`` and ``hamming`` differ only at the edges: Hamming keeps a small
    non-zero endpoint, trading a slightly wider transition band for a lower
    first side lobe.
    """
    require_positive("length", length)
    require_in("window", kind, FILTER_WINDOWS, ConfigError)
    if kind == "rectangular":
        return np.ones(length, dtype=np.float64)
    if kind == "hamming":
        return np.hamming(length).astype(np.float64)
    if kind == "hann":
        return np.hanning(length).astype(np.float64)
    return np.blackman(length).astype(np.float64)


def design_lowpass_fir(num_taps: int, cutoff: float, window: str = "hamming") -> np.ndarray:
    """Design a windowed-sinc lowpass filter with unit gain at DC.

    ``cutoff`` is a fraction of the Nyquist frequency of the rate the filter runs
    at, so ``0.45`` keeps everything below 45% of Nyquist.  The coefficients are
    normalised to sum to one, which makes the filter amplitude-preserving for
    signals well inside the pass band.
    """
    require(
        isinstance(num_taps, int) and num_taps >= 1,
        f"num_taps must be a positive integer, got {num_taps!r}",
        ConfigError,
    )
    require(
        0.0 < cutoff < 1.0,
        f"cutoff must lie in (0, 1) of Nyquist, got {cutoff!r}",
        ConfigError,
    )
    centre = (num_taps - 1) / 2.0
    offsets = np.arange(num_taps, dtype=np.float64) - centre
    taps = cutoff * np.sinc(cutoff * offsets) * window_vector(window, num_taps)
    return taps / taps.sum()


def linear_resample(samples: np.ndarray, source_rate: int, target_rate: int) -> SampleArray:
    """Resample by interpolating between the two nearest input samples.

    The first and last input samples map onto the first and last output samples,
    so the output duration matches the input duration even for short buffers.
    That is the trade-off: no anti-alias filter is applied, so energy above the
    target Nyquist folds back into the result.
    """
    up, down = rational_factors(source_rate, target_rate)
    x = np.asarray(samples, dtype=np.float64)
    require(x.ndim == 1, f"linear_resample needs 1-D samples, got shape {x.shape}", ConfigError)
    count = x.size
    if count == 0:
        return np.zeros(0, dtype=SAMPLE_DTYPE)
    length = max(1, int(round(count * up / down)))
    if length == 1 or count == 1:
        return np.full(length, x[0], dtype=SAMPLE_DTYPE)
    positions = np.arange(length, dtype=np.float64) * ((count - 1) / (length - 1))
    left = np.floor(positions).astype(np.intp)
    fraction = positions - left
    right = np.minimum(left + 1, count - 1)
    mixed = x[left] * (1.0 - fraction) + x[right] * fraction
    return mixed.astype(SAMPLE_DTYPE)
