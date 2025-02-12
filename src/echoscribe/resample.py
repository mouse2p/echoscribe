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

from dataclasses import dataclass
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


def polyphase_filter_bank(
    up: int,
    down: int,
    taps_per_phase: int = 16,
    window: str = "hamming",
    cutoff: float = 0.9,
) -> np.ndarray:
    """Build the ``(up, taps_per_phase)`` polyphase decomposition of the filter.

    The prototype lowpass runs at the ``up``-sampled rate, so its cutoff is
    divided by ``max(up, down)``: that places the transition band below the
    Nyquist of whichever rate is lower, which is exactly where aliasing starts.
    Multiplying by ``up`` compensates for the zeros the upsampler inserts.
    """
    require(
        isinstance(taps_per_phase, int) and taps_per_phase >= 1,
        f"taps_per_phase must be a positive integer, got {taps_per_phase!r}",
        ConfigError,
    )
    prototype = design_lowpass_fir(taps_per_phase * up, cutoff / max(up, down), window) * up
    return prototype.reshape(taps_per_phase, up).T


def polyphase_resample(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
    taps_per_phase: int = 16,
    window: str = "hamming",
    cutoff: float = 0.9,
) -> SampleArray:
    """Resample by exact rational polyphase filtering.

    The output length is ``ceil(n * up / down)``, so every input sample that a
    output slot would read is covered and none is dropped silently.  The filter
    contributes a group delay of ``(taps_per_phase - 1) / 2`` output samples,
    which callers measuring latency need to account for.
    """
    up, down = rational_factors(source_rate, target_rate)
    x = np.asarray(samples, dtype=np.float64)
    require(x.ndim == 1, f"polyphase_resample needs 1-D samples, got shape {x.shape}", ConfigError)
    count = x.size
    if count == 0:
        return np.zeros(0, dtype=SAMPLE_DTYPE)
    phases = polyphase_filter_bank(up, down, taps_per_phase, window, cutoff)
    length = -(-count * up // down)
    padded = np.concatenate([np.zeros(taps_per_phase - 1), x, np.zeros(taps_per_phase - 1)])
    slots = np.arange(length, dtype=np.int64) * down
    phase = slots % up
    base = (slots - phase) // up
    # Walk the taps backwards so gather column 0 pairs with tap 0.
    offsets = np.arange(taps_per_phase - 1, -1, -1, dtype=np.int64)
    gather = padded[base[:, None] + offsets[None, :]]
    return np.sum(phases[phase] * gather, axis=1).astype(SAMPLE_DTYPE)


#: Conversion methods accepted by :class:`ResampleConfig`.
RESAMPLE_METHODS = ("polyphase", "linear")


@dataclass(frozen=True)
class ResampleConfig:
    """How to convert a waveform to another sample rate."""

    target_rate: int = 16000
    method: str = "polyphase"
    taps_per_phase: int = 16
    window: str = "hamming"
    cutoff: float = 0.9

    def __post_init__(self) -> None:
        require(
            isinstance(self.target_rate, int)
            and not isinstance(self.target_rate, bool)
            and self.target_rate > 0,
            f"target_rate must be a positive integer, got {self.target_rate!r}",
            ConfigError,
        )
        require_in("method", self.method, RESAMPLE_METHODS, ConfigError)
        require(
            isinstance(self.taps_per_phase, int) and self.taps_per_phase >= 1,
            f"taps_per_phase must be a positive integer, got {self.taps_per_phase!r}",
            ConfigError,
        )
        require_in("window", self.window, FILTER_WINDOWS, ConfigError)
        require(
            0.0 < self.cutoff < 1.0,
            f"cutoff must lie in (0, 1) of Nyquist, got {self.cutoff!r}",
            ConfigError,
        )
