"""A convolutional room simulator.

Reverberation is modelled the way a small-room impulse response actually looks,
without solving anything: a direct path, a handful of early reflections at plausible
delays, and a diffuse tail whose envelope decays by 60 dB over ``rt60_ms``.  The
tail is Gaussian noise shaped by that envelope, which is the standard stand-in for
the late, statistically diffuse part of a response.

The exponential constant ``6.91`` is ``ln(1000)``: an envelope of
``exp(-6.91 * t / rt60)`` reaches ``1e-3`` of its initial value at exactly ``t =
rt60``, which is the definition of RT60 in amplitude terms.

This is not an image-source model and does not know the room's geometry.  It
produces a response with the right decay and a plausible direct-to-reverberant
ratio, which is all a feature-robustness test needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.seeds import DEFAULT_ROOT, Seed, new_rng
from echoscribe.types import FEATURE_DTYPE, FloatArray
from echoscribe.waveform import Audio

#: ``ln(1000)``: the decay constant that makes an envelope reach -60 dB at RT60.
RT60_DECAY_CONSTANT = math.log(1000.0)


@dataclass(frozen=True)
class RoomConfig:
    """How reverberant the simulated room is."""

    rt60_ms: float = 200.0
    direct_gain: float = 1.0
    tail_gain: float = 0.3
    early_reflections: int = 4
    seed: Seed = DEFAULT_ROOT

    def __post_init__(self) -> None:
        require_positive("rt60_ms", self.rt60_ms, ConfigError)
        require_positive("direct_gain", self.direct_gain, ConfigError)
        require_positive("tail_gain", self.tail_gain, ConfigError)
        require(
            isinstance(self.early_reflections, int) and self.early_reflections >= 0,
            f"early_reflections must be a non-negative integer, got {self.early_reflections!r}",
            ConfigError,
        )
        require(isinstance(self.seed, Seed), f"seed must be a Seed, got {self.seed!r}", ConfigError)


def decay_envelope(rt60_ms: float, sample_rate: int, num_samples: int) -> FloatArray:
    """The ``-60 dB per rt60`` exponential envelope of a diffuse tail."""
    require_positive("rt60_ms", rt60_ms, ConfigError)
    require_positive("sample_rate", sample_rate, ConfigError)
    require_non_negative_samples(num_samples)
    seconds = np.arange(num_samples, dtype=FEATURE_DTYPE) / sample_rate
    return np.exp(-RT60_DECAY_CONSTANT * seconds / (rt60_ms / 1000.0))


def require_non_negative_samples(num_samples: int) -> None:
    """Shared length guard."""
    require(
        isinstance(num_samples, int) and not isinstance(num_samples, bool) and num_samples >= 0,
        f"num_samples must be a non-negative integer, got {num_samples!r}",
        ConfigError,
    )


def impulse_response(
    config: RoomConfig, sample_rate: int, num_samples: int | None = None
) -> FloatArray:
    """Render the room's impulse response: direct path, early reflections, tail.

    The response starts with the direct path at ``config.direct_gain`` so that
    convolution does not delay the signal, and its length defaults to slightly more
    than one RT60 -- past that point the tail is below -60 dB and contributes
    nothing but cost.
    """
    require_positive("sample_rate", sample_rate, ConfigError)
    length = num_samples
    if length is None:
        length = int(math.ceil(config.rt60_ms / 1000.0 * sample_rate * 1.2)) + 1
    require_non_negative_samples(length)
    if length == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    response = np.zeros(max(length, 1), dtype=FEATURE_DTYPE)
    response[0] = config.direct_gain
    rng = new_rng(config.seed.fork("room"))
    if config.early_reflections:
        # Reflections land between 3 and 25 ms, each weaker than the last, which
        # is the shape a small room produces before the diffuse tail takes over.
        earliest = max(1, int(0.003 * sample_rate))
        latest = max(earliest + 1, int(0.025 * sample_rate))
        starts = rng.integers(earliest, latest, size=config.early_reflections)
        for index, delay in enumerate(sorted(int(value) for value in starts)):
            position = min(delay, length - 1)
            response[position] += config.direct_gain * 0.5 ** (index + 1)
    tail = rng.standard_normal(length) * decay_envelope(config.rt60_ms, sample_rate, length)
    return response + tail * config.tail_gain


def fft_convolve(signal: np.ndarray, kernel: np.ndarray) -> FloatArray:
    """Full linear convolution through the frequency domain.

    Zero-padded to the next power of two at least ``len(signal) + len(kernel) - 1``
    so the circular convolution the FFT computes is indistinguishable from the
    linear one.  For a 4000-tap room response this is orders of magnitude faster
    than ``numpy.convolve`` and, unlike an overlap-add implementation, has no block
    boundaries to get wrong.
    """
    x = np.asarray(signal, dtype=FEATURE_DTYPE)
    h = np.asarray(kernel, dtype=FEATURE_DTYPE)
    require(x.ndim == 1 and h.ndim == 1, "fft_convolve needs two 1-D inputs", ConfigError)
    if x.size == 0 or h.size == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    length = x.size + h.size - 1
    size = 1 << max(1, (length - 1).bit_length())
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), n=size)[:length]


def apply_room(audio: Audio, config: RoomConfig) -> Audio:
    """Return ``audio`` convolved with the room's impulse response.

    The result is longer than the input by ``len(response) - 1`` samples, which is
    what a linear convolution has to be: the reverb tail outlives the utterance.
    """
    require(audio.num_frames > 0, "cannot reverberate an empty waveform", ConfigError)
    response = impulse_response(config, audio.sample_rate)
    reverberant = fft_convolve(audio.samples.astype(FEATURE_DTYPE), response)
    return Audio(reverberant.astype(np.float32), audio.sample_rate)
