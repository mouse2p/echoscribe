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
from echoscribe.seeds import DEFAULT_ROOT, Seed
from echoscribe.types import FEATURE_DTYPE, FloatArray

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
