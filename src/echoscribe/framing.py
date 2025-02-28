"""Framing, windowing and pre-emphasis: the stage between a waveform and a spectrum.

A frame is a short slice of the signal over which the spectrum can be treated as
stationary.  Three numbers define the slicing, and getting them wrong silently
changes every downstream feature:

``frame_length``
    how much signal one frame covers.  25 ms is the usual choice: long enough for
    a reliable low-frequency estimate, short enough to stay stationary.
``frame_shift``
    how far the window advances.  10 ms gives the standard 100 frames per second.
``snip_edges``
    whether frames that would hang off either end are dropped (``True``) or
    zero-padded into existence (``False``).

``snip_edges=True`` never invents signal but drops up to a frame at each end, so
the frame count depends on the utterance length.  ``snip_edges=False`` gives
``ceil(n / shift)`` frames with the grid centred on the samples, which is what a
streaming decoder wants because frame ``m`` then always covers the samples that
arrived by ``m * shift``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from echoscribe.errors import (
    ConfigError,
    require,
    require_in,
    require_non_negative,
    require_positive,
    require_range,
)
from echoscribe.types import INDEX_DTYPE, IntArray
from echoscribe.windows import FRAME_WINDOWS

#: Policies for the first pre-emphasis sample, which has no predecessor.
PREEMPHASIS_EDGE_POLICIES = ("replicate", "zero")


def _samples_from_ms(milliseconds: float, sample_rate: int) -> int:
    """Convert milliseconds to whole samples, rounding halves away from zero.

    Python's ``round`` uses banker's rounding, so 25 ms at 44.1 kHz -- exactly
    1102.5 samples -- would come out as 1102 and silently shorten every frame in
    the corpus.  Rounding half up matches the published fbank recipes and keeps
    frame lengths comparable across sample rates.
    """
    return max(1, int(math.floor(milliseconds * sample_rate / 1000.0 + 0.5)))


@dataclass(frozen=True)
class FrameConfig:
    """Framing parameters, validated once at construction time."""

    sample_rate: int = 16000
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    window: str = "povey"
    preemphasis: float = 0.97
    remove_dc_offset: bool = True
    snip_edges: bool = True
    round_to_power_of_two: bool = True

    def __post_init__(self) -> None:
        require(
            isinstance(self.sample_rate, int)
            and not isinstance(self.sample_rate, bool)
            and self.sample_rate > 0,
            f"sample_rate must be a positive integer, got {self.sample_rate!r}",
            ConfigError,
        )
        require_positive("frame_length_ms", self.frame_length_ms, ConfigError)
        require_positive("frame_shift_ms", self.frame_shift_ms, ConfigError)
        require(
            self.frame_length_ms >= self.frame_shift_ms,
            f"frame_length_ms ({self.frame_length_ms}) must be at least frame_shift_ms "
            f"({self.frame_shift_ms})",
            ConfigError,
        )
        require_in("window", self.window, FRAME_WINDOWS, ConfigError)
        require_range("preemphasis", self.preemphasis, 0.0, 1.0, ConfigError)

    @property
    def frame_length(self) -> int:
        """Frame length in samples, rounded half up from ``frame_length_ms``."""
        return _samples_from_ms(self.frame_length_ms, self.sample_rate)

    @property
    def frame_shift(self) -> int:
        """Hop between consecutive frame starts, in samples."""
        return _samples_from_ms(self.frame_shift_ms, self.sample_rate)

    @property
    def fft_size(self) -> int:
        """FFT length the frames should be zero-padded to.

        Rounding up to a power of two is not cosmetic: it keeps the transform on
        the fast radix-2 path and makes the bin frequencies identical for every
        utterance, so features from different files stay comparable.
        """
        if not self.round_to_power_of_two:
            return self.frame_length
        return 1 << (self.frame_length - 1).bit_length()


def frame_count(num_samples: int, config: FrameConfig) -> int:
    """Number of frames ``num_samples`` produces under ``config``.

    With ``snip_edges`` this counts the frames fully inside the signal.  Without
    it, one frame per hop: the signal is conceptually padded by half a frame at
    each end, so a frame is emitted for every sample position the shift lands on.
    """
    require_non_negative("num_samples", num_samples)
    length, shift = config.frame_length, config.frame_shift
    if num_samples == 0:
        return 0
    if config.snip_edges:
        if num_samples < length:
            return 0
        return 1 + (num_samples - length) // shift
    return (num_samples + shift - 1) // shift


def frame_starts(num_samples: int, config: FrameConfig) -> IntArray:
    """Start index of every frame, relative to the *padded* signal.

    When ``snip_edges`` is false the padded signal has ``frame_length // 2`` zeros
    in front, and frame ``m`` starts at ``m * frame_shift`` in that padded buffer;
    in raw-signal coordinates that is ``m * frame_shift - frame_length // 2``.
    When ``snip_edges`` is true there is no padding and the two coincide.
    """
    count = frame_count(num_samples, config)
    return (np.arange(count, dtype=INDEX_DTYPE) * config.frame_shift).astype(INDEX_DTYPE)
