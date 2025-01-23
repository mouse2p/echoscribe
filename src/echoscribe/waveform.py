"""The waveform value object shared by every audio-facing entry point.

``Audio`` is deliberately dumb.  It holds mono ``float32`` samples in
``[-1.0, 1.0]`` plus the sample rate they belong to, and offers only
measurements and structural operations.  Anything that changes the signal --
resampling, filtering, noise mixing -- lives in its own module and returns a new
``Audio`` instead of mutating one, which is what makes feature extraction
reproducible from a waveform object.

Attribute-style access (``duration``, ``num_frames``) is used for values that are
free to compute; method calls (``peak()``, ``rms()`) are used for reductions that
walk the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.errors import AudioError, require
from echoscribe.types import SampleArray, as_samples


@dataclass(frozen=True)
class Audio:
    """An immutable mono waveform and the sample rate it was captured at."""

    samples: SampleArray
    sample_rate: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", as_samples(self.samples, "audio samples"))
        require(
            isinstance(self.sample_rate, int)
            and not isinstance(self.sample_rate, bool)
            and self.sample_rate > 0,
            f"sample rate must be a positive integer, got {self.sample_rate!r}",
            AudioError,
        )

    @property
    def num_frames(self) -> int:
        """Number of samples in the buffer."""
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """Length in seconds; exactly zero for an empty buffer."""
        return self.num_frames / self.sample_rate

    def peak(self) -> float:
        """Largest absolute sample value, or 0.0 for an empty buffer."""
        if self.num_frames == 0:
            return 0.0
        return float(np.max(np.abs(self.samples.astype(np.float64))))

    def rms(self) -> float:
        """Root-mean-square level, or 0.0 for an empty buffer."""
        if self.num_frames == 0:
            return 0.0
        squared = self.samples.astype(np.float64) ** 2
        return float(np.sqrt(np.mean(squared)))

    def is_silent(self, threshold: float = 1e-6) -> bool:
        """True when no sample exceeds ``threshold`` in absolute value."""
        require(threshold >= 0.0, f"silence threshold must not be negative, got {threshold!r}")
        return self.peak() <= threshold

    def slice(self, start: int, end: int) -> Audio:
        """Return the half-open sample range ``[start, end)`` as a new Audio.

        Indices follow Python semantics: ``end`` may equal ``num_frames`` and an
        inverted range is an error rather than an empty result, because a caller
        asking for ``slice(50, 10)`` has a bug worth surfacing.
        """
        require(
            0 <= start <= end <= self.num_frames,
            f"slice ({start}, {end}) is outside [0, {self.num_frames}]",
            AudioError,
        )
        return Audio(self.samples[start:end], self.sample_rate)
