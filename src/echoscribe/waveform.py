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
