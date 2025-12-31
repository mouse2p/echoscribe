"""Real-time-factor measurement with warmup, repeats and honest statistics.

The real-time factor (RTF) is wall-clock processing time divided by the
duration of the audio processed: an RTF below 1.0 means the pipeline kept up
with the stream, and how far below says how much headroom one machine has.
It says nothing about any other machine -- see the module's closing notes for
what a number from here does and does not claim.

Measurements are taken with warmup runs that are thrown away, because the
first pass through a pipeline pays for imports, branch prediction and cache
fills that the steady state never sees.  Repeats are kept individually rather
than pre-averaged, so a report can show the spread and not just its middle.
The clock is injectable, which is what lets the tests below pin timing
behaviour exactly instead of sleeping and hoping.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from echoscribe.errors import ConfigError, require

BENCH_SCHEMA_VERSION = 1
TIME_DECIMALS = 9
RTF_DECIMALS = 6


@dataclass(frozen=True)
class TimingStatistics:
    """Minimum, median and mean over a sample of wall-clock seconds."""

    minimum: float
    median: float
    mean: float


def timing_statistics(samples: Sequence[float]) -> TimingStatistics:
    """Summarize timing samples.

    The samples are sorted before anything is summed, so the floating-point
    addition order never depends on the order measurements happened to arrive
    in.  Two runs over the same multiset of samples therefore produce
    bit-identical statistics, which a golden test can rely on.
    """
    require(len(samples) > 0, "timing statistics need at least one sample", ConfigError)
    values = tuple(sorted(float(sample) for sample in samples))
    return TimingStatistics(
        minimum=values[0],
        median=round(float(median(values)), TIME_DECIMALS),
        mean=round(sum(values) / len(values), TIME_DECIMALS),
    )


@dataclass(frozen=True)
class BenchResult:
    """One benchmarked task: how much wall time per second of audio.

    The headline number is the *median* sample divided by the audio duration.
    The median rather than the mean because a benchmark occasionally shares
    the machine with something else, and one stalled sample should move the
    reported factor as little as possible.  The fastest sample is exposed
    separately as ``rtf_best`` for the same reason.
    """

    task: str
    audio_seconds: float
    warmups: int
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        require(bool(self.task), "a bench result needs a task name", ConfigError)
        require(
            self.audio_seconds > 0,
            f"audio_seconds must be positive, got {self.audio_seconds!r}",
            ConfigError,
        )
        require(
            self.warmups >= 0,
            f"warmups must not be negative, got {self.warmups!r}",
            ConfigError,
        )
        require(len(self.samples) > 0, "a bench result needs at least one sample", ConfigError)
        require(
            all(sample >= 0 for sample in self.samples),
            "sample times must not be negative",
            ConfigError,
        )

    @property
    def repeats(self) -> int:
        """How many measured runs produced the samples."""
        return len(self.samples)

    @property
    def statistics(self) -> TimingStatistics:
        """Spread of the measured runs."""
        return timing_statistics(self.samples)

    @property
    def rtf(self) -> float:
        """Median wall seconds per audio second, rounded where it is produced."""
        return round(self.statistics.median / self.audio_seconds, RTF_DECIMALS)

    @property
    def rtf_best(self) -> float:
        """Fastest-sample RTF: the headroom this machine showed at its best."""
        return round(self.statistics.minimum / self.audio_seconds, RTF_DECIMALS)
