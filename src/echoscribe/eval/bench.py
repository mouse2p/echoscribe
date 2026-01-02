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

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

from echoscribe.errors import ConfigError, SchemaError, require

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

    def to_dict(self) -> dict:
        """A JSON-ready mapping of the result, schema version included."""
        return {
            "schema": BENCH_SCHEMA_VERSION,
            "task": self.task,
            "audio_seconds": self.audio_seconds,
            "warmups": self.warmups,
            "samples": list(self.samples),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BenchResult:
        """Rebuild a result from :meth:`to_dict`, refusing foreign schemas."""
        version = payload.get("schema")
        require(
            version == BENCH_SCHEMA_VERSION,
            f"bench schema {version!r} is not supported; this build reads {BENCH_SCHEMA_VERSION}",
            SchemaError,
        )
        for name in ("task", "audio_seconds", "warmups", "samples"):
            require(name in payload, f"bench result is missing {name!r}", SchemaError)
        return cls(
            task=str(payload["task"]),
            audio_seconds=float(payload["audio_seconds"]),
            warmups=int(payload["warmups"]),
            samples=tuple(float(sample) for sample in payload["samples"]),
        )


def measure_rtf(
    task: str,
    run: Callable[[], object],
    audio_seconds: float,
    warmups: int = 1,
    repeats: int = 3,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchResult:
    """Time ``run`` and express the cost as wall seconds per audio second.

    ``run`` is called ``warmups + repeats`` times; only the last ``repeats``
    calls are timed.  Every sample is rounded where it is produced, to nine
    decimals, so a serialized result never depends on the repr precision of
    the machine that measured it.  The clock is a parameter, which is what
    allows the tests to drive the whole sequence without sleeping.
    """
    require(repeats >= 1, f"repeats must be at least 1, got {repeats!r}", ConfigError)
    require(warmups >= 0, f"warmups must not be negative, got {warmups!r}", ConfigError)
    require(
        audio_seconds > 0,
        f"audio_seconds must be positive, got {audio_seconds!r}",
        ConfigError,
    )
    for _ in range(warmups):
        run()
    samples: list[float] = []
    for _ in range(repeats):
        started = clock()
        run()
        samples.append(round(clock() - started, TIME_DECIMALS))
    return BenchResult(
        task=task, audio_seconds=audio_seconds, warmups=warmups, samples=tuple(samples)
    )


# What an RTF from this harness claims, and what it does not
# ----------------------------------------------------------
#
# It claims: on this machine, in this process, at this moment, the callable
# that was handed to ``measure_rtf`` consumed ``rtf`` wall seconds per second
# of audio, as a median over the recorded repeats.  That is all.
#
# It does not claim a hardware-independent throughput number.  RTF depends on
# the CPU, the core count the process happened to get, the numpy build, the
# Python version and whatever else was running.  A number measured here is
# comparable with another number measured the same way on the same machine --
# useful for "did this change make decoding slower", which is the question a
# benchmark in a test suite should answer -- and is not comparable with
# numbers from other machines or other harnesses.
#
# It also does not include time spent outside the callable: loading audio
# from disk before the measurement, or writing results after it, are the
# caller's responsibility to put inside ``run`` or consciously leave out.
