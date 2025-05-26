"""Cepstral mean and variance normalisation.

CMVN removes a channel's spectral tilt: a microphone, a codec or a room shifts and
scales every log-mel band by roughly a constant, and that shift is far larger than
the differences between phonemes.  Subtracting the mean and dividing by the
standard deviation, estimated over enough frames, cancels most of it.

Two estimators are provided because they answer to different constraints:

:data:`CmvnStatistics`
    a running sum and sum-of-squares over a whole corpus, so a trained model can be
    normalised with statistics that were not available at training time;
:func:`sliding_window_cmvn`
    a causal window over recent frames only, which is what a streaming decoder can
    actually compute -- the utterance mean does not exist yet when frame 10 arrives.

Both are exact for the frames they see; neither is a substitute for a matched
acoustic condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from echoscribe.errors import ConfigError, EvalError, require, require_positive
from echoscribe.types import FEATURE_DTYPE, FloatArray, require_rank

#: Smallest variance a normalisation will divide by.
VARIANCE_FLOOR = 1e-10


@dataclass
class CmvnStatistics:
    """Running count, sum and sum of squares of a feature stream.

    ``total`` and ``squared_total`` default to ``None`` and are filled with zeros
    of the right width during validation, so ``CmvnStatistics(dimension=80)`` is
    a usable empty accumulator.
    """

    dimension: int = 0
    count: int = 0
    total: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=FEATURE_DTYPE))
    squared_total: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=FEATURE_DTYPE))

    def __post_init__(self) -> None:
        require_positive("dimension", self.dimension, ConfigError)
        require(
            isinstance(self.count, int) and self.count >= 0,
            f"count must be a non-negative integer, got {self.count!r}",
            ConfigError,
        )
        # An empty accumulator means "not supplied", so it is filled to the right
        # width here; a caller-supplied array of the wrong width is an error.
        for name in ("total", "squared_total"):
            value = np.asarray(getattr(self, name), dtype=FEATURE_DTYPE)
            if value.size == 0:
                value = np.zeros(self.dimension, dtype=FEATURE_DTYPE)
            require(
                value.shape == (self.dimension,),
                f"{name} must have shape ({self.dimension},), got {value.shape}",
                ConfigError,
            )
            setattr(self, name, value)

    def update(self, features: np.ndarray) -> None:
        """Fold a block of frames into the statistics."""
        matrix = require_rank("features", np.asarray(features, dtype=FEATURE_DTYPE), 2)
        require(
            matrix.shape[1] == self.dimension,
            f"features have {matrix.shape[1]} columns but the statistics track {self.dimension}",
            EvalError,
        )
        self.total += matrix.sum(axis=0)
        self.squared_total += (matrix**2).sum(axis=0)
        self.count += matrix.shape[0]

    def mean(self) -> FloatArray:
        """Per-dimension mean, or zeros before any frame has been seen."""
        if self.count == 0:
            return np.zeros(self.dimension, dtype=FEATURE_DTYPE)
        return self.total / self.count

    def variance(self) -> FloatArray:
        """Per-dimension population variance, or zeros before any frame is seen."""
        if self.count == 0:
            return np.zeros(self.dimension, dtype=FEATURE_DTYPE)
        second = self.squared_total / self.count
        return np.maximum(second - self.mean() ** 2, 0.0)

    def merge(self, other: CmvnStatistics) -> CmvnStatistics:
        """Return the statistics of both streams concatenated.

        Merging is how per-utterance counts are combined into a corpus estimate
        without keeping every frame in memory.
        """
        require(
            self.dimension == other.dimension,
            f"cannot merge statistics of {self.dimension} and {other.dimension} dimensions",
            ConfigError,
        )
        return CmvnStatistics(
            dimension=self.dimension,
            count=self.count + other.count,
            total=self.total + other.total,
            squared_total=self.squared_total + other.squared_total,
        )


def apply_cmvn(
    features: np.ndarray,
    statistics: CmvnStatistics,
    normalize_variance: bool = True,
    variance_floor: float = VARIANCE_FLOOR,
) -> FloatArray:
    """Subtract the mean and optionally divide by the standard deviation.

    Empty statistics are a no-op rather than an error, so a caller can pass
    corpus statistics that happen not to cover a dimension yet.  A variance below
    ``variance_floor`` is raised to it, which keeps a dead band at zero instead of
    amplifying it into noise.
    """
    matrix = require_rank("features", np.asarray(features, dtype=FEATURE_DTYPE), 2)
    require_positive("variance_floor", variance_floor, ConfigError)
    require(
        statistics.dimension == matrix.shape[1],
        f"features have {matrix.shape[1]} columns but the statistics track {statistics.dimension}",
        EvalError,
    )
    centred = matrix - statistics.mean()
    if not normalize_variance:
        return centred
    scale = np.sqrt(np.maximum(statistics.variance(), variance_floor))
    return centred / scale


def sliding_window_cmvn(
    features: np.ndarray,
    window_frames: int = 300,
    normalize_variance: bool = True,
    variance_floor: float = VARIANCE_FLOOR,
) -> FloatArray:
    """Normalise each frame by the statistics of the ``window_frames`` before it.

    The window is *causal*: frame ``t`` only sees frames ``max(0, t - window + 1)``
    to ``t``.  That is what makes it usable while audio is still arriving, and it
    is also why the first frames are normalised by fewer samples than the rest --
    a stream has no past to borrow from.

    The running sums are computed with cumulative sums, so the cost is linear in
    the number of frames rather than quadratic in the window.
    """
    require(
        isinstance(window_frames, int)
        and not isinstance(window_frames, bool)
        and window_frames >= 1,
        f"window_frames must be a positive integer, got {window_frames!r}",
        ConfigError,
    )
    require_positive("variance_floor", variance_floor, ConfigError)
    matrix = require_rank("features", np.asarray(features, dtype=FEATURE_DTYPE), 2)
    frames, width = matrix.shape
    if frames == 0:
        return np.zeros_like(matrix)
    counts = np.minimum(np.arange(frames) + 1, window_frames).astype(FEATURE_DTYPE)[:, None]
    zeros = np.zeros((1, width), dtype=FEATURE_DTYPE)
    # prefix[t] sums the frames strictly before t, so the causal window ending at
    # t starts at prefix[max(0, t - window + 1)].
    prefix = np.vstack([zeros, np.cumsum(matrix, axis=0)[:-1]])
    prefix_squared = np.vstack([zeros, np.cumsum(matrix**2, axis=0)[:-1]])
    running = np.cumsum(matrix, axis=0)
    running_squared = np.cumsum(matrix**2, axis=0)
    start = np.clip(np.arange(frames) - window_frames + 1, 0, None)
    mean = (running - prefix[start]) / counts
    second = (running_squared - prefix_squared[start]) / counts
    variance = np.maximum(second - mean**2, 0.0)
    centred = matrix - mean
    if not normalize_variance:
        return centred
    return centred / np.sqrt(np.maximum(variance, variance_floor))


#: Bumped when the serialized shape of :class:`CmvnStatistics` changes.
STATISTICS_SCHEMA_VERSION = 1


def statistics_to_dict(statistics: CmvnStatistics) -> dict:
    """Render statistics as a JSON-ready mapping, including the schema version."""
    return {
        "schema": STATISTICS_SCHEMA_VERSION,
        "dimension": statistics.dimension,
        "count": statistics.count,
        # Rounded here, at the point the value is produced, so a serialized
        # statistics file reads the same on every interpreter.
        "total": [round(float(value), 12) for value in statistics.total],
        "squared_total": [round(float(value), 12) for value in statistics.squared_total],
    }


def statistics_from_dict(payload: dict) -> CmvnStatistics:
    """Rebuild statistics from :func:`statistics_to_dict`, rejecting other schemas."""
    version = payload.get("schema")
    require(
        version == STATISTICS_SCHEMA_VERSION,
        f"CMVN statistics schema {version!r} is not supported; this build reads "
        f"{STATISTICS_SCHEMA_VERSION}",
        ConfigError,
    )
    for name in ("dimension", "count", "total", "squared_total"):
        require(name in payload, f"CMVN statistics are missing {name!r}", ConfigError)
    return CmvnStatistics(
        dimension=int(payload["dimension"]),
        count=int(payload["count"]),
        total=np.asarray(payload["total"], dtype=FEATURE_DTYPE),
        squared_total=np.asarray(payload["squared_total"], dtype=FEATURE_DTYPE),
    )
