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

from dataclasses import dataclass

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
    total: np.ndarray | None = None
    squared_total: np.ndarray | None = None

    def __post_init__(self) -> None:
        require_positive("dimension", self.dimension, ConfigError)
        require(
            isinstance(self.count, int) and self.count >= 0,
            f"count must be a non-negative integer, got {self.count!r}",
            ConfigError,
        )
        for name in ("total", "squared_total"):
            value = getattr(self, name)
            if value is None:
                value = np.zeros(self.dimension, dtype=FEATURE_DTYPE)
            value = np.asarray(value, dtype=FEATURE_DTYPE)
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
