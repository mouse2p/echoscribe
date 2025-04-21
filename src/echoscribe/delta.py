"""Delta and delta-delta features.

A delta is the local slope of a feature trajectory, estimated by least-squares
regression over a symmetric window rather than by a plain first difference.  The
regression form

    ``d[t] = sum_{n=1..N} n * (x[t + n] - x[t - n]) / (2 * sum_{n=1..N} n^2)``

is what the classic recipe uses: it is the slope of the line fitted to the
``2N + 1`` frames around ``t``, which suppresses the frame-to-frame noise a raw
difference amplifies.

Frames near either end replicate the nearest available sample, so a delta is
defined for every frame the input has.  That keeps the output length equal to the
input length, which matters when deltas are concatenated with statics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require
from echoscribe.types import FEATURE_DTYPE, FloatArray, require_rank


@dataclass(frozen=True)
class DeltaConfig:
    """How many derivative orders to compute and over what window."""

    window: int = 2
    order: int = 2

    def __post_init__(self) -> None:
        require(
            isinstance(self.window, int) and not isinstance(self.window, bool) and self.window >= 1,
            f"window must be a positive integer, got {self.window!r}",
            ConfigError,
        )
        require(
            isinstance(self.order, int) and not isinstance(self.order, bool) and self.order >= 1,
            f"order must be a positive integer, got {self.order!r}",
            ConfigError,
        )

    @property
    def denominator(self) -> float:
        """``2 * sum n^2`` over the regression window."""
        return 2.0 * sum(float(n * n) for n in range(1, self.window + 1))

    @property
    def stacked_width_factor(self) -> int:
        """How many times wider the stacked features are than the statics."""
        return self.order + 1


def delta_features(features: np.ndarray, config: DeltaConfig | None = None) -> FloatArray:
    """Return the first derivative of every feature trajectory."""
    settings = DeltaConfig() if config is None else config
    matrix = require_rank("features", np.asarray(features, dtype=FEATURE_DTYPE), 2)
    frames, dimension = matrix.shape
    if frames == 0:
        return np.zeros_like(matrix)
    padded = np.pad(matrix, ((settings.window, settings.window), (0, 0)), mode="edge")
    total = np.zeros((frames, dimension), dtype=FEATURE_DTYPE)
    for offset in range(1, settings.window + 1):
        ahead = padded[settings.window + offset :][:frames]
        behind = padded[settings.window - offset :][:frames]
        total += offset * (ahead - behind)
    return total / settings.denominator
