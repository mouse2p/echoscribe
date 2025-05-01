"""Lexical tone contours for the synthetic tone language.

A tone is an F0 trajectory, not a pitch value: the same syllable at tone 2 and
tone 4 differs in the *shape* of its fundamental over the syllable's duration.
Each contour here is stored as a handful of (normalised time, F0 multiplier)
control points and linearly interpolated between them, which keeps the
description small, exact and easy to test.

The multipliers are relative to a speaker's base F0, so the same contour works at
any pitch.  The four shapes follow the classic level / rising / dipping / falling
pattern; their exact control points are invented for this laboratory and are not
a transcription of any real speaker's productions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require, require_range


@dataclass(frozen=True)
class ToneContour:
    """An F0 multiplier trajectory over a normalised ``[0, 1]`` syllable."""

    index: int
    name: str
    control_points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        require(
            len(self.control_points) >= 2,
            f"tone {self.index} needs at least two points",
            ConfigError,
        )
        times = [point[0] for point in self.control_points]
        require(times[0] == 0.0, f"tone {self.index} must start at normalised time 0", ConfigError)
        require(times[-1] == 1.0, f"tone {self.index} must end at normalised time 1", ConfigError)
        require(
            all(later > earlier for earlier, later in zip(times, times[1:], strict=False)),
            f"tone {self.index} control times must strictly increase",
            ConfigError,
        )
        for time, multiplier in self.control_points:
            require_range(
                f"tone {self.index} multiplier at {time}", multiplier, 0.5, 2.0, ConfigError
            )

    def at(self, position: float) -> float:
        """F0 multiplier at normalised ``position``, clamped to ``[0, 1]``."""
        times = np.array([point[0] for point in self.control_points], dtype=np.float64)
        values = np.array([point[1] for point in self.control_points], dtype=np.float64)
        return float(np.interp(np.clip(position, 0.0, 1.0), times, values))

    def trajectory(self, samples: int) -> np.ndarray:
        """The multiplier at each of ``samples`` evenly spaced positions."""
        require(samples >= 1, f"a trajectory needs at least one sample, got {samples}", ConfigError)
        positions = np.arange(samples, dtype=np.float64) / max(samples - 1, 1)
        return np.interp(positions, self._times(), self._values())

    def _times(self) -> np.ndarray:
        return np.array([point[0] for point in self.control_points], dtype=np.float64)

    def _values(self) -> np.ndarray:
        return np.array([point[1] for point in self.control_points], dtype=np.float64)
