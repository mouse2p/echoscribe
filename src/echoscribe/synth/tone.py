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


#: The four lexical tones of the synthetic tone language.
TONES: tuple[ToneContour, ...] = (
    ToneContour(1, "high-level", ((0.0, 1.12), (1.0, 1.1))),
    ToneContour(2, "rising", ((0.0, 0.86), (0.45, 1.0), (1.0, 1.26))),
    ToneContour(3, "dipping", ((0.0, 0.98), (0.35, 0.78), (0.7, 0.86), (1.0, 1.12))),
    ToneContour(4, "falling", ((0.0, 1.3), (0.55, 1.0), (1.0, 0.78))),
)

#: Tone indices keyed for lookup, plus the neutral tone used in unstressed syllables.
TONE_INDICES = tuple(tone.index for tone in TONES)
NEUTRAL_TONE = ToneContour(0, "neutral", ((0.0, 0.95), (1.0, 0.9)))


def tone_by_index(index: int) -> ToneContour:
    """Return the contour for a lexical tone index, including the neutral tone."""
    for tone in (*TONES, NEUTRAL_TONE):
        if tone.index == index:
            return tone
    raise ConfigError(
        f"unknown tone index {index}; the laboratory defines {NEUTRAL_TONE.index} "
        f"(neutral) and {TONE_INDICES}"
    )
