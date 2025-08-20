"""The output records every decoder in this package produces.

CTC and transducer decoding differ in how they walk the frame axis, but they agree
on what they hand back: a sequence of symbols, each with the frames that produced it
and the log-probability accumulated over those frames.  Keeping that in one record
is what lets an alignment, a confidence estimate and a latency measurement all read
the same structure.

``start`` is inclusive and ``end`` is exclusive, both in frames.  A token's span is
therefore never empty, and the spans of a CTC greedy decode tile the frame axis
exactly -- including the blank runs, which are dropped from the token list but
still occupy time.  That last point matters for latency: the time a word finishes is
the end of its last token's span, not the end of the utterance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from echoscribe.errors import DecodeError, require, require_non_negative

#: Accumulated log-domain sums can land a hair above zero through rounding: a
#: hypothesis whose probability is 1.0 to within this margin is legal, and rejecting
#: it would turn a correct decode into a crash.
LOG_PROBABILITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DecodedToken:
    """One emitted symbol, the frames it came from, and its log-probability."""

    symbol: int
    start: int
    end: int
    score: float = 0.0

    def __post_init__(self) -> None:
        require_non_negative("symbol", self.symbol, DecodeError)
        require(
            isinstance(self.start, int) and isinstance(self.end, int),
            f"frame bounds must be integers, got ({self.start!r}, {self.end!r})",
            DecodeError,
        )
        require(
            0 <= self.start <= self.end,
            f"frame span ({self.start}, {self.end}) is not a valid half-open range",
            DecodeError,
        )
        require(
            self.score <= LOG_PROBABILITY_TOLERANCE,
            f"a log-probability cannot be positive, got {self.score}",
            DecodeError,
        )

    @property
    def num_frames(self) -> int:
        """How many frames this token spans."""
        return self.end - self.start

    def times(self, frame_shift_ms: float = 10.0) -> tuple[float, float]:
        """Start and end time in seconds for this token's span."""
        shift = frame_shift_ms / 1000.0
        return self.start * shift, self.end * shift


def symbols_of(tokens: Sequence[DecodedToken]) -> tuple[int, ...]:
    """The symbol sequence alone, which is what a scorer compares."""
    return tuple(token.symbol for token in tokens)


def total_score(tokens: Sequence[DecodedToken]) -> float:
    """Sum of the per-token log-probabilities."""
    return float(sum(token.score for token in tokens))


def span_gaps(tokens: Sequence[DecodedToken]) -> tuple[int, ...]:
    """Frame counts of the gaps *between* consecutive tokens.

    For a CTC decode these are the blank runs that separate repeated symbols; a
    gap of zero means two tokens were emitted from adjacent frames.
    """
    return tuple(
        later.start - earlier.end for earlier, later in zip(tokens, tokens[1:], strict=False)
    )
