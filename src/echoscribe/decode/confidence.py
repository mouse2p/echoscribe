"""Confidence estimation from posterior marginals, and its calibration.

A decoder's score is a *ranking* number: it says which hypothesis won, not how
likely that hypothesis is to be right.  Confidence answers the second question, and
the only honest way to answer it from a frame-synchronous model is to read the
posterior the model actually produced for the symbol it emitted.

Two numbers are reported per token, because they fail in different ways:

``probability``
    the geometric mean of the emitted symbol's posterior over the frames that
    produced it.  Geometric rather than arithmetic because a single frame with near
    zero posterior should dominate -- and should, since one frame of doubt is enough
    to make the symbol wrong.
``margin``
    the gap between the best posterior on the token's strongest frame and the
    emitted symbol's.  A margin of zero means the decoder picked a symbol that tied
    with a better one, which no amount of averaging over frames will reveal.

Neither is a probability of correctness.  That is what the calibration machinery
below measures, and the measurement is the point: a confidence number that has never
been calibrated against known-correct labels is a ranking score wearing a nicer hat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from echoscribe.decode.posteriors import check_log_posteriors
from echoscribe.decode.tokens import DecodedToken
from echoscribe.errors import DecodeError, require
from echoscribe.types import FEATURE_DTYPE

#: Smallest posterior that survives the logarithm used for a geometric mean.
CONFIDENCE_FLOOR = 1e-300


@dataclass(frozen=True)
class TokenConfidence:
    """How sure the decoder's own posteriors were about one emitted symbol."""

    symbol: int
    probability: float
    rank: int
    margin: float
    frame: int

    def __post_init__(self) -> None:
        require(
            -1e-9 <= self.probability <= 1.0 + 1e-9,
            f"a confidence must lie in [0, 1], got {self.probability!r}",
            DecodeError,
        )
        require(self.rank >= 0, f"rank must not be negative, got {self.rank}", DecodeError)
        require(
            self.margin >= -1e-9, f"margin must not be negative, got {self.margin}", DecodeError
        )
        require(self.frame >= 0, f"frame must not be negative, got {self.frame}", DecodeError)


def token_confidences(
    posteriors: np.ndarray, tokens: Sequence[DecodedToken]
) -> tuple[TokenConfidence, ...]:
    """One confidence per decoded token, read from the frames that produced it.

    A zero-width span -- two labels emitted from one frame, which a transducer does
    routinely -- is treated as covering its start frame, because that is the frame
    whose posterior decided it.
    """
    matrix = check_log_posteriors(posteriors, "posteriors")
    frames = matrix.shape[0]
    if frames == 0:
        return ()
    probabilities = np.exp(matrix)
    result: list[TokenConfidence] = []
    for token in tokens:
        require(
            0 <= token.start < frames,
            f"token frame {token.start} is outside the {frames}-frame posterior matrix",
            DecodeError,
        )
        require(
            token.symbol < matrix.shape[1],
            f"symbol {token.symbol} is outside the {matrix.shape[1]}-symbol alphabet",
            DecodeError,
        )
        last = max(min(token.end, frames), token.start + 1)
        window = probabilities[token.start : last, token.symbol]
        probability = float(np.exp(np.mean(np.log(np.maximum(window, CONFIDENCE_FLOOR)))))
        strongest = token.start + int(np.argmax(window))
        row = probabilities[strongest]
        value = float(row[token.symbol])
        result.append(
            TokenConfidence(
                symbol=token.symbol,
                probability=min(probability, 1.0),
                rank=int(np.count_nonzero(row > value)),
                margin=float(row.max()) - value,
                frame=strongest,
            )
        )
    return tuple(result)


def probabilities_of(confidences: Sequence[TokenConfidence]) -> tuple[float, ...]:
    """The probability component of each confidence, as a plain tuple."""
    return tuple(item.probability for item in confidences)


def geometric_mean(values: Sequence[float]) -> float:
    """Geometric mean of values in ``[0, 1]``; 1.0 for an empty sequence.

    Computed through the logarithm so a single near-zero value pulls the result down
    instead of underflowing the product to zero.
    """
    require(
        all(0.0 <= value <= 1.0 for value in values),
        "a geometric mean of confidences needs values in [0, 1]",
        DecodeError,
    )
    if not values:
        return 1.0
    logs = np.log(np.maximum(np.asarray(values, dtype=FEATURE_DTYPE), CONFIDENCE_FLOOR))
    return float(np.exp(logs.mean()))


def utterance_confidence(confidences: Sequence[TokenConfidence]) -> float:
    """One number for a whole hypothesis: the geometric mean of its token scores.

    An empty hypothesis scores 0.0 rather than the 1.0 :func:`geometric_mean` returns
    for an empty product, because "no words" is not a confident transcription.
    """
    if not confidences:
        return 0.0
    return geometric_mean(probabilities_of(confidences))


@dataclass(frozen=True)
class CalibrationBin:
    """One bucket of a reliability diagram."""

    index: int
    low: float
    high: float
    count: int
    correct: int
    confidence_total: float

    @property
    def accuracy(self) -> float:
        """Fraction of the bucket's tokens that were correct; 0.0 when empty."""
        return self.correct / self.count if self.count else 0.0

    @property
    def mean_confidence(self) -> float:
        """Mean reported confidence in the bucket; 0.0 when empty."""
        return self.confidence_total / self.count if self.count else 0.0

    @property
    def is_empty(self) -> bool:
        """True when no token fell in this bucket."""
        return self.count == 0


def calibration_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    num_bins: int = 10,
) -> tuple[CalibrationBin, ...]:
    """Bucket ``(confidence, was-it-correct)`` pairs into a reliability diagram.

    Buckets are half-open on the right except the last, so a confidence of exactly
    1.0 lands in the top bucket rather than off the end.  Empty buckets are kept in
    the result -- dropping them would make the diagram's x-axis lie about which
    confidence ranges were never observed.
    """
    require(
        isinstance(num_bins, int) and not isinstance(num_bins, bool) and num_bins >= 1,
        f"num_bins must be a positive integer, got {num_bins!r}",
        DecodeError,
    )
    require(
        len(confidences) == len(correct),
        f"{len(confidences)} confidences cannot be paired with {len(correct)} labels",
        DecodeError,
    )
    require(
        all(0.0 <= value <= 1.0 for value in confidences),
        "a confidence must lie in [0, 1]",
        DecodeError,
    )
    counts = [0] * num_bins
    hits = [0] * num_bins
    totals = [0.0] * num_bins
    for value, was_correct in zip(confidences, correct, strict=True):
        index = min(int(value * num_bins), num_bins - 1)
        counts[index] += 1
        totals[index] += float(value)
        if was_correct:
            hits[index] += 1
    return tuple(
        CalibrationBin(
            index,
            index / num_bins,
            (index + 1) / num_bins,
            counts[index],
            hits[index],
            totals[index],
        )
        for index in range(num_bins)
    )


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float:
    """Count-weighted mean gap between reported confidence and accuracy.

    Zero means every bucket reported exactly the accuracy it achieved.  The number is
    only meaningful alongside the bucket counts: an ECE over ten tokens says almost
    nothing, and a confident-looking ECE over an empty set says nothing at all.
    """
    total = sum(item.count for item in bins)
    if total == 0:
        return 0.0
    return float(
        sum(item.count * abs(item.accuracy - item.mean_confidence) for item in bins if item.count)
        / total
    )


def reliability_curve(bins: Sequence[CalibrationBin]) -> tuple[tuple[float, float], ...]:
    """``(mean confidence, accuracy)`` for every non-empty bucket, in bucket order.

    Empty buckets are dropped here and only here: a reliability *diagram* has no
    point to draw for a range that was never observed, while the bucket list itself
    keeps them so a reader can see the gaps.
    """
    return tuple((item.mean_confidence, item.accuracy) for item in bins if not item.is_empty)


def bucket_counts(bins: Sequence[CalibrationBin]) -> tuple[int, ...]:
    """How many tokens fell in each bucket, including the empty ones."""
    return tuple(item.count for item in bins)
