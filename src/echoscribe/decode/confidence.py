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
