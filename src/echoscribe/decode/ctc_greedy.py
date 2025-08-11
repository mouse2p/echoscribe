"""CTC greedy decoding: argmax per frame, collapse repeats, drop blanks.

Greedy CTC is the reference decoder -- the one every more expensive search has to
agree with when the posterior matrix has an unambiguous answer.  Three rules define
it, and they are the reason CTC can emit a repeated symbol at all:

1. take the argmax symbol of every frame;
2. merge runs of the *same* symbol into one emission;
3. delete the blank symbols, but only after step 2.

The order of 2 and 3 is what makes ``a <blank> a`` decode to ``a a`` while ``a a``
decodes to ``a``.  Doing it the other way round would make the two indistinguishable
and no repeated word could ever be emitted.

Ties in the argmax resolve to the lowest symbol id, which ``numpy.argmax`` already
does.  That is a choice rather than a coincidence: it makes greedy decoding a
deterministic function of the posterior matrix, so a regression in a decoder shows
up as a changed symbol rather than a flickering one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.decode.posteriors import check_log_posteriors
from echoscribe.decode.tokens import DecodedToken
from echoscribe.errors import DecodeError, require


@dataclass(frozen=True)
class GreedyResult:
    """The collapsed token sequence, the raw argmax path and its path score."""

    tokens: tuple[DecodedToken, ...]
    path: tuple[int, ...]
    score: float

    @property
    def symbols(self) -> tuple[int, ...]:
        """The emitted symbols, blanks removed and repeats collapsed."""
        return tuple(token.symbol for token in self.tokens)

    @property
    def num_frames(self) -> int:
        """Frames the decode consumed."""
        return len(self.path)


def ctc_greedy(posteriors: np.ndarray, blank_id: int = 0) -> GreedyResult:
    """Decode a log-posterior matrix greedily."""
    matrix = check_log_posteriors(posteriors, "CTC posteriors")
    frames, symbols = matrix.shape
    require(
        0 <= blank_id < symbols,
        f"blank id {blank_id} is outside the {symbols}-symbol alphabet",
        DecodeError,
    )
    if frames == 0:
        return GreedyResult((), (), 0.0)
    path = matrix.argmax(axis=-1)
    rows = np.arange(frames)
    score = float(matrix[rows, path].sum())
    tokens: list[DecodedToken] = []
    index = 0
    while index < frames:
        symbol = int(path[index])
        end = index + 1
        while end < frames and int(path[end]) == symbol:
            end += 1
        if symbol != blank_id:
            tokens.append(DecodedToken(symbol, index, end, float(matrix[index:end, symbol].sum())))
        index = end
    return GreedyResult(tuple(tokens), tuple(int(value) for value in path), score)
