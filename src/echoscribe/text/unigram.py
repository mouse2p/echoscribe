"""A unigram language-model subword tokenizer.

Where BPE decides *how* to split by applying merges in a fixed order, a unigram
model decides by probability: every piece carries a log-probability, and a word is
segmented along the highest-scoring path through its piece lattice.  The two
properties that make it worth having alongside BPE are:

* the same vocabulary can produce more than one segmentation, and the model picks
  the best one rather than the first one a merge order happens to give;
* training is pruning.  Start with too many candidate pieces and delete the ones
  the corpus does not need, which is the opposite direction from BPE's growth and
  catches a different class of mistake.

The estimator here is *hard* EM: the E-step uses the single best path rather than
the forward-sum over all paths.  That is a simplification, it converges to a
different (usually slightly worse) model than soft EM would, and it is much easier
to reason about -- the objective it increases is exactly the one
:func:`best_segmentation` optimises.
"""

from __future__ import annotations

from collections.abc import Mapping

from echoscribe.errors import TokenizerError, require
from echoscribe.numeric import NEGATIVE_INFINITY


def best_segmentation(word: str, log_probabilities: Mapping[str, float]) -> tuple[str, ...]:
    """The highest-probability segmentation of ``word`` into known pieces.

    A Viterbi pass over the lattice of piece matches.  Ties are broken towards the
    segmentation found first, which is the one that greedily takes the longest piece
    at the earliest position, so the result does not depend on dictionary order.
    """
    require(bool(word), "cannot segment an empty word", TokenizerError)
    length = len(word)
    best = [NEGATIVE_INFINITY] * (length + 1)
    previous = [-1] * (length + 1)
    best[0] = 0.0
    for position in range(length):
        if best[position] == NEGATIVE_INFINITY:
            continue
        for end in range(position + 1, length + 1):
            score = log_probabilities.get(word[position:end])
            if score is None:
                continue
            candidate = best[position] + score
            if candidate > best[end]:
                best[end] = candidate
                previous[end] = position
    require(
        best[length] > NEGATIVE_INFINITY,
        f"{word!r} cannot be segmented with this vocabulary",
        TokenizerError,
    )
    pieces = []
    cursor = length
    while cursor > 0:
        pieces.append(word[previous[cursor] : cursor])
        cursor = previous[cursor]
    return tuple(reversed(pieces))
