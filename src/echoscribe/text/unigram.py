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

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from echoscribe.errors import ConfigError, TokenizerError, require, require_positive
from echoscribe.numeric import NEGATIVE_INFINITY

#: Floor that keeps an unused piece's probability finite so it can still be pruned.
PROBABILITY_FLOOR = 1e-12


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


def corpus_log_likelihood(words: Iterable[str], log_probabilities: Mapping[str, float]) -> float:
    """Total log-probability of the corpus under its best segmentations.

    This is the objective hard EM increases, and the number a pruning step reports
    when it claims a deletion cost almost nothing.
    """
    frequencies = Counter(words)
    total = 0.0
    for word, frequency in frequencies.items():
        path = best_segmentation(word, log_probabilities)
        total += frequency * sum(log_probabilities[piece] for piece in path)
    return total


def candidate_pieces(
    words: Iterable[str], max_length: int = 6, min_count: int = 2
) -> tuple[str, ...]:
    """Every character, plus every substring seen at least ``min_count`` times.

    Starting from characters guarantees the lattice is always connected, so no word
    can be unsegmentable during training.  Longer candidates are what let the model
    discover that ``"stat"`` earns its place; ``min_count`` keeps the candidate set
    from exploding on a small corpus.
    """
    require_positive("max_length", max_length, ConfigError)
    require_positive("min_count", min_count, ConfigError)
    frequencies = Counter(words)
    counts: Counter = Counter()
    for word, frequency in frequencies.items():
        seen = set()
        for start in range(len(word)):
            for end in range(start + 1, min(start + max_length, len(word)) + 1):
                seen.add(word[start:end])
        for piece in seen:
            counts[piece] += frequency
    return tuple(sorted(piece for piece, count in counts.items() if count >= min_count))


def estimate_log_probabilities(
    words: Iterable[str],
    pieces: Sequence[str],
    iterations: int = 5,
) -> dict[str, float]:
    """Hard-EM estimate of one log-probability per piece.

    The E-step segments every word along its best path and counts the pieces on it;
    the M-step renormalises those counts into a distribution.  A piece that no word
    uses is floored rather than set to ``-inf`` so it stays a candidate for pruning
    instead of poisoning every later sum.
    """
    require(len(pieces) > 0, "cannot estimate over an empty piece set", ConfigError)
    require_positive("iterations", iterations, ConfigError)
    frequencies = Counter(words)
    uniform = -math.log(len(pieces))
    log_probabilities = dict.fromkeys(pieces, uniform)
    for _ in range(int(iterations)):
        counts = dict.fromkeys(pieces, 0.0)
        for word, frequency in frequencies.items():
            for piece in best_segmentation(word, log_probabilities):
                counts[piece] = counts.get(piece, 0.0) + frequency
        total = sum(counts.values())
        if total <= 0.0:
            break
        log_probabilities = {
            piece: math.log(max(count, PROBABILITY_FLOOR) / total)
            for piece, count in counts.items()
        }
    return log_probabilities
