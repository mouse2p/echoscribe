"""CTC prefix-beam search.

Greedy decoding scores one alignment.  Prefix search scores a *hypothesis*: the
total probability of every alignment that collapses to it.  That difference is the
whole reason the algorithm exists -- the most likely single path is often not the
path whose label sequence is most likely, because a label sequence can be reached
by many alignments and their probabilities add.

The bookkeeping that makes the addition correct is the pair of scores every prefix
carries:

``blank``
    probability of the alignments for this prefix whose *last* frame was a blank;
``label``
    probability of those whose last frame was a real symbol.

They are kept apart because of the repeat rule.  Extending a prefix that already
ends in ``c`` with another ``c`` produces the *same* prefix if no blank intervened,
and a longer one if a blank did.  So the same-prefix case may only draw on the
``label`` score, and the extension case only on the ``blank`` score.  Merging the
two is the classic bug: it double-counts alignments and quietly inflates repeated
tokens.

Scores are accumulated in the log domain with :func:`echoscribe.numeric.log_add`.

Complexity is ``O(frames * beam * symbols)`` per utterance, dominated by the Python
loop over the beam.  That is fine for the sizes this toolkit targets and hopeless
for production decoding, where the inner loop is fused into a GPU kernel and the
beam is a fixed-size tensor.  Nothing here is a claim about throughput."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.decode.posteriors import check_log_posteriors
from echoscribe.decode.tokens import LOG_PROBABILITY_TOLERANCE
from echoscribe.errors import DecodeError, require
from echoscribe.numeric import NEGATIVE_INFINITY, log_add


@dataclass(frozen=True)
class PrefixHypothesis:
    """One collapsed label sequence and its total log-probability."""

    tokens: tuple[int, ...]
    score: float

    def __post_init__(self) -> None:
        require(
            self.score <= LOG_PROBABILITY_TOLERANCE,
            f"a log-probability cannot be positive, got {self.score}",
            DecodeError,
        )


@dataclass(frozen=True)
class PrefixBeamResult:
    """The beam's surviving hypotheses, best first."""

    hypotheses: tuple[PrefixHypothesis, ...]

    @property
    def best(self) -> PrefixHypothesis:
        """The highest-scoring hypothesis."""
        require(len(self.hypotheses) > 0, "the beam is empty", DecodeError)
        return self.hypotheses[0]

    @property
    def symbols(self) -> tuple[int, ...]:
        """The best hypothesis as a plain symbol tuple."""
        return self.best.tokens

    def n_best(self, count: int) -> tuple[PrefixHypothesis, ...]:
        """The first ``count`` hypotheses, in score order."""
        require(count >= 1, f"n_best needs at least one hypothesis, got {count}", DecodeError)
        return self.hypotheses[:count]


def ctc_prefix_search(
    posteriors: np.ndarray,
    beam_size: int | None = None,
    blank_id: int = 0,
) -> PrefixBeamResult:
    """Search for the most probable collapsed label sequences.

    ``beam_size=None`` keeps every reachable prefix, which is exponential in the
    frame count and only usable on short inputs -- but it is the setting under which
    the hypotheses are a complete distribution and therefore have to sum to one.
    """
    matrix = check_log_posteriors(posteriors, "CTC posteriors")
    frames, symbols = matrix.shape
    require(
        0 <= blank_id < symbols,
        f"blank id {blank_id} is outside the {symbols}-symbol alphabet",
        DecodeError,
    )
    if beam_size is not None:
        require(
            isinstance(beam_size, int) and not isinstance(beam_size, bool) and beam_size >= 1,
            f"beam_size must be a positive integer or None, got {beam_size!r}",
            DecodeError,
        )
    blank_scores: dict[tuple[int, ...], float] = {(): 0.0}
    label_scores: dict[tuple[int, ...], float] = {(): NEGATIVE_INFINITY}
    for frame in range(frames):
        blank_scores, label_scores = _advance(matrix[frame], blank_scores, label_scores, blank_id)
        if beam_size is not None:
            blank_scores, label_scores = _prune(blank_scores, label_scores, beam_size)
    return PrefixBeamResult(_rank(blank_scores, label_scores))


def _advance(
    row: np.ndarray,
    blank_scores: dict,
    label_scores: dict,
    blank_id: int,
) -> tuple[dict, dict]:
    """Extend every live prefix by one frame."""
    next_blank: dict = {}
    next_label: dict = {}
    symbols = row.shape[0]
    # The live set is the union of the two maps, not the blank map alone: a prefix
    # created by a label extension has no blank entry yet, and iterating only the
    # blank keys would drop every hypothesis that emitted its last symbol in the
    # frame just processed.
    for prefix in set(blank_scores) | set(label_scores):
        blank = blank_scores.get(prefix, NEGATIVE_INFINITY)
        label = label_scores.get(prefix, NEGATIVE_INFINITY)
        total = log_add(blank, label)
        if total == NEGATIVE_INFINITY:
            continue
        _accumulate(next_blank, prefix, total + float(row[blank_id]))
        for symbol in range(symbols):
            if symbol == blank_id:
                continue
            score = float(row[symbol])
            if score == NEGATIVE_INFINITY:
                continue
            if prefix and prefix[-1] == symbol:
                _accumulate(next_label, prefix, label + score)
                _accumulate(next_label, prefix + (symbol,), blank + score)
            else:
                _accumulate(next_label, prefix + (symbol,), total + score)
    return next_blank, next_label


def _accumulate(table: dict, prefix: tuple, score: float) -> None:
    """Add a log-probability into a beam entry, creating it if needed."""
    table[prefix] = log_add(table.get(prefix, NEGATIVE_INFINITY), score)


def _live_prefixes(blank_scores: dict, label_scores: dict) -> set:
    """Every prefix either map still carries score for."""
    return set(blank_scores) | set(label_scores)


def _total(blank_scores: dict, label_scores: dict, prefix: tuple) -> float:
    """Combined log-probability of one prefix across both branches."""
    return log_add(
        blank_scores.get(prefix, NEGATIVE_INFINITY),
        label_scores.get(prefix, NEGATIVE_INFINITY),
    )


def _prune(blank_scores: dict, label_scores: dict, beam_size: int) -> tuple[dict, dict]:
    """Keep the ``beam_size`` prefixes with the highest total score."""
    live = _live_prefixes(blank_scores, label_scores)
    if len(live) <= beam_size:
        return blank_scores, label_scores
    ranked = sorted(live, key=lambda prefix: (-_total(blank_scores, label_scores, prefix), prefix))
    keep = set(ranked[:beam_size])
    return (
        {prefix: score for prefix, score in blank_scores.items() if prefix in keep},
        {prefix: score for prefix, score in label_scores.items() if prefix in keep},
    )


def _rank(blank_scores: dict, label_scores: dict) -> tuple[PrefixHypothesis, ...]:
    """Merge the two score maps into a ranked hypothesis tuple."""
    hypotheses = [
        PrefixHypothesis(prefix, _total(blank_scores, label_scores, prefix))
        for prefix in _live_prefixes(blank_scores, label_scores)
    ]
    hypotheses.sort(key=lambda item: (-item.score, item.tokens))
    return tuple(hypotheses)
