"""Word and character error rates over an edit-distance alignment.

The error rate is the alignment's error count divided by the reference length:

    rate = (substitutions + deletions + insertions) / N_reference

Three conventions matter here, and each is a choice rather than a fact:

* The denominator is the *reference* length, so a hypothesis longer than the
  reference can produce a rate above 1.0.  Capping it would hide insertions.
* An empty reference gives the rate no defined value; scoring one is refused
  rather than answered with a number that looks meaningful.
* The rate is rounded where it is produced, to six decimals, so a golden test
  that pins it pins the same value on every platform.

This module scores *token sequences*.  What counts as a token -- a word, a
character, a subword unit -- is the caller's choice, and the two entry points
here (:func:`wer` and :func:`cer`) are conveniences over the same arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from echoscribe.errors import EvalError
from echoscribe.eval.edit import Alignment, align

ERROR_RATE_DECIMALS = 6


@dataclass(frozen=True)
class ScoreReport:
    """The outcome of scoring one hypothesis against one reference."""

    reference_tokens: tuple[str, ...]
    hypothesis_tokens: tuple[str, ...]
    substitutions: int
    deletions: int
    insertions: int
    alignment: Alignment

    @property
    def errors(self) -> int:
        """Total edit distance: substitutions plus deletions plus insertions."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        """The denominator of the error rate."""
        return len(self.reference_tokens)

    @property
    def error_rate(self) -> float:
        """Errors over reference length, rounded where it is produced."""
        if self.reference_length == 0:
            return 0.0
        return round(self.errors / self.reference_length, ERROR_RATE_DECIMALS)


def score_tokens(reference: Sequence[str], hypothesis: Sequence[str]) -> ScoreReport:
    """Score one pair of token sequences at minimum edit distance.

    The alignment inside the returned report is the same one the counts come
    from, so a report can always show its work: ``report.alignment`` explains
    every error the rate is made of.
    """
    ref = tuple(reference)
    hyp = tuple(hypothesis)
    if not ref and hyp:
        raise EvalError(
            "the reference is empty but the hypothesis is not, so the error "
            "rate would divide by zero; this pair has no defined score"
        )
    result = align(ref, hyp)
    return ScoreReport(
        reference_tokens=ref,
        hypothesis_tokens=hyp,
        substitutions=result.substitutions,
        deletions=result.deletions,
        insertions=result.insertions,
        alignment=result,
    )


def word_sequence(text: str) -> tuple[str, ...]:
    """Split text into words: whitespace runs are the only separators."""
    return tuple(text.split())


def character_sequence(text: str) -> tuple[str, ...]:
    """Split text into characters, dropping whitespace entirely.

    This is the convention character error rate uses for Chinese: spaces carry
    no linguistic information there, and scoring them would measure the typist
    rather than the recognizer.
    """
    return tuple(character for character in text if not character.isspace())


def wer(reference: str, hypothesis: str) -> ScoreReport:
    """Word error rate: score two strings as whitespace-separated words."""
    return score_tokens(word_sequence(reference), word_sequence(hypothesis))


def cer(reference: str, hypothesis: str) -> ScoreReport:
    """Character error rate: score two strings as whitespace-free characters."""
    return score_tokens(character_sequence(reference), character_sequence(hypothesis))
