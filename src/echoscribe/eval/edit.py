"""Edit-distance alignment with a substitution/insertion/deletion breakdown.

Word error rate is not a distance; it is a *count over an alignment*.  Two hypotheses
with the same number of errors can differ completely in kind -- one substituted every
word, the other inserted a run of them -- and the difference matters for diagnosis.
So the primitive here is the alignment, and the counts are read off it.

The dynamic program is the textbook Wagner-Fischer one.  What is not textbook, and
what has to be pinned, is the tie-break: many alignments achieve the minimum cost,
and which one is reported decides whether a difference shows up as a substitution or
as a deletion-plus-insertion.  The order used here is

    diagonal (correct, then substitution) > deletion > insertion

which prefers to line tokens up rather than to shift them.  It is arbitrary in the
sense that another order would be equally minimal, and deliberate in the sense that
it is fixed, so the same pair of sequences always produces the same alignment and a
report can be diffed against a previous one.
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import EvalError, require, require_in

#: Kinds of edit operation an alignment can contain.
EDIT_OPERATIONS = ("correct", "substitution", "deletion", "insertion")

#: Kinds that count as an error.
ERROR_OPERATIONS = ("substitution", "deletion", "insertion")


@dataclass(frozen=True)
class EditOp:
    """One step of an alignment."""

    kind: str
    ref_index: int | None
    hyp_index: int | None
    ref_token: str
    hyp_token: str

    def __post_init__(self) -> None:
        require_in("kind", self.kind, EDIT_OPERATIONS, EvalError)
        if self.kind == "correct" or self.kind == "substitution":
            require(
                self.ref_index is not None and self.hyp_index is not None,
                f"a {self.kind} consumes one token from each side",
                EvalError,
            )
        elif self.kind == "deletion":
            require(self.ref_index is not None, "a deletion consumes a reference token", EvalError)
            require(self.hyp_index is None, "a deletion consumes no hypothesis token", EvalError)
        else:
            require(
                self.hyp_index is not None, "an insertion consumes a hypothesis token", EvalError
            )
            require(self.ref_index is None, "an insertion consumes no reference token", EvalError)

    @property
    def is_error(self) -> bool:
        """True for anything that is not a match."""
        return self.kind in ERROR_OPERATIONS
