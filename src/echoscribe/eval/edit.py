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

from collections.abc import Sequence
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


@dataclass(frozen=True)
class Alignment:
    """An aligned pair of sequences, as the operations that connect them."""

    operations: tuple[EditOp, ...]

    @property
    def correct(self) -> int:
        """Tokens that matched."""
        return sum(1 for op in self.operations if op.kind == "correct")

    @property
    def substitutions(self) -> int:
        """Tokens present on both sides but different."""
        return sum(1 for op in self.operations if op.kind == "substitution")

    @property
    def deletions(self) -> int:
        """Reference tokens the hypothesis missed."""
        return sum(1 for op in self.operations if op.kind == "deletion")

    @property
    def insertions(self) -> int:
        """Hypothesis tokens the reference did not have."""
        return sum(1 for op in self.operations if op.kind == "insertion")

    @property
    def errors(self) -> int:
        """Total edit distance: substitutions plus deletions plus insertions."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        """Tokens consumed from the reference, which is correct + sub + deletion."""
        return self.correct + self.substitutions + self.deletions

    @property
    def hypothesis_length(self) -> int:
        """Tokens consumed from the hypothesis, which is correct + sub + insertion."""
        return self.correct + self.substitutions + self.insertions

    def apply(self, reference: Sequence[str]) -> tuple[str, ...]:
        """Rebuild the hypothesis by replaying the operations over ``reference``.

        This is the alignment's own consistency check, and the one that catches a
        wrong backtrace: an alignment that reports the right *number* of errors can
        still describe a path that does not lead from the reference to the
        hypothesis.
        """
        rebuilt: list[str] = []
        for op in self.operations:
            if op.kind == "insertion":
                rebuilt.append(op.hyp_token)
                continue
            if op.ref_index is None or op.ref_index >= len(reference):
                raise EvalError(
                    f"operation {op.kind!r} refers to reference index {op.ref_index} "
                    f"of a {len(reference)}-token sequence"
                )
            if reference[op.ref_index] != op.ref_token:
                raise EvalError(
                    f"operation at reference index {op.ref_index} expects "
                    f"{op.ref_token!r} but found {reference[op.ref_index]!r}"
                )
            if op.kind != "deletion":
                rebuilt.append(op.hyp_token)
        return tuple(rebuilt)

    def pairs(self) -> tuple[tuple[str | None, str | None], ...]:
        """The alignment as ``(reference, hypothesis)`` pairs, ``None`` for a gap.

        This is the shape a report renders and the shape a scorer iterates, and
        having one place produce it is what keeps the two from disagreeing.
        """
        return tuple(
            (
                op.ref_token if op.ref_index is not None else None,
                op.hyp_token if op.hyp_index is not None else None,
            )
            for op in self.operations
        )


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Align two token sequences at minimum edit distance.

    The cost matrix is Wagner-Fischer with unit costs; the backtrace walks it from
    the bottom-right corner and takes the first available move in the documented
    preference order, which is what makes the result a function of its inputs rather
    than of a dictionary's iteration order.
    """
    ref = list(reference)
    hyp = list(hypothesis)
    rows, columns = len(ref), len(hyp)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for index in range(1, rows + 1):
        cost[index][0] = index
    for index in range(1, columns + 1):
        cost[0][index] = index
    for i in range(1, rows + 1):
        current, previous = cost[i], cost[i - 1]
        ref_token = ref[i - 1]
        for j in range(1, columns + 1):
            diagonal = previous[j - 1] + (0 if ref_token == hyp[j - 1] else 1)
            up = previous[j] + 1
            left = current[j - 1] + 1
            current[j] = diagonal if diagonal < up else up
            if left < current[j]:
                current[j] = left
    return Alignment(_backtrace(cost, ref, hyp))


def _backtrace(cost: list, ref: list, hyp: list) -> tuple[EditOp, ...]:
    """Walk the cost matrix back to the origin, collecting operations."""
    operations: list[EditOp] = []
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if cost[i][j] == cost[i - 1][j - 1] + (0 if same else 1):
                kind = "correct" if same else "substitution"
                operations.append(EditOp(kind, i - 1, j - 1, ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            operations.append(EditOp("deletion", i - 1, None, ref[i - 1], ""))
            i -= 1
            continue
        if j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            operations.append(EditOp("insertion", None, j - 1, "", hyp[j - 1]))
            j -= 1
            continue
        raise EvalError(f"the cost matrix has no valid move at ({i}, {j})")
    return tuple(reversed(operations))


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """The minimum number of single-token edits between two sequences."""
    return align(reference, hypothesis).errors
