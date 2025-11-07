"""Alignment is checked against brute force, because a DP bug is silent."""

from __future__ import annotations

import pytest

from echoscribe.errors import EvalError
from echoscribe.eval.edit import EDIT_OPERATIONS, ERROR_OPERATIONS, EditOp


def test_the_operation_kinds_are_the_four_an_alignment_can_hold() -> None:
    assert EDIT_OPERATIONS == ("correct", "substitution", "deletion", "insertion")
    assert ERROR_OPERATIONS == ("substitution", "deletion", "insertion")


def test_a_correct_operation_consumes_both_sides() -> None:
    op = EditOp("correct", 0, 0, "a", "a")
    assert not op.is_error


def test_a_substitution_is_an_error_on_both_sides() -> None:
    op = EditOp("substitution", 1, 1, "a", "b")
    assert op.is_error
    assert op.ref_token == "a"
    assert op.hyp_token == "b"


def test_a_deletion_consumes_only_the_reference() -> None:
    op = EditOp("deletion", 2, None, "a", "")
    assert op.is_error
    with pytest.raises(EvalError, match="consumes no hypothesis token"):
        EditOp("deletion", 2, 2, "a", "")


def test_an_insertion_consumes_only_the_hypothesis() -> None:
    op = EditOp("insertion", None, 3, "", "a")
    assert op.is_error
    with pytest.raises(EvalError, match="consumes no reference token"):
        EditOp("insertion", 1, 3, "", "a")


def test_a_correct_operation_needs_both_indices() -> None:
    with pytest.raises(EvalError, match="one token from each side"):
        EditOp("correct", None, 0, "a", "a")


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(EvalError, match="kind must be one of"):
        EditOp("transposition", 0, 0, "a", "b")


def test_an_operation_is_immutable() -> None:
    op = EditOp("correct", 0, 0, "a", "a")
    with pytest.raises(AttributeError):
        op.kind = "insertion"  # type: ignore[misc]
