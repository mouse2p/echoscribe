"""Alignment is checked against brute force, because a DP bug is silent."""

from __future__ import annotations

import pytest

from echoscribe.errors import EvalError
from echoscribe.eval.edit import (
    EDIT_OPERATIONS,
    ERROR_OPERATIONS,
    Alignment,
    EditOp,
    align,
    edit_distance,
)


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


def test_identical_sequences_align_without_errors() -> None:
    result = align(["a", "b", "c"], ["a", "b", "c"])
    assert result.errors == 0
    assert result.correct == 3
    assert all(op.kind == "correct" for op in result.operations)


def test_one_substitution_is_one_error() -> None:
    result = align(["a", "b", "c"], ["a", "x", "c"])
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.errors == 1


def test_a_missing_token_is_a_deletion() -> None:
    result = align(["a", "b", "c"], ["a", "c"])
    assert result.deletions == 1
    assert result.errors == 1


def test_an_extra_token_is_an_insertion() -> None:
    result = align(["a", "c"], ["a", "b", "c"])
    assert result.insertions == 1
    assert result.errors == 1


def test_an_empty_hypothesis_deletes_everything() -> None:
    result = align(["a", "b", "c"], [])
    assert result.deletions == 3
    assert result.correct == 0
    assert result.errors == 3


def test_an_empty_reference_inserts_everything() -> None:
    result = align([], ["a", "b"])
    assert result.insertions == 2
    assert result.errors == 2


def test_two_empty_sequences_align_to_nothing() -> None:
    result = align([], [])
    assert isinstance(result, Alignment)
    assert result.operations == ()
    assert result.errors == 0


def test_edit_distance_matches_the_alignment() -> None:
    assert edit_distance("kitten", "sitting") == align("kitten", "sitting").errors == 3


def test_a_complete_replacement_costs_the_longer_side() -> None:
    assert edit_distance(list("abcd"), list("wxyz")) == 4


def test_the_distance_to_itself_is_zero() -> None:
    assert edit_distance(list("hello world"), list("hello world")) == 0


def test_the_distance_is_symmetric() -> None:
    for left, right in [("abc", "abd"), ("", "xyz"), ("aabb", "ab"), ("hello", "hallo")]:
        assert edit_distance(list(left), list(right)) == edit_distance(list(right), list(left))


def test_reference_and_hypothesis_lengths_are_recovered() -> None:
    result = align(list("abcdef"), list("abxdefg"))
    assert result.reference_length == 6
    assert result.hypothesis_length == 7


def test_operation_indices_walk_both_sequences_in_order() -> None:
    ref = list("abcdef")
    hyp = list("abxdf")
    result = align(ref, hyp)
    seen_ref = [op.ref_index for op in result.operations if op.ref_index is not None]
    seen_hyp = [op.hyp_index for op in result.operations if op.hyp_index is not None]
    assert seen_ref == list(range(len(ref)))
    assert seen_hyp == list(range(len(hyp)))


def test_alignment_is_deterministic() -> None:
    ref, hyp = list("the quick brown fox"), list("a quick brown box")
    assert align(ref, hyp) == align(ref, hyp)


def test_replaying_the_operations_rebuilds_the_hypothesis() -> None:
    for ref, hyp in [
        ("abc", "abc"),
        ("abc", "abd"),
        ("abc", "ac"),
        ("abc", "abxc"),
        ("", "xyz"),
        ("xyz", ""),
        ("the cat sat", "a cat sat down"),
    ]:
        reference, hypothesis = list(ref.split()), list(hyp.split())
        assert align(reference, hypothesis).apply(reference) == tuple(hypothesis)


def test_replaying_a_character_alignment_rebuilds_the_string() -> None:
    reference = list("kitten")
    hypothesis = list("sitting")
    result = align(reference, hypothesis)
    assert "".join(result.apply(reference)) == "sitting"


def test_replaying_over_the_wrong_reference_is_refused() -> None:
    result = align(list("abc"), list("abd"))
    with pytest.raises(EvalError, match="expects 'a' but found 'z'"):
        result.apply(list("zbc"))


def test_the_pair_view_has_one_entry_per_operation() -> None:
    result = align(list("abc"), list("axc"))
    assert result.pairs() == (("a", "a"), ("b", "x"), ("c", "c"))


def test_a_gap_appears_as_none_in_the_pair_view() -> None:
    assert align(list("ab"), list("a")).pairs() == (("a", "a"), ("b", None))
    assert align(list("a"), list("ab")).pairs() == (("a", "a"), (None, "b"))


@pytest.mark.parametrize("seed", range(6))
def test_random_pairs_always_rebuild_the_hypothesis(seed: int) -> None:
    import random

    alphabet = "abcde"
    rng = random.Random(seed)
    for _ in range(30):
        reference = [rng.choice(alphabet) for _ in range(rng.randint(0, 9))]
        hypothesis = [rng.choice(alphabet) for _ in range(rng.randint(0, 9))]
        result = align(reference, hypothesis)
        assert result.apply(reference) == tuple(hypothesis)
        assert result.errors == edit_distance(reference, hypothesis)


@pytest.mark.parametrize("seed", range(8))
def test_the_dynamic_program_matches_brute_force(seed: int) -> None:
    import random

    from echoscribe.eval.edit import brute_force_distance

    alphabet = "abc"
    rng = random.Random(100 + seed)
    for _ in range(40):
        reference = [rng.choice(alphabet) for _ in range(rng.randint(0, 6))]
        hypothesis = [rng.choice(alphabet) for _ in range(rng.randint(0, 6))]
        assert edit_distance(reference, hypothesis) == brute_force_distance(reference, hypothesis)


def test_brute_force_agrees_on_the_classic_examples() -> None:
    from echoscribe.eval.edit import brute_force_distance

    assert brute_force_distance(list("kitten"), list("sitting")) == 3
    assert brute_force_distance([], list("abc")) == 3
    assert brute_force_distance(list("abc"), []) == 3
    assert brute_force_distance(list("flaw"), list("lawn")) == 2


def test_the_error_count_is_the_operation_count_minus_the_matches() -> None:
    result = align(list("abcdef"), list("abxdefg"))
    assert len(result.operations) - result.correct == result.errors


def test_every_operation_consumes_exactly_one_side_or_both() -> None:
    result = align(list("the cat sat"), list("a cat sat down"))
    for op in result.operations:
        consumed = (op.ref_index is not None) + (op.hyp_index is not None)
        assert consumed == (1 if op.kind in ("deletion", "insertion") else 2)


def test_a_long_sequence_with_one_change_reports_one_error() -> None:
    reference = [str(index) for index in range(500)]
    hypothesis = list(reference)
    hypothesis[237] = "changed"
    result = align(reference, hypothesis)
    assert result.errors == 1
    assert result.substitutions == 1
    assert result.correct == 499
    assert result.apply(reference) == tuple(hypothesis)


def test_a_long_insertion_reports_only_insertions() -> None:
    reference = [str(index) for index in range(200)]
    hypothesis = reference[:100] + ["extra"] * 50 + reference[100:]
    result = align(reference, hypothesis)
    assert result.insertions == 50
    assert result.deletions == 0
    assert result.substitutions == 0


def test_a_reversed_sequence_costs_no_more_than_its_length() -> None:
    reference = [str(index) for index in range(60)]
    assert edit_distance(reference, list(reversed(reference))) <= len(reference)


def test_the_alignment_of_a_long_pair_is_reproducible() -> None:
    reference = [str(index % 17) for index in range(300)]
    hypothesis = [str((index + 3) % 17) for index in range(300)]
    assert align(reference, hypothesis) == align(reference, hypothesis)


def test_alignment_accepts_tuples_as_well_as_lists() -> None:
    assert align(("a", "b"), ("a", "c")).errors == 1
