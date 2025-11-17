"""The scorer's arithmetic is checked against the alignment it came from."""

from __future__ import annotations

import pytest

from echoscribe.errors import EvalError
from echoscribe.eval.wer import score_tokens


def test_the_breakdown_sums_to_the_error_count() -> None:
    report = score_tokens("the cat sat".split(), "a cat sat down".split())
    assert report.errors == report.substitutions + report.deletions + report.insertions
    assert report.errors == report.alignment.errors


def test_the_deletion_insertion_gap_explains_the_length_difference() -> None:
    for ref, hyp in [
        ("a b c", "a b c"),
        ("a b c", "a b"),
        ("a b", "a b c"),
        ("a b c d", "b d e"),
    ]:
        report = score_tokens(ref.split(), hyp.split())
        assert report.deletions - report.insertions == report.reference_length - len(hyp.split())


def test_the_alignment_inside_the_report_still_rebuilds_the_hypothesis() -> None:
    reference = "the cat sat on the mat".split()
    hypothesis = "a cat sat on the mat now".split()
    report = score_tokens(reference, hypothesis)
    assert report.alignment.apply(reference) == tuple(hypothesis)


def test_an_identical_pair_scores_zero() -> None:
    report = score_tokens(["a", "b"], ["a", "b"])
    assert report.errors == 0
    assert report.error_rate == 0.0


def test_scoring_is_a_pure_function_of_its_inputs() -> None:
    first = score_tokens("a b c".split(), "a x c".split())
    second = score_tokens("a b c".split(), "a x c".split())
    assert first == second


def test_the_report_keeps_its_tokens_as_tuples() -> None:
    report = score_tokens(["a", "b"], ["b"])
    assert report.reference_tokens == ("a", "b")
    assert report.hypothesis_tokens == ("b",)


def test_a_report_cannot_be_modified_after_the_fact() -> None:
    from dataclasses import FrozenInstanceError

    report = score_tokens(["a"], ["b"])
    with pytest.raises(FrozenInstanceError):
        report.substitutions = 0


@pytest.mark.parametrize("seed", range(6))
def test_random_pairs_keep_every_invariant(seed: int) -> None:
    import random

    vocabulary = ["a", "b", "c", "d"]
    rng = random.Random(seed)
    for _ in range(25):
        reference = [rng.choice(vocabulary) for _ in range(rng.randint(1, 8))]
        hypothesis = [rng.choice(vocabulary) for _ in range(rng.randint(0, 8))]
        report = score_tokens(reference, hypothesis)
        assert report.errors == report.alignment.errors
        assert report.deletions - report.insertions == len(reference) - len(hypothesis)
        assert report.alignment.apply(reference) == tuple(hypothesis)
        assert report.error_rate >= 0.0


def test_an_empty_reference_with_a_hypothesis_is_refused() -> None:
    with pytest.raises(EvalError, match="the reference is empty"):
        score_tokens([], ["anything"])


def test_two_empty_sequences_score_zero() -> None:
    report = score_tokens([], [])
    assert report.errors == 0
    assert report.error_rate == 0.0
    assert report.alignment.operations == ()


def test_an_empty_hypothesis_against_a_reference_is_all_deletions() -> None:
    report = score_tokens(["a", "b", "c"], [])
    assert report.deletions == 3
    assert report.substitutions == 0
    assert report.insertions == 0
    assert report.error_rate == 1.0
