"""Prefix search is checked against the property only exhaustive search has."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoscribe.decode.ctc_prefix import PrefixBeamResult, ctc_prefix_search
from echoscribe.decode.posteriors import scripted_posteriors, to_log_posteriors
from echoscribe.errors import DecodeError


def test_a_prefix_created_in_the_final_frame_is_still_ranked() -> None:
    # One frame is enough to expose the bug: the label extension that creates (1,)
    # writes no blank entry, so a ranking pass over the blank map alone returns only
    # the empty hypothesis and loses almost all of the probability mass.
    matrix = to_log_posteriors(np.array([[0.25, 0.75]]))
    result = ctc_prefix_search(matrix, beam_size=None)
    scores = {item.tokens: math.exp(item.score) for item in result.hypotheses}
    assert scores[(1,)] == pytest.approx(0.75)
    assert scores[()] == pytest.approx(0.25)
    assert sum(scores.values()) == pytest.approx(1.0)


def test_a_two_frame_search_accounts_for_all_of_its_mass() -> None:
    matrix = to_log_posteriors(np.array([[0.5, 0.5], [0.2, 0.8]]))
    result = ctc_prefix_search(matrix, beam_size=None)
    total = sum(math.exp(item.score) for item in result.hypotheses)
    assert total == pytest.approx(1.0, rel=1e-12)


def test_an_unlimited_beam_is_a_complete_distribution() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.05, 1.0, size=(5, 3))
    matrix = to_log_posteriors(probabilities / probabilities.sum(axis=1, keepdims=True))
    result = ctc_prefix_search(matrix, beam_size=None)
    total = sum(math.exp(item.score) for item in result.hypotheses)
    assert total == pytest.approx(1.0, rel=1e-9)


def test_hypotheses_are_ordered_by_score() -> None:
    matrix = scripted_posteriors([0, 1, 1, 0, 2, 1], num_symbols=4)
    result = ctc_prefix_search(matrix, beam_size=None)
    scores = [item.score for item in result.hypotheses]
    assert scores == sorted(scores, reverse=True)


def test_no_prefix_appears_twice() -> None:
    matrix = scripted_posteriors([1, 0, 2, 2, 0, 1], num_symbols=4)
    result = ctc_prefix_search(matrix, beam_size=None)
    sequences = [item.tokens for item in result.hypotheses]
    assert len(sequences) == len(set(sequences))


def test_the_empty_prefix_is_always_present() -> None:
    matrix = scripted_posteriors([1, 2], num_symbols=3)
    result = ctc_prefix_search(matrix, beam_size=None)
    assert () in [item.tokens for item in result.hypotheses]


def test_a_repeated_label_is_reachable_only_through_a_blank() -> None:
    # With no blank frame at all, (1, 1) must be impossible: its probability is the
    # blank-branch term, and there is no blank mass to draw on.
    matrix = scripted_posteriors([1, 1, 1], num_symbols=3, peak_logit=40.0)
    result = ctc_prefix_search(matrix, beam_size=None)
    scores = {item.tokens: item.score for item in result.hypotheses}
    assert scores[(1,)] == pytest.approx(0.0, abs=1e-6)
    assert scores.get((1, 1), float("-inf")) < -20.0


def test_the_best_hypothesis_is_exposed_directly() -> None:
    matrix = scripted_posteriors([0, 1, 0, 2, 2], num_symbols=4)
    result = ctc_prefix_search(matrix, beam_size=None)
    assert isinstance(result, PrefixBeamResult)
    assert result.best is result.hypotheses[0]
    assert result.symbols == result.best.tokens


def test_an_empty_beam_has_no_best() -> None:
    with pytest.raises(DecodeError, match="the beam is empty"):
        _ = PrefixBeamResult(()).best


def test_n_best_truncates_in_order() -> None:
    matrix = scripted_posteriors([0, 1, 0, 2, 0, 3], num_symbols=4)
    result = ctc_prefix_search(matrix, beam_size=None)
    assert result.n_best(2) == result.hypotheses[:2]
    assert result.n_best(len(result.hypotheses)) == result.hypotheses


def test_n_best_rejects_a_count_below_one() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=3)
    with pytest.raises(DecodeError, match="at least one hypothesis"):
        ctc_prefix_search(matrix, beam_size=None).n_best(0)


def test_a_bad_blank_id_is_refused() -> None:
    with pytest.raises(DecodeError, match="outside the 3-symbol alphabet"):
        ctc_prefix_search(scripted_posteriors([0, 1], num_symbols=3), blank_id=9)


def test_a_bad_beam_size_is_refused() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=3)
    with pytest.raises(DecodeError, match="positive integer or None"):
        ctc_prefix_search(matrix, beam_size=0)


def test_an_empty_matrix_yields_only_the_empty_prefix() -> None:
    result = ctc_prefix_search(np.zeros((0, 3)))
    assert [item.tokens for item in result.hypotheses] == [()]
    assert result.best.score == pytest.approx(0.0)
