"""Greedy CTC is checked against hand-written frame paths."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.ctc_greedy import GreedyResult, ctc_greedy
from echoscribe.decode.posteriors import scripted_posteriors
from echoscribe.errors import DecodeError


def test_blanks_are_dropped() -> None:
    matrix = scripted_posteriors([0, 0, 0], num_symbols=4)
    assert ctc_greedy(matrix).symbols == ()


def test_a_run_of_one_symbol_emits_once() -> None:
    matrix = scripted_posteriors([1, 1, 1, 1], num_symbols=4)
    assert ctc_greedy(matrix).symbols == (1,)


def test_a_blank_separates_two_copies_of_the_same_symbol() -> None:
    matrix = scripted_posteriors([1, 0, 1], num_symbols=4)
    assert ctc_greedy(matrix).symbols == (1, 1)


def test_collapse_happens_before_blank_removal() -> None:
    # "a a <blank> a" has to yield three a's, not one.
    matrix = scripted_posteriors([2, 2, 0, 2], num_symbols=4)
    assert ctc_greedy(matrix).symbols == (2, 2)


def test_a_mixed_path_decodes_in_order() -> None:
    matrix = scripted_posteriors([0, 1, 1, 0, 2, 0, 3, 3, 3, 0], num_symbols=5)
    assert ctc_greedy(matrix).symbols == (1, 2, 3)


def test_the_argmax_path_is_returned_verbatim() -> None:
    path = [0, 1, 1, 0, 2]
    result = ctc_greedy(scripted_posteriors(path, num_symbols=4))
    assert result.path == tuple(path)
    assert result.num_frames == 5


def test_the_score_is_the_sum_of_the_chosen_log_posteriors() -> None:
    matrix = scripted_posteriors([1, 0, 2], num_symbols=4)
    result = ctc_greedy(matrix)
    expected = float(sum(matrix[frame, symbol] for frame, symbol in enumerate(result.path)))
    assert result.score == pytest.approx(expected)


def test_a_score_is_never_positive() -> None:
    result = ctc_greedy(scripted_posteriors([1, 2, 3], num_symbols=4))
    assert result.score <= 0.0


def test_an_empty_matrix_decodes_to_nothing() -> None:
    result = ctc_greedy(np.zeros((0, 4)))
    assert isinstance(result, GreedyResult)
    assert result.symbols == ()
    assert result.score == 0.0


def test_a_blank_outside_the_alphabet_is_refused() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=3)
    with pytest.raises(DecodeError, match="outside the 3-symbol alphabet"):
        ctc_greedy(matrix, blank_id=5)


def test_an_unnormalised_matrix_is_refused() -> None:
    with pytest.raises(DecodeError, match="not normalised"):
        ctc_greedy(np.log(np.full((3, 4), 0.5)))


def test_a_custom_blank_id_is_honoured() -> None:
    # Blank in the last slot instead of the first: the same rules, different index.
    matrix = scripted_posteriors([3, 1, 1, 3, 2], num_symbols=4)
    assert ctc_greedy(matrix, blank_id=3).symbols == (1, 2)


def test_greedy_decoding_is_deterministic() -> None:
    matrix = scripted_posteriors([0, 1, 2, 0, 1], num_symbols=4)
    assert ctc_greedy(matrix).symbols == ctc_greedy(matrix).symbols


def test_token_spans_are_disjoint_and_ordered() -> None:
    matrix = scripted_posteriors([0, 1, 1, 0, 2, 0, 0, 3], num_symbols=4)
    tokens = ctc_greedy(matrix).tokens
    for earlier, later in zip(tokens, tokens[1:], strict=False):
        assert earlier.end <= later.start


def test_every_token_span_covers_its_own_symbol() -> None:
    path = [0, 1, 1, 0, 2, 2, 2, 0]
    matrix = scripted_posteriors(path, num_symbols=4)
    for token in ctc_greedy(matrix).tokens:
        assert set(path[token.start : token.end]) == {token.symbol}


def test_a_repeated_symbol_gets_two_spans_separated_by_the_blank() -> None:
    matrix = scripted_posteriors([1, 1, 0, 1, 1], num_symbols=3)
    tokens = ctc_greedy(matrix).tokens
    assert [token.symbol for token in tokens] == [1, 1]
    assert tokens[0].end == 2
    assert tokens[1].start == 3


def test_a_token_score_is_the_sum_over_its_span() -> None:
    matrix = scripted_posteriors([1, 1, 0, 2], num_symbols=4)
    for token in ctc_greedy(matrix).tokens:
        assert token.score == pytest.approx(
            float(matrix[token.start : token.end, token.symbol].sum())
        )


def test_token_times_follow_the_frame_shift() -> None:
    matrix = scripted_posteriors([0, 1, 1, 0, 2], num_symbols=4)
    tokens = ctc_greedy(matrix).tokens
    assert tokens[0].times(10.0) == (0.01, pytest.approx(0.03))
    assert tokens[1].times(10.0) == (0.04, pytest.approx(0.05))


def test_greedy_spans_never_revisit_a_frame() -> None:
    rng = np.random.default_rng(1)
    path = [int(value) for value in rng.integers(0, 5, size=64)]
    tokens = ctc_greedy(scripted_posteriors(path, num_symbols=5)).tokens
    covered = []
    for token in tokens:
        covered.extend(range(token.start, token.end))
    assert len(covered) == len(set(covered))
    assert max(covered, default=-1) < 64
