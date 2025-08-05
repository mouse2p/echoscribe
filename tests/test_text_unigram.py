"""The Viterbi path is checked against an exhaustive search over segmentations."""

from __future__ import annotations

import itertools
import math

import pytest

from echoscribe.errors import ConfigError, TokenizerError
from echoscribe.text.unigram import (
    best_segmentation,
    candidate_pieces,
    corpus_log_likelihood,
    estimate_log_probabilities,
)


def _all_segmentations(word: str) -> list[tuple[str, ...]]:
    """Every way to cut ``word`` into contiguous non-empty pieces."""
    cuts = range(1, len(word))
    result = []
    for size in range(len(cuts) + 1):
        for positions in itertools.combinations(cuts, size):
            bounds = (0, *positions, len(word))
            result.append(tuple(word[a:b] for a, b in zip(bounds, bounds[1:], strict=False)))
    return result


def test_the_viterbi_path_beats_every_alternative() -> None:
    word = "abcdef"
    probabilities = {
        piece: math.log(0.01)
        for piece in {word[i:j] for i in range(len(word)) for j in range(i + 1, len(word) + 1)}
    }
    probabilities["abc"] = math.log(0.5)
    probabilities["def"] = math.log(0.4)
    chosen = best_segmentation(word, probabilities)
    best_score = sum(probabilities[piece] for piece in chosen)
    for alternative in _all_segmentations(word):
        score = sum(probabilities.get(piece, float("-inf")) for piece in alternative)
        assert best_score >= score - 1e-12


def test_a_single_piece_wins_when_it_is_probable_enough() -> None:
    probabilities = {
        "low": math.log(0.6),
        "l": math.log(0.1),
        "o": math.log(0.1),
        "w": math.log(0.1),
    }
    assert best_segmentation("low", probabilities) == ("low",)


def test_characters_win_when_no_longer_piece_is_known() -> None:
    probabilities = {character: math.log(0.25) for character in "low"}
    assert best_segmentation("low", probabilities) == ("l", "o", "w")


def test_a_rare_long_piece_loses_to_frequent_short_ones() -> None:
    probabilities = {
        "low": math.log(1e-6),
        "l": math.log(0.3),
        "o": math.log(0.3),
        "w": math.log(0.3),
    }
    assert best_segmentation("low", probabilities) == ("l", "o", "w")


def test_ties_resolve_to_the_earliest_longest_cut() -> None:
    probabilities = {"ab": math.log(0.25), "cd": math.log(0.25)}
    probabilities.update({character: math.log(0.25) for character in "abcd"})
    assert best_segmentation("abcd", probabilities) == ("ab", "cd")


def test_an_unsegmentable_word_is_reported() -> None:
    with pytest.raises(TokenizerError, match="cannot be segmented"):
        best_segmentation("xyz", {"a": math.log(0.5)})


def test_an_empty_word_is_refused() -> None:
    with pytest.raises(TokenizerError, match="cannot segment an empty word"):
        best_segmentation("", {"a": math.log(0.5)})


def test_a_longer_vocabulary_still_finds_the_optimal_path() -> None:
    word = "statistics"
    probabilities = {character: math.log(0.05) for character in set(word)}
    for piece in ("stat", "istic", "stics", "tis"):
        probabilities[piece] = math.log(0.2)
    chosen = best_segmentation(word, probabilities)
    best_score = sum(probabilities[piece] for piece in chosen)
    for alternative in _all_segmentations(word):
        score = sum(probabilities.get(piece, float("-inf")) for piece in alternative)
        assert best_score >= score - 1e-12


def test_a_word_of_one_character_segments_to_itself() -> None:
    assert best_segmentation("a", {"a": math.log(1.0)}) == ("a",)


def test_repeated_pieces_are_handled() -> None:
    probabilities = {"aa": math.log(0.4), "a": math.log(0.3)}
    assert best_segmentation("aaaa", probabilities) == ("aa", "aa")


def test_the_path_pieces_rebuild_the_word() -> None:
    probabilities = {character: math.log(0.1) for character in "abcdef"}
    probabilities["bc"] = math.log(0.3)
    probabilities["ef"] = math.log(0.3)
    for word in ("abcdef", "abc", "fedcba"):
        assert "".join(best_segmentation(word, probabilities)) == word


def test_unknown_pieces_are_simply_not_used() -> None:
    probabilities = {"a": math.log(0.5), "zzz": math.log(0.9)}
    assert best_segmentation("aa", probabilities) == ("a", "a")


TRAIN = 6 * ["low"] + 4 * ["lowest"] + 8 * ["newer"] + 8 * ["wider"] + 3 * ["new"]


def test_candidate_pieces_always_include_every_character() -> None:
    pieces = candidate_pieces(TRAIN, min_count=1)
    for word in set(TRAIN):
        assert set(word) <= set(pieces)


def test_candidate_pieces_respect_the_minimum_count() -> None:
    pieces = candidate_pieces(["abc", "abd"], max_length=3, min_count=2)
    assert "ab" in pieces
    assert "abc" not in pieces


def test_candidate_pieces_respect_the_maximum_length() -> None:
    pieces = candidate_pieces(["abcdef"] * 4, max_length=3, min_count=1)
    assert all(len(piece) <= 3 for piece in pieces)


def test_candidate_pieces_are_sorted_and_unique() -> None:
    pieces = candidate_pieces(TRAIN, min_count=1)
    assert list(pieces) == sorted(set(pieces))


def test_one_em_iteration_does_not_lower_the_likelihood() -> None:
    pieces = candidate_pieces(TRAIN, max_length=5, min_count=2)
    uniform = {piece: -math.log(len(pieces)) for piece in pieces}
    before = corpus_log_likelihood(TRAIN, uniform)
    after = corpus_log_likelihood(TRAIN, estimate_log_probabilities(TRAIN, pieces, 1))
    assert after >= before - 1e-9


def test_em_probabilities_form_a_distribution() -> None:
    pieces = candidate_pieces(TRAIN, max_length=5, min_count=2)
    log_probabilities = estimate_log_probabilities(TRAIN, pieces, 5)
    assert sum(math.exp(value) for value in log_probabilities.values()) == pytest.approx(
        1.0, rel=1e-6
    )


def test_em_concentrates_mass_on_the_pieces_it_actually_uses() -> None:
    pieces = candidate_pieces(TRAIN, max_length=5, min_count=2)
    log_probabilities = estimate_log_probabilities(TRAIN, pieces, 10)
    used = set()
    for word in set(TRAIN):
        used.update(best_segmentation(word, log_probabilities))
    unused = set(pieces) - used
    # Hard EM counts only the best path, so a piece that never appears on one ends
    # up at the floor while the ones that do share all of the mass.
    assert used and unused
    assert min(log_probabilities[piece] for piece in used) > max(
        log_probabilities[piece] for piece in unused
    )


def test_unused_pieces_are_floored_rather_than_minus_infinite() -> None:
    pieces = candidate_pieces(TRAIN, max_length=6, min_count=1)
    log_probabilities = estimate_log_probabilities(TRAIN, pieces, 8)
    assert all(math.isfinite(value) for value in log_probabilities.values())


def test_estimating_over_an_empty_piece_set_is_refused() -> None:
    with pytest.raises(ConfigError, match="empty piece set"):
        estimate_log_probabilities(TRAIN, [])


def test_estimating_from_no_words_keeps_the_uniform_start() -> None:
    log_probabilities = estimate_log_probabilities([], ["a", "b"], 3)
    assert log_probabilities["a"] == pytest.approx(-math.log(2.0))


def test_more_iterations_do_not_lower_the_likelihood() -> None:
    pieces = candidate_pieces(TRAIN, max_length=5, min_count=2)
    scores = [
        corpus_log_likelihood(TRAIN, estimate_log_probabilities(TRAIN, pieces, count))
        for count in (1, 3, 8)
    ]
    assert all(later >= earlier - 1e-6 for earlier, later in zip(scores, scores[1:], strict=False))
