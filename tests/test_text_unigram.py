"""The Viterbi path is checked against an exhaustive search over segmentations."""

from __future__ import annotations

import itertools
import math

import pytest

from echoscribe.errors import TokenizerError
from echoscribe.text.unigram import best_segmentation


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
