"""The Viterbi path is checked against an exhaustive search over segmentations."""

from __future__ import annotations

import itertools
import math

import pytest

from echoscribe.errors import ConfigError, TokenizerError
from echoscribe.text.unigram import (
    UnigramModel,
    best_segmentation,
    candidate_pieces,
    corpus_log_likelihood,
    estimate_log_probabilities,
    learn_unigram,
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


def test_a_learned_model_has_the_requested_size() -> None:
    model = learn_unigram(TRAIN, target_size=20, max_length=5)
    assert len(model) == 20
    assert isinstance(model, UnigramModel)


def test_a_learned_model_encodes_every_training_word() -> None:
    model = learn_unigram(TRAIN, target_size=20, max_length=5)
    for word in set(TRAIN):
        pieces = model.encode(word)
        assert model.decode(pieces) == word


def test_encoding_is_deterministic() -> None:
    model = learn_unigram(TRAIN, target_size=18, max_length=5)
    assert model.encode("newer") == model.encode("newer")


def test_learning_twice_gives_the_same_model() -> None:
    left = learn_unigram(TRAIN, target_size=16, max_length=4)
    right = learn_unigram(TRAIN, target_size=16, max_length=4)
    assert left.pieces == right.pieces
    assert left.log_probabilities == right.log_probabilities


def test_a_smaller_target_still_covers_the_corpus() -> None:
    # Ten distinct characters appear in the corpus, so ten pieces always suffice;
    # anything at or above that is reachable by the greedy prune.
    model = learn_unigram(TRAIN, target_size=12, max_length=4)
    for word in set(TRAIN):
        assert model.decode(model.encode(word)) == word


def test_probabilities_are_sorted_consistently_with_the_pieces() -> None:
    model = learn_unigram(TRAIN, target_size=12, max_length=4)
    for piece in model.pieces:
        assert model.log_probability(piece) == pytest.approx(
            model.log_probabilities[model.pieces.index(piece)]
        )


def test_an_unknown_piece_scores_minus_infinity() -> None:
    model = learn_unigram(TRAIN, target_size=12, max_length=4)
    assert model.log_probability("\u0000not-a-piece") == float("-inf")


def test_a_target_larger_than_the_candidates_is_refused() -> None:
    with pytest.raises(ConfigError, match="fewer than the"):
        learn_unigram(TRAIN, target_size=10000, max_length=2)


def test_learning_from_no_words_is_refused() -> None:
    with pytest.raises(ConfigError, match="from no words"):
        learn_unigram([], target_size=5)


def test_a_model_rejects_mismatched_pieces_and_probabilities() -> None:
    with pytest.raises(ConfigError, match="cannot carry"):
        UnigramModel(("a", "b"), (math.log(0.5),))


def test_a_model_rejects_a_repeated_piece() -> None:
    with pytest.raises(ConfigError, match="repeated piece"):
        UnigramModel(("a", "a"), (math.log(0.5), math.log(0.5)))


def test_the_text_subpackage_re_exports_the_unigram_model() -> None:
    from echoscribe import text

    assert text.UnigramModel is UnigramModel
    assert text.learn_unigram is learn_unigram
    for name in ("UnigramModel", "learn_unigram", "best_segmentation"):
        assert name in text.__all__
