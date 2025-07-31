"""BPE is a pure function of the corpus, so every stage is pinned."""

from __future__ import annotations

from collections import Counter

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.bpe import (
    WORD_END,
    count_pairs,
    learn_bpe,
    merge_pair,
    word_symbols,
)


def test_a_word_is_spelled_with_an_end_marker() -> None:
    assert word_symbols("ab") == ("a", "b", WORD_END)


def test_a_single_character_word_has_two_symbols() -> None:
    assert word_symbols("a") == ("a", WORD_END)


def test_an_empty_word_is_refused() -> None:
    with pytest.raises(ConfigError, match="cannot spell an empty word"):
        word_symbols("")


def test_a_non_string_word_is_refused() -> None:
    with pytest.raises(ConfigError, match="must be a string"):
        word_symbols(7)  # type: ignore[arg-type]


def test_pair_counts_cover_every_adjacent_pair() -> None:
    assert count_pairs(["ab"]) == Counter({("a", "b"): 1, ("b", WORD_END): 1})


def test_repeats_are_counted() -> None:
    counts = count_pairs(["ab", "ab", "ab"])
    assert counts[("a", "b")] == 3


def test_a_single_character_word_has_one_pair() -> None:
    assert count_pairs(["a"]) == Counter({("a", WORD_END): 1})


def test_pair_counts_of_nothing_are_empty() -> None:
    assert count_pairs([]) == Counter()


def test_a_merge_joins_every_occurrence() -> None:
    assert merge_pair(("a", "b", "a", "b"), ("a", "b")) == ("ab", "ab")


def test_a_merge_does_not_overlap_itself() -> None:
    assert merge_pair(("a", "a", "a"), ("a", "a")) == ("aa", "a")


def test_a_merge_that_does_not_occur_changes_nothing() -> None:
    symbols = ("x", "y", WORD_END)
    assert merge_pair(symbols, ("a", "b")) == symbols


def test_a_merge_can_produce_the_end_marker_pair() -> None:
    assert merge_pair(("a", WORD_END), ("a", WORD_END)) == (f"a{WORD_END}",)


def test_merging_an_empty_sequence_is_empty() -> None:
    assert merge_pair((), ("a", "b")) == ()


LOW = 3 * ["low"] + 2 * ["lowest"] + 4 * ["newer"] + 4 * ["wider"] + 1 * ["new"]


def test_the_first_merge_is_the_most_frequent_pair() -> None:
    merges = learn_bpe(LOW, 1)
    counts = count_pairs(LOW)
    best = max(counts.values())
    tied = sorted(pair for pair, count in counts.items() if count == best)
    assert merges[0] == tied[0]


def test_learning_stops_early_when_nothing_is_left_to_merge() -> None:
    assert learn_bpe(["ab"], 500) == (("a", "b"), ("ab", WORD_END))


def test_more_merges_are_learned_when_asked_for() -> None:
    assert len(learn_bpe(LOW, 3)) == 3
    assert len(learn_bpe(LOW, 10)) > 3


def test_the_merge_order_is_a_function_of_the_counts() -> None:
    import random

    shuffled = list(LOW)
    random.Random(0).shuffle(shuffled)
    assert learn_bpe(shuffled, 12) == learn_bpe(LOW, 12)


def test_learning_from_nothing_yields_no_merges() -> None:
    assert learn_bpe([], 10) == ()


def test_a_non_positive_merge_count_is_refused() -> None:
    with pytest.raises(ConfigError, match="num_merges must be positive"):
        learn_bpe(LOW, 0)


def test_merges_are_pairs_of_strings() -> None:
    for pair in learn_bpe(LOW, 6):
        assert isinstance(pair, tuple) and len(pair) == 2
        assert all(isinstance(part, str) and part for part in pair)


def test_a_tie_is_broken_lexicographically() -> None:
    # All four pairs occur once.  The smallest by (first, second) wins, and "</w>"
    # sorts before any letter because "<" is below "a" -- a consequence worth
    # pinning, since it decides the tail of every merge list.
    assert learn_bpe(["ab", "ba"], 1) == (("a", WORD_END),)


def test_frequency_beats_lexicographic_order() -> None:
    # (z, z) occurs four times against (a, b) once, so frequency wins outright.
    assert learn_bpe(["zzz", "zzz", "ab"], 1) == (("z", "z"),)


def test_a_longer_corpus_produces_a_prefix_of_the_same_order() -> None:
    short = learn_bpe(LOW, 4)
    long = learn_bpe(LOW, 12)
    assert long[:4] == short


def test_learning_twice_gives_an_identical_tuple() -> None:
    assert learn_bpe(LOW, 8) == learn_bpe(LOW, 8)


def test_a_corpus_of_one_repeated_word_still_merges_to_the_end_marker() -> None:
    merges = learn_bpe(["abc"] * 5, 10)
    assert merges[-1] == ("abc", WORD_END) or merges[-1][1] == WORD_END
