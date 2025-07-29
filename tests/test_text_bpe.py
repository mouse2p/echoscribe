"""BPE is a pure function of the corpus, so every stage is pinned."""

from __future__ import annotations

from collections import Counter

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.bpe import WORD_END, count_pairs, merge_pair, word_symbols


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
