"""The lexicon is checked against the phone inventory it claims to use."""

from __future__ import annotations

from echoscribe.synth.lexicon import ENGLISH_SPELLINGS, english_vocabulary
from echoscribe.synth.phones import ENGLISH_PHONES


def test_the_vocabulary_is_large_enough_to_force_a_choice() -> None:
    assert len(ENGLISH_SPELLINGS) >= 40


def test_english_vocabulary_matches_the_table_order() -> None:
    assert english_vocabulary() == tuple(ENGLISH_SPELLINGS)


def test_every_spelling_is_a_non_empty_phone_tuple() -> None:
    for word, phones in ENGLISH_SPELLINGS.items():
        assert isinstance(phones, tuple), word
        assert phones, word


def test_every_spelled_phone_exists_in_the_inventory() -> None:
    missing = {
        (word, symbol)
        for word, phones in ENGLISH_SPELLINGS.items()
        for symbol in phones
        if symbol not in ENGLISH_PHONES
    }
    assert not missing


def test_the_vocabulary_is_case_uniform() -> None:
    assert all(word == word.lower() for word in ENGLISH_SPELLINGS)
    assert all(word.isalpha() for word in ENGLISH_SPELLINGS)


def test_no_two_words_share_a_spelling() -> None:
    spellings = list(ENGLISH_SPELLINGS.values())
    assert len(spellings) == len(set(spellings))


def test_the_vocabulary_covers_most_of_the_phone_inventory() -> None:
    # A lexicon that only exercises five phones cannot stress a decoder.
    used = {symbol for phones in ENGLISH_SPELLINGS.values() for symbol in phones}
    assert len(used) >= 25
    assert {"sil"} | used <= set(ENGLISH_PHONES)
