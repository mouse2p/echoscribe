"""The lexicon is checked against the phone inventory it claims to use."""

from __future__ import annotations

import pytest

from echoscribe.errors import TokenizerError
from echoscribe.synth.lexicon import (
    ENGLISH_SPELLINGS,
    MANDARIN_SYLLABLES,
    MandarinSyllable,
    english_vocabulary,
    mandarin_vocabulary,
)
from echoscribe.synth.phones import ENGLISH_PHONES, FINALS, INITIALS


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


def test_the_syllable_table_is_big_enough_for_a_corpus() -> None:
    assert len(MANDARIN_SYLLABLES) >= 20
    assert mandarin_vocabulary() == tuple(MANDARIN_SYLLABLES)


def test_every_initial_and_final_exists_in_the_phone_tables() -> None:
    for character, syllable in MANDARIN_SYLLABLES.items():
        assert syllable.initial in INITIALS, character
        assert syllable.final in FINALS, character


def test_every_tone_is_one_of_the_four_lexical_tones() -> None:
    assert {syllable.tone for syllable in MANDARIN_SYLLABLES.values()} == {1, 2, 3, 4}


def test_romanization_joins_initial_final_and_tone() -> None:
    assert MANDARIN_SYLLABLES["你"].romanization == "ni3"
    assert MANDARIN_SYLLABLES["中"].romanization == "zhong1"
    assert MANDARIN_SYLLABLES["去"].romanization == "qv4"


def test_a_syllable_rejects_an_out_of_range_tone() -> None:
    with pytest.raises(TokenizerError, match="tone must be 1-4"):
        MandarinSyllable("x", "n", "i", 5)
    with pytest.raises(TokenizerError, match="tone must be 1-4"):
        MandarinSyllable("x", "n", "i", 0)


def test_a_syllable_rejects_an_empty_character() -> None:
    with pytest.raises(TokenizerError, match="needs a character"):
        MandarinSyllable("", "n", "i", 1)


def test_two_characters_may_share_a_pronunciation() -> None:
    # 他 and 她 are homophones; the lexicon is a map from characters, not from
    # pronunciations, so both must be present.
    assert MANDARIN_SYLLABLES["他"].romanization == MANDARIN_SYLLABLES["她"].romanization
    assert MANDARIN_SYLLABLES["他"] is not MANDARIN_SYLLABLES["她"]


def test_the_syllable_table_covers_several_initials() -> None:
    initials = {syllable.initial for syllable in MANDARIN_SYLLABLES.values()}
    assert len(initials) >= 10
