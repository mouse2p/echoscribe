"""The lexicon is checked against the phone inventory it claims to use."""

from __future__ import annotations

import pytest

from echoscribe.errors import TokenizerError
from echoscribe.synth.lexicon import (
    ENGLISH_SPELLINGS,
    MANDARIN_SYLLABLES,
    Coverage,
    Lexicon,
    MandarinSyllable,
    PhoneSequence,
    TokenSpan,
    coverage_of,
    english_vocabulary,
    mandarin_vocabulary,
    transcribe,
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


def test_lookup_classifies_english_and_chinese_tokens() -> None:
    lexicon = Lexicon()
    assert lexicon.classify("hello") == "en"
    assert lexicon.classify("你") == "zh"


def test_lookup_rejects_an_unknown_token_with_the_table_sizes() -> None:
    with pytest.raises(TokenizerError) as info:
        Lexicon().classify("kazan")
    message = str(info.value)
    assert "unknown token 'kazan'" in message
    assert str(len(ENGLISH_SPELLINGS)) in message
    assert str(len(MANDARIN_SYLLABLES)) in message


def test_word_phones_resolves_the_spelling() -> None:
    phones = Lexicon().word_phones("hello")
    assert [phone.symbol for phone in phones] == ["hh", "ah", "l", "ow"]


def test_word_phones_rejects_a_chinese_token() -> None:
    with pytest.raises(TokenizerError, match="not an English word"):
        Lexicon().word_phones("你")


def test_syllable_phones_puts_the_initial_before_the_final() -> None:
    phones = Lexicon().syllable_phones("你")
    assert [phone.symbol for phone in phones] == ["n", "i"]


def test_syllable_phones_rejects_an_english_token() -> None:
    with pytest.raises(TokenizerError, match="not a character"):
        Lexicon().syllable_phones("hello")


def test_phones_for_dispatches_on_the_token() -> None:
    lexicon = Lexicon()
    assert lexicon.phones_for("hello") == lexicon.word_phones("hello")
    assert lexicon.phones_for("好") == lexicon.syllable_phones("好")


def test_a_custom_lexicon_overrides_the_default_tables() -> None:
    lexicon = Lexicon(english={"zog": ("z", "aa", "g")}, mandarin={})
    assert [phone.symbol for phone in lexicon.phones_for("zog")] == ["z", "aa", "g"]
    with pytest.raises(TokenizerError):
        lexicon.classify("hello")


def test_every_english_word_resolves() -> None:
    lexicon = Lexicon()
    for word in ENGLISH_SPELLINGS:
        assert lexicon.word_phones(word)


def test_every_character_resolves() -> None:
    lexicon = Lexicon()
    for character in MANDARIN_SYLLABLES:
        assert lexicon.syllable_phones(character)


def test_transcribe_returns_a_phone_sequence_value() -> None:
    sequence = transcribe(["hello"])
    assert isinstance(sequence, PhoneSequence)
    assert isinstance(sequence.spans[0], TokenSpan)


def test_spans_cover_the_phone_sequence_exactly() -> None:
    sequence = transcribe(["hello", "你", "好", "world"])
    cursor = 0
    for span in sequence.spans:
        assert span.start == cursor
        cursor = span.stop
    assert cursor == len(sequence.phones)


def test_spans_record_the_right_language() -> None:
    sequence = transcribe(["hello", "你"])
    assert [span.language for span in sequence.spans] == ["en", "zh"]


def test_a_chinese_token_contributes_two_phones_and_an_english_word_more() -> None:
    sequence = transcribe(["你", "hello"])
    assert sequence.spans[0].length == 2
    assert sequence.spans[1].length == 4
    assert sequence.symbols() == ("n", "i", "hh", "ah", "l", "ow")


def test_tokens_round_trip_through_the_spans() -> None:
    tokens = ["please", "看", "这", "system", "好"]
    sequence = transcribe(tokens)
    assert sequence.tokens() == tuple(tokens)
    for token in tokens:
        expected = Lexicon().phones_for(token)
        assert sequence.phones_of(token) == expected


def test_phones_of_rejects_a_foreign_token() -> None:
    sequence = transcribe(["hello"])
    with pytest.raises(TokenizerError, match="is not part of this sequence"):
        sequence.phones_of("world")


def test_transcribe_rejects_an_empty_sequence() -> None:
    with pytest.raises(TokenizerError, match="cannot transcribe an empty"):
        transcribe([])


def test_transcribe_rejects_an_unknown_token() -> None:
    with pytest.raises(TokenizerError, match="unknown token"):
        transcribe(["hello", "kazan"])


def test_a_repeated_token_gets_two_spans() -> None:
    sequence = transcribe(["好", "好"])
    assert len(sequence.spans) == 2
    assert sequence.spans[0].start == 0
    assert sequence.spans[1].start == 2
    assert len(sequence.phones) == 4


def test_a_phone_sequence_is_immutable() -> None:
    sequence = transcribe(["hello"])
    with pytest.raises(AttributeError):
        sequence.phones = ()  # type: ignore[misc]


def test_a_token_span_reports_its_stop() -> None:
    span = TokenSpan("hello", "en", 3, 4)
    assert span.stop == 7


def test_transcribe_is_deterministic() -> None:
    tokens = ["start", "你", "test"]
    assert transcribe(tokens).symbols() == transcribe(tokens).symbols()


def test_coverage_is_a_small_value_object() -> None:
    report = coverage_of(["hello", "你"])
    assert isinstance(report, Coverage)
    assert report.total == 2
    assert report.known == ("hello", "你")


def test_coverage_of_a_fully_known_list_is_complete() -> None:
    report = coverage_of(["hello", "你", "好"])
    assert report.total == 3
    assert report.known == ("hello", "你", "好")
    assert report.unknown == ()
    assert report.ratio == pytest.approx(1.0)
    assert report.complete


def test_coverage_splits_known_from_unknown_in_order() -> None:
    report = coverage_of(["hello", "kazan", "你", "zzz"])
    assert report.known == ("hello", "你")
    assert report.unknown == ("kazan", "zzz")
    assert report.ratio == pytest.approx(0.5)
    assert not report.complete


def test_coverage_of_nothing_is_complete_by_convention() -> None:
    report = coverage_of([])
    assert report.total == 0
    assert report.ratio == 1.0
    assert report.complete


def test_coverage_honours_a_custom_lexicon() -> None:
    lexicon = Lexicon(english={"zog": ("z", "aa", "g")}, mandarin={})
    report = coverage_of(["zog", "hello"], lexicon)
    assert report.known == ("zog",)
    assert report.unknown == ("hello",)


def test_coverage_does_not_mutate_its_input() -> None:
    tokens = ["hello", "kazan"]
    coverage_of(tokens)
    assert tokens == ["hello", "kazan"]
