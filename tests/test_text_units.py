"""Unit symbols are the decoder's output alphabet, so they must round-trip."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError, TokenizerError
from echoscribe.synth.lexicon import MANDARIN_SYLLABLES, MandarinSyllable
from echoscribe.text.units import (
    TONE_MARKS,
    UNIT_LEVELS,
    UNIT_SEPARATOR,
    ZERO_INITIAL_MARK,
    ToneUnit,
    syllable_units,
    utterance_units,
)


def test_a_unit_symbol_joins_its_three_parts() -> None:
    assert ToneUnit("n", "i", 3).symbol == "n.i.3"
    assert ToneUnit("zh", "ong", 1).symbol == "zh.ong.1"


def test_a_zero_initial_uses_the_placeholder() -> None:
    assert ToneUnit("", "a", 1).symbol == f"{ZERO_INITIAL_MARK}.a.1"


def test_symbols_round_trip_through_parse() -> None:
    for syllable in MANDARIN_SYLLABLES.values():
        unit = ToneUnit.of(syllable)
        assert ToneUnit.parse(unit.symbol) == unit


def test_parse_recovers_a_zero_initial() -> None:
    assert ToneUnit.parse("_.a.1") == ToneUnit("", "a", 1)


def test_parse_rejects_a_wrong_part_count() -> None:
    with pytest.raises(TokenizerError, match="is not a unit symbol"):
        ToneUnit.parse("ni3")
    with pytest.raises(TokenizerError, match="is not a unit symbol"):
        ToneUnit.parse("n.i.3.4")


def test_parse_rejects_an_empty_final() -> None:
    with pytest.raises(TokenizerError, match="empty final"):
        ToneUnit.parse("n..3")


def test_parse_rejects_an_unknown_tone() -> None:
    with pytest.raises(TokenizerError, match="which is not one of"):
        ToneUnit.parse("n.i.7")
    with pytest.raises(TokenizerError, match="which is not one of"):
        ToneUnit.parse("n.i.x")


def test_the_display_form_uses_a_diacritic() -> None:
    assert ToneUnit("n", "i", 3).display == "ni\u02c7"
    assert ToneUnit("", "a", 1).display == "a\u02c9"


def test_every_tone_has_a_distinct_mark() -> None:
    assert len(set(TONE_MARKS.values())) == 4
    assert set(TONE_MARKS) == {1, 2, 3, 4}


def test_a_unit_rejects_a_separator_inside_a_part() -> None:
    with pytest.raises(ConfigError, match="must not contain"):
        ToneUnit("n.x", "i", 3)


def test_a_unit_needs_a_final() -> None:
    with pytest.raises(ConfigError, match="needs a final"):
        ToneUnit("n", "", 3)


def test_a_unit_rejects_an_out_of_range_tone() -> None:
    with pytest.raises(ConfigError, match="tone must be one of"):
        ToneUnit("n", "i", 5)


def test_the_declared_levels_are_syllable_and_phone() -> None:
    assert UNIT_LEVELS == ("syllable", "phone")
    assert UNIT_SEPARATOR == "."


def test_a_unit_is_built_from_a_lexicon_syllable() -> None:
    syllable = MandarinSyllable("X", "sh", "ang", 2)
    assert ToneUnit.of(syllable) == ToneUnit("sh", "ang", 2)


def test_distinct_pronunciations_give_distinct_symbols() -> None:
    syllables = list(MANDARIN_SYLLABLES.values())
    symbols = {ToneUnit.of(syllable).symbol for syllable in syllables}
    pronunciations = {(s.initial, s.final, s.tone) for s in syllables}
    # Homophones collapse onto one symbol, which is correct: a phone-level decoder
    # cannot tell them apart either, and the transcript still records both.
    assert len(symbols) == len(pronunciations)


def test_homophones_share_a_symbol() -> None:
    # 他 and 她 differ as characters but not as units, which is exactly what a
    # phone-level decoder can distinguish and no more.
    assert ToneUnit.of(MANDARIN_SYLLABLES["他"]) == ToneUnit.of(MANDARIN_SYLLABLES["她"])


def test_the_four_tones_of_one_syllable_give_four_symbols() -> None:
    symbols = {ToneUnit("h", "a", tone).symbol for tone in (1, 2, 3, 4)}
    assert len(symbols) == 4


def test_symbols_are_ascii_and_tokenizer_safe() -> None:
    for syllable in MANDARIN_SYLLABLES.values():
        symbol = ToneUnit.of(syllable).symbol
        assert symbol.isascii()
        assert " " not in symbol
        assert symbol.strip() == symbol


def test_an_english_word_expands_to_its_phones() -> None:
    assert utterance_units(("hello",), ("en",)) == ("hh", "ah", "l", "ow")


def test_a_chinese_character_expands_to_one_toned_unit() -> None:
    assert utterance_units(("你",), ("zh",)) == ("n.i.3",)


def test_a_code_switched_utterance_concatenates_both() -> None:
    units = utterance_units(("hello", "你", "好"), ("en", "zh", "zh"))
    assert units == ("hh", "ah", "l", "ow", "n.i.3", "h.ao.3")


def test_the_phone_level_splits_a_syllable_in_two() -> None:
    assert syllable_units(MANDARIN_SYLLABLES["你"], "phone") == ("n", "_.i.3")


def test_the_phone_level_of_a_zero_initial_syllable_is_one_unit() -> None:
    syllable = MandarinSyllable("X", "", "a", 1)
    assert syllable_units(syllable, "phone") == ("_.a.1",)


def test_utterance_units_honour_the_level() -> None:
    units = utterance_units(("你",), ("zh",), level="phone")
    assert units == ("n", "_.i.3")


def test_an_unknown_level_is_rejected() -> None:
    with pytest.raises(ConfigError, match="level must be one of"):
        syllable_units(MANDARIN_SYLLABLES["你"], "morpheme")
    with pytest.raises(ConfigError, match="level must be one of"):
        utterance_units(("你",), ("zh",), level="morpheme")


def test_a_label_count_mismatch_is_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot carry"):
        utterance_units(("hello", "你"), ("en",))


def test_units_of_an_empty_utterance_are_empty() -> None:
    assert utterance_units((), ()) == ()


def test_a_custom_lexicon_is_honoured() -> None:
    from echoscribe.synth.lexicon import Lexicon

    lexicon = Lexicon(english={"zog": ("z", "aa")}, mandarin={})
    assert utterance_units(("zog",), ("en",), lexicon) == ("z", "aa")
