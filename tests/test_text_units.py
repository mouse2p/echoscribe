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
