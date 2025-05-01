"""The inventory is validated structurally: uniqueness, ranges, completeness."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.phones import (
    ENGLISH_PHONES,
    FINALS,
    INITIALS,
    MANNERS,
    ZERO_INITIAL,
    Phone,
)


def _vowel(symbol: str = "iy", **overrides) -> Phone:
    fields = {
        "symbol": symbol,
        "manner": "vowel",
        "voiced": True,
        "duration_ms": 120.0,
        "formants": ((300.0, 60.0), (2300.0, 120.0), (3000.0, 160.0)),
    }
    fields.update(overrides)
    return Phone(**fields)


def test_a_phone_keeps_its_fields() -> None:
    phone = _vowel()
    assert phone.symbol == "iy"
    assert phone.manner == "vowel"
    assert phone.voiced
    assert phone.gain == 1.0
    assert not phone.turbulent
    assert len(phone.formants) == 3


def test_manners_cover_every_class_the_synthesizer_renders() -> None:
    assert MANNERS == (
        "silence",
        "vowel",
        "nasal",
        "plosive",
        "affricate",
        "fricative",
        "approximant",
    )


def test_silence_is_the_only_silent_manner() -> None:
    assert _vowel(manner="silence").is_silent
    assert not _vowel().is_silent
    assert not _vowel(manner="fricative").is_silent


def test_a_phone_is_immutable() -> None:
    phone = _vowel()
    with pytest.raises(AttributeError):
        phone.duration_ms = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("manner", ["phone", "", "VOWEL", "stop"])
def test_an_unknown_manner_is_rejected(manner: str) -> None:
    with pytest.raises(ConfigError, match="manner must be one of"):
        _vowel(manner=manner)


@pytest.mark.parametrize(
    "overrides",
    [
        {"duration_ms": 0.0},
        {"duration_ms": -5.0},
        {"gain": 0.0},
        {"gain": 1.5},
        {"formants": ()},
        {"formants": ((0.0, 60.0),)},
        {"formants": ((300.0, 0.0),)},
        {"symbol": ""},
    ],
)
def test_invalid_phones_are_rejected(overrides) -> None:
    with pytest.raises(ConfigError):
        _vowel(**overrides)


def test_the_rejection_message_names_the_offending_formant() -> None:
    with pytest.raises(ConfigError, match="formant 1 bandwidth must be positive"):
        _vowel(formants=((300.0, 60.0), (2000.0, -1.0)))


def test_the_english_inventory_covers_vowels_and_consonants() -> None:
    vowels = {name for name, phone in ENGLISH_PHONES.items() if phone.manner == "vowel"}
    assert len(vowels) == 13
    assert {"iy", "aa", "eh", "ow"} <= vowels


def test_every_english_phone_declares_a_known_manner() -> None:
    present = {phone.manner for phone in ENGLISH_PHONES.values()}
    assert present <= set(MANNERS)
    # Affricates belong to the tone-language onsets only.
    assert present == set(MANNERS) - {"affricate"}


def test_exactly_one_phone_is_silent() -> None:
    silent = [name for name, phone in ENGLISH_PHONES.items() if phone.is_silent]
    assert silent == ["sil"]


def test_silence_has_a_negligible_gain() -> None:
    assert ENGLISH_PHONES["sil"].gain < 0.01


def test_voiced_consonants_are_marked_voiced() -> None:
    for symbol in ("b", "d", "g", "v", "z", "m", "n", "ng", "l", "r", "w", "y"):
        assert ENGLISH_PHONES[symbol].voiced, symbol
    for symbol in ("p", "t", "k", "f", "s", "sh", "th", "hh"):
        assert not ENGLISH_PHONES[symbol].voiced, symbol


def test_fricatives_are_turbulent_and_vowels_are_not() -> None:
    for name, phone in ENGLISH_PHONES.items():
        if phone.manner == "fricative":
            assert phone.turbulent, name
        elif phone.manner == "vowel":
            assert not phone.turbulent, name


def test_plosives_are_shorter_than_vowels() -> None:
    longest_plosive = max(p.duration_ms for p in ENGLISH_PHONES.values() if p.manner == "plosive")
    shortest_vowel = min(p.duration_ms for p in ENGLISH_PHONES.values() if p.manner == "vowel")
    assert longest_plosive < shortest_vowel


def test_vowel_first_formants_spread_the_f1_f2_plane() -> None:
    vowels = [p for p in ENGLISH_PHONES.values() if p.manner == "vowel"]
    first = [p.formants[0][0] for p in vowels]
    second = [p.formants[1][0] for p in vowels]
    assert max(first) - min(first) > 300.0
    assert max(second) - min(second) > 1000.0


def test_no_two_phones_share_a_symbol() -> None:
    symbols = [phone.symbol for phone in ENGLISH_PHONES.values()]
    assert len(symbols) == len(set(symbols))
    assert set(symbols) == set(ENGLISH_PHONES)


def test_every_formant_centre_is_below_nyquist_at_16k() -> None:
    for name, phone in ENGLISH_PHONES.items():
        for centre, bandwidth in phone.formants:
            assert centre < 8000.0, name
            assert bandwidth < centre, name


def test_the_initial_table_covers_the_usual_onsets() -> None:
    assert {"b", "p", "m", "d", "t", "n", "l", "g", "k", "h"} <= set(INITIALS)
    assert len(INITIALS) == 20


def test_the_final_table_covers_the_usual_nuclei() -> None:
    assert {"a", "o", "e", "i", "u", "ai", "ei", "ao", "ou", "an", "en"} <= set(FINALS)
    assert len(FINALS) == 15


def test_no_initial_is_also_a_final() -> None:
    assert not set(INITIALS) & set(FINALS)


def test_every_final_is_a_voiced_vowel() -> None:
    for symbol, phone in FINALS.items():
        assert phone.manner == "vowel", symbol
        assert phone.voiced, symbol
        assert not phone.turbulent, symbol


def test_finals_outlast_initials() -> None:
    longest_initial = max(phone.duration_ms for phone in INITIALS.values())
    shortest_final = min(phone.duration_ms for phone in FINALS.values())
    assert longest_initial < shortest_final


def test_the_zero_initial_is_the_empty_spelling() -> None:
    assert ZERO_INITIAL == ""
    assert ZERO_INITIAL not in INITIALS


def test_affricates_are_turbulent_and_voiceless() -> None:
    affricates = [p for p in INITIALS.values() if p.manner == "affricate"]
    assert len(affricates) == 6
    for phone in affricates:
        assert phone.turbulent
        assert not phone.voiced


def test_every_initial_and_final_survives_phone_validation() -> None:
    # Building the tables already ran __post_init__; assert the invariants the
    # synthesizer depends on rather than trusting construction alone.
    for phone in [*INITIALS.values(), *FINALS.values()]:
        assert phone.duration_ms > 0.0
        assert 0.0 < phone.gain <= 1.0
        assert all(centre > bandwidth > 0.0 for centre, bandwidth in phone.formants)
