"""The inventory is validated structurally: uniqueness, ranges, completeness."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.phones import MANNERS, Phone


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
