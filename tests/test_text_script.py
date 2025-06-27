"""Script classification decides the metric, so every class is pinned."""

from __future__ import annotations

import pytest

from echoscribe.errors import TokenizerError
from echoscribe.text.script import SCRIPTS, is_han, script_of


@pytest.mark.parametrize(
    ("character", "expected"),
    [
        ("你", "han"),
        ("好", "han"),
        ("a", "latin"),
        ("Z", "latin"),
        ("é", "latin"),
        ("7", "digit"),
        (",", "punctuation"),
        ("。", "punctuation"),
        (" ", "space"),
        ("\u3000", "space"),
        ("\u05d0", "other"),
        ("\u0915", "other"),
    ],
)
def test_script_classification(character: str, expected: str) -> None:
    assert script_of(character) == expected


def test_every_declared_script_is_reachable() -> None:
    samples = {"han": "你", "latin": "a", "digit": "1", "punctuation": ".", "space": " "}
    assert set(SCRIPTS) == set(samples) | {"other"}
    for script, character in samples.items():
        assert script_of(character) == script


def test_script_of_rejects_a_multi_character_string() -> None:
    with pytest.raises(TokenizerError, match="classifies one character"):
        script_of("ab")
    with pytest.raises(TokenizerError, match="classifies one character"):
        script_of("")


def test_is_han_matches_the_classifier() -> None:
    assert is_han("你")
    assert not is_han("a")
    assert not is_han("1")


def test_the_cjk_extension_blocks_count_as_han() -> None:
    assert script_of("\u3400") == "han"
    assert script_of("\u9fff") == "han"
    assert script_of("\uf900") == "han"


def test_the_han_range_boundaries_are_exact() -> None:
    assert script_of("\u3400") == "han"
    assert script_of("\u33ff") != "han"
    assert script_of("\u4dbf") == "han"
    assert script_of("\u4dc0") != "han"
    assert script_of("\u4e00") == "han"
    assert script_of("\u9fff") == "han"
    assert script_of("\ua000") != "han"


def test_hebrew_and_devanagari_fall_through_to_other() -> None:
    # Neither is Han, and calling them Latin would silently score them per word
    # instead of per character.
    assert script_of("\u05d0") == "other"
    assert script_of("\u0915") == "other"
