"""Segmentation decides what a scorer counts, so its boundaries are pinned."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.segment import (
    INTERNAL_WORD_PUNCTUATION,
    WORD_SCRIPTS,
    SegmentConfig,
    Token,
)


def test_a_token_knows_its_own_span() -> None:
    token = Token("hello", "latin", 3)
    assert token.stop == 8
    assert token.is_word


def test_a_han_token_is_not_a_word() -> None:
    assert not Token("你", "han", 0).is_word


def test_word_scripts_are_latin_and_digit() -> None:
    assert WORD_SCRIPTS == ("latin", "digit")
    assert INTERNAL_WORD_PUNCTUATION == ("-", "'")


def test_a_token_rejects_empty_text_or_a_negative_start() -> None:
    with pytest.raises(ConfigError, match="must not be empty"):
        Token("", "latin", 0)
    with pytest.raises(ConfigError, match="start must not be negative"):
        Token("a", "latin", -1)


def test_config_defaults_drop_punctuation() -> None:
    config = SegmentConfig()
    assert not config.keep_punctuation
    assert not config.split_on_hyphen
    assert config.min_token_length == 1


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2"])
def test_a_bad_minimum_length_is_rejected(bad) -> None:
    with pytest.raises(ConfigError, match="min_token_length must be an integer"):
        SegmentConfig(min_token_length=bad)


def test_config_is_frozen() -> None:
    with pytest.raises(AttributeError):
        SegmentConfig().keep_punctuation = True  # type: ignore[misc]
