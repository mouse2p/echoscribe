"""Segmentation decides what a scorer counts, so its boundaries are pinned."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.segment import (
    INTERNAL_WORD_PUNCTUATION,
    WORD_SCRIPTS,
    SegmentConfig,
    Token,
    segment,
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


def test_offsets_point_back_at_the_source_text() -> None:
    text = "hello 你好 world"
    for token in segment(text):
        assert text[token.start : token.stop] == token.text


def test_tokens_are_ordered_and_do_not_overlap() -> None:
    text = "start 你 stop 好 end"
    tokens = segment(text)
    cursor = 0
    for token in tokens:
        assert token.start >= cursor
        cursor = token.stop


def test_scripts_are_recorded_per_token() -> None:
    tokens = segment("hello 你好 42")
    assert [(token.text, token.script) for token in tokens] == [
        ("hello", "latin"),
        ("你", "han"),
        ("好", "han"),
        ("42", "digit"),
    ]


def test_an_empty_string_has_no_tokens() -> None:
    assert segment("") == ()
    assert segment("   ") == ()


def test_a_non_string_is_rejected() -> None:
    with pytest.raises(ConfigError, match="needs a string"):
        segment(42)  # type: ignore[arg-type]


def test_segmentation_is_deterministic() -> None:
    text = "mixed 混合 text 文本 123"
    assert segment(text) == segment(text)


def test_a_word_language_splits_on_whitespace() -> None:
    assert [token.text for token in segment("hello there world")] == [
        "hello",
        "there",
        "world",
    ]


def test_a_character_language_splits_per_character() -> None:
    assert [token.text for token in segment("你好世界")] == ["你", "好", "世", "界"]


def test_a_code_switch_without_spaces_still_splits() -> None:
    assert [token.text for token in segment("hello你好world")] == [
        "hello",
        "你",
        "好",
        "world",
    ]


def test_an_apostrophe_stays_inside_its_word() -> None:
    assert [token.text for token in segment("don't stop")] == ["don't", "stop"]


def test_a_hyphen_stays_inside_its_word_by_default() -> None:
    assert [token.text for token in segment("code-switch tools")] == ["code-switch", "tools"]


def test_a_trailing_hyphen_ends_the_word() -> None:
    assert [token.text for token in segment("well- known")] == ["well", "known"]


def test_digits_next_to_letters_stay_separate() -> None:
    assert [token.text for token in segment("room42")] == ["room", "42"]


def test_a_decimal_number_splits_at_its_point() -> None:
    assert [token.text for token in segment("3.14")] == ["3", "14"]


def test_a_minimum_length_drops_short_tokens() -> None:
    config = SegmentConfig(min_token_length=2)
    # Single Han characters are one token each, so a minimum of two removes them
    # along with the single-letter English word.  That is the documented effect of
    # the flag, not an accident of the scan.
    assert [token.text for token in segment("I am 你好 ok", config)] == ["am", "ok"]


def test_a_minimum_length_never_drops_a_two_character_word() -> None:
    config = SegmentConfig(min_token_length=2)
    assert [token.text for token in segment("hi 你", config)] == ["hi"]


def test_punctuation_is_kept_when_asked_for() -> None:
    config = SegmentConfig(keep_punctuation=True)
    tokens = segment("hello, 你好.", config)
    assert [(token.text, token.script) for token in tokens] == [
        ("hello", "latin"),
        (",", "punctuation"),
        ("你", "han"),
        ("好", "han"),
        (".", "punctuation"),
    ]


def test_punctuation_is_dropped_by_default() -> None:
    assert [token.text for token in segment("hello, 你好.")] == ["hello", "你", "好"]


def test_hyphen_splitting_separates_a_compound() -> None:
    config = SegmentConfig(split_on_hyphen=True)
    assert [token.text for token in segment("code-switch", config)] == ["code", "switch"]


def test_hyphen_splitting_also_splits_an_apostrophe() -> None:
    config = SegmentConfig(split_on_hyphen=True)
    assert [token.text for token in segment("don't", config)] == ["don", "t"]


def test_hyphen_splitting_leaves_a_single_word_alone() -> None:
    config = SegmentConfig(split_on_hyphen=True)
    assert [token.text for token in segment("hello world", config)] == ["hello", "world"]


def test_both_flags_can_be_on_at_once() -> None:
    config = SegmentConfig(keep_punctuation=True, split_on_hyphen=True)
    assert [token.text for token in segment("code-switch, ok", config)] == [
        "code",
        "-",
        "switch",
        ",",
        "ok",
    ]
