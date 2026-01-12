"""Report output is a byte contract, so the tests pin bytes."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.eval.report import escape_text, format_length


def test_every_reserved_character_becomes_an_entity() -> None:
    assert escape_text("&") == "&amp;"
    assert escape_text("<") == "&lt;"
    assert escape_text(">") == "&gt;"
    assert escape_text('"') == "&quot;"
    assert escape_text("'") == "&apos;"


def test_the_ampersand_is_escaped_first() -> None:
    assert escape_text("<&>") == "&lt;&amp;&gt;"


def test_escaping_is_not_idempotent_by_design() -> None:
    assert escape_text("&amp;") == "&amp;amp;"


def test_plain_and_cjk_text_passes_through_unchanged() -> None:
    assert escape_text("hello world") == "hello world"
    assert escape_text("今天天气不错") == "今天天气不错"


def test_the_empty_string_escapes_to_the_empty_string() -> None:
    assert escape_text("") == ""


def test_a_non_string_is_refused() -> None:
    with pytest.raises(ConfigError, match="needs a string"):
        escape_text(42)


def test_whole_numbers_render_without_a_decimal_point() -> None:
    assert format_length(12.0) == "12"
    assert format_length(0.0) == "0"
    assert format_length(-3.0) == "-3"
    assert format_length(7) == "7"


def test_two_decimals_survive_and_trailing_zeros_do_not() -> None:
    assert format_length(12.5) == "12.5"
    assert format_length(12.34) == "12.34"
    assert format_length(0.1) == "0.1"


def test_values_are_rounded_to_two_decimals() -> None:
    assert format_length(12.345) == "12.35"
    assert format_length(12.344) == "12.34"


def test_float_noise_below_the_second_decimal_disappears() -> None:
    assert format_length(0.1 + 0.2) == "0.3"


def test_a_tiny_negative_rounds_to_plain_zero() -> None:
    assert format_length(-0.001) == "0"


def test_non_finite_lengths_are_refused() -> None:
    with pytest.raises(ConfigError, match="must be finite"):
        format_length(float("nan"))
    with pytest.raises(ConfigError, match="must be finite"):
        format_length(float("inf"))


def test_a_string_or_boolean_length_is_refused() -> None:
    with pytest.raises(ConfigError, match="must be a number"):
        format_length("12")
    with pytest.raises(ConfigError, match="must be a number"):
        format_length(True)
