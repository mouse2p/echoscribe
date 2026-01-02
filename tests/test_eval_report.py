"""Report output is a byte contract, so the tests pin bytes."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.eval.report import escape_text


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
