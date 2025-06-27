"""Normalisation is pinned on the exact strings that break a naive scorer."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.normalize import (
    NORMALIZATION_FORMS,
    NormalizeConfig,
    collapse_whitespace,
    normalize_text,
)


def test_the_default_config_folds_compatibility_characters() -> None:
    assert normalize_text("ＨＥＬＬＯ") == "hello"


def test_the_default_config_lowercases() -> None:
    assert normalize_text("Hello World") == "hello world"


def test_case_folding_can_be_switched_off() -> None:
    assert normalize_text("Hello World", NormalizeConfig(lower=False)) == "Hello World"


def test_whitespace_runs_collapse_to_one_space() -> None:
    assert normalize_text("a \t\n  b") == "a b"


def test_collapse_can_be_switched_off() -> None:
    config = NormalizeConfig(collapse_spaces=False)
    assert normalize_text("a  b", config) == "a  b"


def test_leading_and_trailing_space_is_removed() -> None:
    assert normalize_text("  hello  ") == "hello"


def test_collapse_whitespace_joins_on_any_separator() -> None:
    assert collapse_whitespace("a \t\n b   c") == "a b c"
    assert collapse_whitespace("   ") == ""


def test_an_ideographic_space_becomes_an_ascii_one() -> None:
    assert normalize_text("你\u3000好") == "你 好"


def test_every_declared_form_is_accepted() -> None:
    for form in NORMALIZATION_FORMS:
        assert normalize_text("café", NormalizeConfig(form=form))


def test_an_unknown_form_is_rejected() -> None:
    with pytest.raises(ConfigError, match="form must be one of"):
        NormalizeConfig(form="NFKC7")


def test_a_non_string_is_rejected() -> None:
    with pytest.raises(ConfigError, match="needs a string"):
        normalize_text(17)  # type: ignore[arg-type]


def test_an_empty_string_normalises_to_itself() -> None:
    assert normalize_text("") == ""


def test_config_is_frozen() -> None:
    config = NormalizeConfig()
    with pytest.raises(AttributeError):
        config.lower = False  # type: ignore[misc]
