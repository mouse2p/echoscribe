"""Normalisation is pinned on the exact strings that break a naive scorer."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.normalize import (
    NORMALIZATION_FORMS,
    PUNCTUATION_FOLD,
    NormalizeConfig,
    collapse_whitespace,
    fold_punctuation,
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


def test_the_ideographic_comma_becomes_an_ascii_comma() -> None:
    assert fold_punctuation("你、好") == "你,好"
    assert normalize_text("你、好") == "你,好"


def test_the_ideographic_full_stop_becomes_a_period() -> None:
    assert normalize_text("你好。") == "你好."


def test_a_corner_bracket_pair_becomes_quotes() -> None:
    assert normalize_text("「你好」") == '"你好"'


def test_an_ellipsis_becomes_three_periods() -> None:
    assert fold_punctuation("wait…") == "wait..."


def test_a_middle_dot_becomes_a_space_and_then_collapses() -> None:
    assert fold_punctuation("a\u00b7b") == "a b"
    assert normalize_text("a\u00b7\u00b7b") == "a b"


def test_folding_can_be_switched_off() -> None:
    config = NormalizeConfig(fold_punctuation=False)
    assert normalize_text("你好。", config) == "你好。"


def test_the_fold_table_has_no_identity_entries() -> None:
    for source, target in PUNCTUATION_FOLD.items():
        assert len(source) == 1
        assert target != source


def test_folding_leaves_ascii_punctuation_alone() -> None:
    assert fold_punctuation("a, b. c! d?") == "a, b. c! d?"


CORPUS_STRINGS = [
    "Hello, world!",
    "你好，世界！",
    "please open the FILE",
    "混合 code-switch 句子",
    "  spaced   out  ",
    "café naïve résumé",
    "ＨＥＬＬＯ　ＷＯＲＬＤ",
    "",
    "3.14159",
    "「quoted」…",
]


@pytest.mark.parametrize("text", CORPUS_STRINGS)
def test_normalising_twice_changes_nothing(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


@pytest.mark.parametrize("form", list(NORMALIZATION_FORMS))
def test_every_form_is_idempotent(form: str) -> None:
    config = NormalizeConfig(form=form)
    for text in CORPUS_STRINGS:
        once = normalize_text(text, config)
        assert normalize_text(once, config) == once


def test_normalisation_never_introduces_a_newline() -> None:
    for text in CORPUS_STRINGS:
        assert "\n" not in normalize_text(text)


def test_a_code_switched_sentence_keeps_both_scripts_separated() -> None:
    assert normalize_text("混合code-switch句子") == "混合code-switch句子"
    assert normalize_text("混合 code-switch 句子") == "混合 code-switch 句子"


def test_folding_runs_before_case_and_collapse() -> None:
    # The middle dot becomes a space, which the collapse stage then absorbs;
    # reversing the two would leave a doubled space behind.
    assert normalize_text("A\u00b7\u00b7B") == "a b"
