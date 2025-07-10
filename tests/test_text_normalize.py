"""Normalisation is pinned on the exact strings that break a naive scorer."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.normalize import (
    NORMALIZATION_FORMS,
    NORMALIZATION_STAGES,
    PUNCTUATION_FOLD,
    NormalizeConfig,
    collapse_whitespace,
    fold_punctuation,
    normalize_text,
    strip_accents,
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


def test_strip_accents_folds_a_precomposed_latin_vowel() -> None:
    assert strip_accents("café") == "cafe"
    assert strip_accents("naïve résumé") == "naive resume"


def test_strip_accents_folds_a_decomposed_latin_vowel() -> None:
    assert strip_accents("cafe\u0301") == "cafe"


def test_strip_accents_leaves_a_non_latin_mark_alone() -> None:
    arabic = "\u0628\u064e"
    assert strip_accents(arabic) == arabic
    devanagari = "\u0915\u093c"
    assert strip_accents(devanagari) == devanagari


def test_strip_accents_can_be_unrestricted() -> None:
    assert strip_accents("\u0628\u064e", latin_only=False) == "\u0628"


def test_unrestricted_stripping_removes_stacked_marks() -> None:
    assert strip_accents("e\u0301\u0300", latin_only=False) == "e"
    assert strip_accents("e\u0301\u0300") == "e"


def test_strip_accents_leaves_han_characters_untouched() -> None:
    assert strip_accents("你好") == "你好"


def test_a_leading_combining_mark_survives_restricted_stripping() -> None:
    # With no base letter before it, the mark is all the information there is.
    assert strip_accents("\u0301a") == "\u0301a"


def test_the_configured_accent_stage_runs_before_collapse() -> None:
    config = NormalizeConfig(strip_accents=True)
    assert normalize_text("  Café  NAÏVE ", config) == "cafe naive"


def test_accent_stripping_is_off_by_default() -> None:
    assert normalize_text("café") == "café"


@pytest.mark.parametrize("text", CORPUS_STRINGS)
def test_normalising_twice_with_accents_stripped_changes_nothing(text: str) -> None:
    config = NormalizeConfig(strip_accents=True)
    once = normalize_text(text, config)
    assert normalize_text(once, config) == once


GOLDEN = {
    "ＨＥＬＬＯ　ＷＯＲＬＤ": "hello world",
    "你好，世界！": "你好,世界!",
    "請開啟 the FILE": "請開啟 the file",
    "a\u00b7b\u3000c": "a b c",
    "Café  Naïve": "cafe naive",
    "one\ttwo\n\nthree": "one two three",
    "「quoted」": '"quoted"',
    "wait…": "wait...",
}


@pytest.mark.parametrize(("source", "expected"), sorted(GOLDEN.items()))
def test_golden_normalisation(source: str, expected: str) -> None:
    assert normalize_text(source, NormalizeConfig(strip_accents=True)) == expected


def test_the_golden_table_is_stable_under_a_second_pass() -> None:
    config = NormalizeConfig(strip_accents=True)
    for expected in GOLDEN.values():
        assert normalize_text(expected, config) == expected


def test_golden_outputs_never_contain_a_fullwidth_form() -> None:
    config = NormalizeConfig(strip_accents=True)
    for source in GOLDEN:
        result = normalize_text(source, config)
        assert not any(0xFF01 <= ord(character) <= 0xFF5E for character in result), source


def test_golden_outputs_never_contain_a_folded_punctuation_character() -> None:
    config = NormalizeConfig(strip_accents=True)
    for source in GOLDEN:
        result = normalize_text(source, config)
        assert not any(character in PUNCTUATION_FOLD for character in result), source


def test_the_stage_order_is_the_documented_one() -> None:
    assert [name for name, _ in NORMALIZATION_STAGES] == [
        "fold_punctuation",
        "lower",
        "strip_accents",
        "collapse_spaces",
    ]


def test_every_stage_has_a_matching_config_field() -> None:
    for name, _ in NORMALIZATION_STAGES:
        assert hasattr(NormalizeConfig(), name), name


def test_the_stage_functions_are_the_public_ones() -> None:
    stages = dict(NORMALIZATION_STAGES)
    assert stages["fold_punctuation"] is fold_punctuation
    assert stages["collapse_spaces"] is collapse_whitespace
    assert stages["strip_accents"] is strip_accents


def test_the_pipeline_still_matches_the_explicit_stage_order() -> None:
    config = NormalizeConfig(strip_accents=True)
    # No compatibility characters here, so NFKC is the identity and the manual
    # chain below is exactly what normalize_text has to produce.
    text = "  Café 「A」…  "
    manual = collapse_whitespace(strip_accents(fold_punctuation(text).lower()))
    assert normalize_text(text, config) == manual


def test_the_text_subpackage_re_exports_its_public_names() -> None:
    from echoscribe import text
    from echoscribe.text.script import script_runs

    assert text.normalize_text is normalize_text
    assert text.script_runs is script_runs
    assert text.NormalizeConfig is NormalizeConfig
    assert set(text.__all__) >= {"normalize_text", "script_of", "script_runs", "SCRIPTS"}
