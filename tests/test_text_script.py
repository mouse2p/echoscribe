"""Script classification decides the metric, so every class is pinned."""

from __future__ import annotations

import pytest

from echoscribe.errors import TokenizerError
from echoscribe.text.script import SCRIPTS, ScriptRun, is_han, script_of, script_runs


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


def test_runs_cover_the_string_without_gaps_or_overlaps() -> None:
    text = "hello 你好 world 世界"
    cursor = 0
    for run in script_runs(text):
        assert run.start == cursor
        assert text[run.start : run.stop] == run.text
        cursor = run.stop
    assert cursor == len(text)


def test_a_code_switched_sentence_splits_at_every_switch() -> None:
    runs = script_runs("hello你好world")
    assert [(run.script, run.text) for run in runs] == [
        ("latin", "hello"),
        ("han", "你好"),
        ("latin", "world"),
    ]


def test_whitespace_merges_into_the_previous_run_by_default() -> None:
    runs = script_runs("hello 你")
    assert [(run.script, run.text) for run in runs] == [("latin", "hello "), ("han", "你")]


def test_whitespace_can_be_kept_as_its_own_run() -> None:
    runs = script_runs("hello 你", merge_space=False)
    assert [run.script for run in runs] == ["latin", "space", "han"]


def test_leading_whitespace_has_nothing_to_merge_into() -> None:
    runs = script_runs(" 你好")
    assert runs[0].script == "space"
    assert runs[1].text == "你好"


def test_digits_form_their_own_run() -> None:
    runs = script_runs("room 42 floor")
    assert [run.script for run in runs] == ["latin", "digit", "latin"]


def test_punctuation_attaches_to_neither_neighbour() -> None:
    runs = script_runs("你好,world")
    assert [run.script for run in runs] == ["han", "punctuation", "latin"]


def test_an_empty_string_has_no_runs() -> None:
    assert script_runs("") == ()


def test_a_run_reports_its_stop_offset() -> None:
    assert ScriptRun("latin", 3, "hello").stop == 8


def test_runs_are_immutable() -> None:
    run = ScriptRun("latin", 0, "a")
    with pytest.raises(AttributeError):
        run.text = "b"  # type: ignore[misc]


def test_a_non_string_is_rejected() -> None:
    with pytest.raises(TokenizerError, match="needs a string"):
        script_runs(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("character", "expected"),
    [
        ("\uff21", "latin"),  # fullwidth A
        ("\uff5a", "latin"),  # fullwidth z
        ("\uff10", "digit"),  # fullwidth 0
        ("\uff0c", "punctuation"),  # fullwidth comma
        ("\uff01", "punctuation"),  # fullwidth exclamation
    ],
)
def test_fullwidth_forms_follow_their_ascii_equivalent(character: str, expected: str) -> None:
    assert script_of(character) == expected


def test_every_fullwidth_ascii_variant_matches_its_base() -> None:
    for code in range(0xFF01, 0xFF5F):
        assert script_of(chr(code)) == script_of(chr(code - 0xFEE0)), hex(code)


def test_the_ideographic_space_still_reads_as_a_space() -> None:
    # U+3000 is outside the fullwidth ASCII range and is caught by isspace first.
    assert script_of("\u3000") == "space"
