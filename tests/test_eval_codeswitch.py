"""Mixed tokenization is the contract every code-switch score rests on."""

from __future__ import annotations

from echoscribe.eval.codeswitch import mixed_token_sequence


def test_han_characters_become_individual_tokens() -> None:
    assert mixed_token_sequence("今天") == (("今", "han"), ("天", "han"))


def test_latin_words_stay_together() -> None:
    assert mixed_token_sequence("use AI") == (("use", "latin"), ("AI", "latin"))


def test_a_mixed_sentence_keeps_every_script_run_in_order() -> None:
    assert mixed_token_sequence("今天use AI好吧") == (
        ("今", "han"),
        ("天", "han"),
        ("use", "latin"),
        ("AI", "latin"),
        ("好", "han"),
        ("吧", "han"),
    )


def test_punctuation_and_spaces_are_dropped() -> None:
    assert mixed_token_sequence("你好，世界！") == (
        ("你", "han"),
        ("好", "han"),
        ("世", "han"),
        ("界", "han"),
    )
    assert mixed_token_sequence("a, b.") == (("a", "latin"), ("b", "latin"))


def test_a_digit_run_is_one_token() -> None:
    assert mixed_token_sequence("room 42 ok") == (
        ("room", "latin"),
        ("42", "digit"),
        ("ok", "latin"),
    )


def test_digits_inside_han_stay_their_own_run() -> None:
    assert mixed_token_sequence("第42层") == (
        ("第", "han"),
        ("42", "digit"),
        ("层", "han"),
    )


def test_an_empty_string_has_no_tokens() -> None:
    assert mixed_token_sequence("") == ()
    assert mixed_token_sequence("   ") == ()
    assert mixed_token_sequence("，。！") == ()
