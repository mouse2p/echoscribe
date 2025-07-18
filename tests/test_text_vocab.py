"""A vocabulary is an id contract, so its slot order is pinned exactly."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.text.vocab import DEFAULT_RESERVED, Vocabulary


def test_build_puts_the_reserved_tokens_first_in_order() -> None:
    vocab = Vocabulary.build(["a", "b", "c"])
    assert vocab.tokens[:4] == DEFAULT_RESERVED
    assert vocab.tokens[4:] == ("a", "b", "c")
    assert len(vocab) == 7


def test_the_reserved_names_are_the_four_a_decoder_needs() -> None:
    assert DEFAULT_RESERVED == ("<blank>", "<unk>", "<sos>", "<eos>")


def test_build_deduplicates_pieces_but_keeps_the_first_order() -> None:
    vocab = Vocabulary.build(["b", "a", "b", "c", "a"])
    assert vocab.tokens[4:] == ("b", "a", "c")


def test_a_piece_that_is_already_reserved_is_refused() -> None:
    with pytest.raises(ConfigError, match="is a reserved token"):
        Vocabulary.build(["a", "<unk>"])


def test_an_empty_piece_is_refused() -> None:
    with pytest.raises(ConfigError, match="must not be empty"):
        Vocabulary.build(["a", ""])


def test_a_custom_reserved_block_is_honoured() -> None:
    vocab = Vocabulary.build(["x"], reserved=("<blank>", "<unk>"))
    assert vocab.tokens == ("<blank>", "<unk>", "x")
    assert vocab.reserved == ("<blank>", "<unk>")


def test_constructing_directly_rejects_a_duplicate() -> None:
    with pytest.raises(ConfigError, match="duplicate tokens"):
        Vocabulary(("<blank>", "<unk>", "<sos>", "<eos>", "a", "a"))


def test_constructing_directly_rejects_missing_reserved_slots() -> None:
    with pytest.raises(ConfigError, match="must hold at least its 4 reserved tokens"):
        Vocabulary(("<blank>", "<unk>"))
    with pytest.raises(ConfigError, match="the first 4 slots must be"):
        Vocabulary(("<unk>", "<blank>", "<sos>", "<eos>"))


def test_membership_uses_the_index() -> None:
    vocab = Vocabulary.build(["hello", "你"])
    assert "hello" in vocab
    assert "你" in vocab
    assert "world" not in vocab


def test_a_vocabulary_is_immutable() -> None:
    vocab = Vocabulary.build(["a"])
    with pytest.raises(AttributeError):
        vocab.tokens = ()  # type: ignore[misc]
