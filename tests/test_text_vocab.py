"""A vocabulary is an id contract, so its slot order is pinned exactly."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError, TokenizerError
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


def test_reserved_ids_are_the_first_four() -> None:
    vocab = Vocabulary.build(["a", "b"])
    assert (vocab.blank_id, vocab.unk_id, vocab.sos_id, vocab.eos_id) == (0, 1, 2, 3)


def test_a_short_reserved_block_has_no_eos_slot() -> None:
    vocab = Vocabulary.build(["x"], reserved=("<blank>", "<unk>"))
    with pytest.raises(ConfigError, match="reserves only 2 slots"):
        _ = vocab.eos_id


def test_id_of_and_token_at_are_inverses() -> None:
    vocab = Vocabulary.build(["hello", "world", "你", "好"])
    for token in vocab.tokens:
        assert vocab.token_at(vocab.id_of(token)) == token


def test_every_id_round_trips() -> None:
    vocab = Vocabulary.build(["a", "b", "c", "d"])
    assert list(vocab.decode(range(len(vocab)))) == list(vocab.tokens)


def test_id_of_raises_on_an_unknown_token() -> None:
    vocab = Vocabulary.build(["a"])
    with pytest.raises(TokenizerError, match="'z' is not in the vocabulary"):
        vocab.id_of("z")


def test_encode_substitutes_unk_by_default() -> None:
    vocab = Vocabulary.build(["a", "b"])
    assert vocab.encode(["a", "z", "b"]) == (4, 1, 5)


def test_encode_can_be_made_strict() -> None:
    vocab = Vocabulary.build(["a"])
    with pytest.raises(TokenizerError, match="allow_unk is off"):
        vocab.encode(["a", "z"], allow_unk=False)


def test_encode_of_nothing_is_empty() -> None:
    assert Vocabulary.build(["a"]).encode([]) == ()


def test_token_at_rejects_an_out_of_range_id() -> None:
    vocab = Vocabulary.build(["a"])
    with pytest.raises(TokenizerError, match="is outside"):
        vocab.token_at(99)
    with pytest.raises(TokenizerError, match="is outside"):
        vocab.token_at(-1)


def test_token_at_rejects_a_non_integer_id() -> None:
    with pytest.raises(TokenizerError, match="must be an int"):
        Vocabulary.build(["a"]).token_at(1.0)  # type: ignore[arg-type]


def test_encode_decode_round_trips_a_known_sequence() -> None:
    vocab = Vocabulary.build(["hello", "world", "你", "好"])
    tokens = ("hello", "你", "好", "world")
    assert vocab.decode(vocab.encode(tokens)) == tokens
