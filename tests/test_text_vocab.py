"""A vocabulary is an id contract, so its slot order is pinned exactly."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError, SchemaError, TokenizerError
from echoscribe.text.vocab import (
    DEFAULT_RESERVED,
    VOCAB_SCHEMA_VERSION,
    Vocabulary,
)


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


def test_serialization_round_trips() -> None:
    vocab = Vocabulary.build(["hello", "world", "你"])
    assert Vocabulary.from_dict(vocab.to_dict()) == vocab


def test_the_payload_names_its_schema_and_digest() -> None:
    payload = Vocabulary.build(["a"]).to_dict()
    assert payload["schema"] == VOCAB_SCHEMA_VERSION
    assert set(payload) == {"schema", "reserved", "tokens", "digest"}
    assert len(payload["digest"]) == 16


def test_the_digest_covers_the_order() -> None:
    forward = Vocabulary.build(["a", "b"])
    backward = Vocabulary.build(["b", "a"])
    assert forward.digest() != backward.digest()


def test_the_digest_is_stable_across_calls() -> None:
    vocab = Vocabulary.build(["a", "b", "c"])
    assert vocab.digest() == Vocabulary.build(["a", "b", "c"]).digest()


def test_the_digest_is_hexadecimal() -> None:
    digest = Vocabulary.build(["a"]).digest()
    assert all(character in "0123456789abcdef" for character in digest)


def test_a_foreign_schema_is_refused() -> None:
    payload = Vocabulary.build(["a"]).to_dict()
    payload["schema"] = VOCAB_SCHEMA_VERSION + 1
    with pytest.raises(SchemaError, match="is not supported"):
        Vocabulary.from_dict(payload)


def test_a_missing_field_is_refused() -> None:
    payload = Vocabulary.build(["a"]).to_dict()
    del payload["tokens"]
    with pytest.raises(SchemaError, match="missing 'tokens'"):
        Vocabulary.from_dict(payload)


def test_a_tampered_token_list_fails_the_digest_check() -> None:
    payload = Vocabulary.build(["a", "b"]).to_dict()
    payload["tokens"][4] = "c"
    with pytest.raises(SchemaError, match="digest mismatch"):
        Vocabulary.from_dict(payload)


def test_a_payload_without_a_digest_still_loads() -> None:
    payload = Vocabulary.build(["a"]).to_dict()
    del payload["digest"]
    assert Vocabulary.from_dict(payload) == Vocabulary.build(["a"])


def test_the_serialized_form_is_json_encodable() -> None:
    import json

    text = json.dumps(Vocabulary.build(["你好", "a"]).to_dict(), ensure_ascii=False)
    assert "你好" in text


def test_a_reserved_token_in_the_pieces_is_refused_even_when_repeated() -> None:
    with pytest.raises(ConfigError, match="is a reserved token"):
        Vocabulary.build(["<sos>", "a"])
    with pytest.raises(ConfigError, match="is a reserved token"):
        Vocabulary.build(["a", "b", "<eos>"])


def test_a_custom_reserved_token_is_also_protected() -> None:
    with pytest.raises(ConfigError, match="is a reserved token"):
        Vocabulary.build(["<pad>"], reserved=("<pad>", "<unk>"))


def test_the_refusal_names_the_offending_piece() -> None:
    with pytest.raises(ConfigError, match="'<blank>' is a reserved token"):
        Vocabulary.build(["<blank>"])
