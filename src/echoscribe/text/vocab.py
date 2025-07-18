"""Vocabulary management: the token<->id table a decoder runs against.

Four tokens are *reserved*, which means they always occupy the first four ids in a
fixed order:

``<blank>``
    the CTC blank and the transducer's "emit nothing this frame" symbol.  It has to
    be id 0 because both decoders treat index 0 as the no-output case.
``<unk>``
    what an out-of-vocabulary piece becomes.  Having a slot for it means encoding
    never has to fail, which matters when a hypothesis contains a word the training
    corpus did not.
``<sos>`` / ``<eos>``
    the predictor's start and stop symbols in a transducer.  ``<eos>`` is what lets
    the decoder end an utterance on its own rather than by running out of frames.

The rest of the table is whatever the tokenizer produced, in the order it produced
it.  Order is part of the vocabulary's identity: a re-ordered table with the same
contents is a different model checkpoint's vocabulary, so the serialized form
records the order and the digest covers it.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from echoscribe.errors import ConfigError, SchemaError, TokenizerError, require

#: Bumped when the serialized shape of a vocabulary changes.
VOCAB_SCHEMA_VERSION = 1

#: The reserved tokens, in the id order they always occupy.
DEFAULT_RESERVED = ("<blank>", "<unk>", "<sos>", "<eos>")


@dataclass(frozen=True)
class Vocabulary:
    """An immutable token list with a lookup index built once."""

    tokens: tuple[str, ...]
    reserved: tuple[str, ...] = DEFAULT_RESERVED
    _index: dict[str, int] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        require(
            len(self.tokens) >= len(self.reserved),
            f"a vocabulary must hold at least its {len(self.reserved)} reserved tokens, "
            f"got {len(self.tokens)}",
            ConfigError,
        )
        require(
            self.tokens[: len(self.reserved)] == tuple(self.reserved),
            f"the first {len(self.reserved)} slots must be {self.reserved}, "
            f"got {self.tokens[: len(self.reserved)]}",
            ConfigError,
        )
        duplicates = sorted(token for token, count in Counter(self.tokens).items() if count > 1)
        require(not duplicates, f"vocabulary holds duplicate tokens: {duplicates[:8]}", ConfigError)
        object.__setattr__(self, "_index", {token: i for i, token in enumerate(self.tokens)})

    @classmethod
    def build(
        cls, pieces: Iterable[str], reserved: tuple[str, ...] = DEFAULT_RESERVED
    ) -> Vocabulary:
        """Build a vocabulary from tokenizer output, skipping repeats.

        A piece that is already reserved is an error rather than a silent skip: it
        would otherwise occupy two ids and a decoder trained on one would decode the
        other.
        """
        reserved_set = set(reserved)
        seen = set(reserved_set)
        ordered = list(reserved)
        for piece in pieces:
            require(
                piece not in reserved_set,
                f"{piece!r} is a reserved token and cannot also be a vocabulary piece",
                ConfigError,
            )
            require(bool(piece), "a vocabulary piece must not be empty", ConfigError)
            if piece not in seen:
                seen.add(piece)
                ordered.append(piece)
        return cls(tuple(ordered), tuple(reserved))

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._index

    @property
    def blank_id(self) -> int:
        """Id of the CTC blank / transducer no-output symbol."""
        return self._reserved_id(0)

    @property
    def unk_id(self) -> int:
        """Id an out-of-vocabulary piece is mapped to, when that is allowed."""
        return self._reserved_id(1)

    @property
    def sos_id(self) -> int:
        """Id of the predictor's start symbol."""
        return self._reserved_id(2)

    @property
    def eos_id(self) -> int:
        """Id of the predictor's stop symbol."""
        return self._reserved_id(3)

    def _reserved_id(self, position: int) -> int:
        """Id of the reserved token at ``position``, or an error if there is none."""
        require(
            position < len(self.reserved),
            f"this vocabulary reserves only {len(self.reserved)} slots, not {position + 1}",
            ConfigError,
        )
        return position

    def id_of(self, token: str) -> int:
        """Id of ``token``; an unknown token is an error, not a silent ``<unk>``."""
        try:
            return self._index[token]
        except KeyError:
            raise TokenizerError(f"{token!r} is not in the vocabulary") from None

    def token_at(self, index: int) -> str:
        """Token at ``index``; an out-of-range id is an error."""
        require(
            isinstance(index, int) and not isinstance(index, bool),
            f"a vocabulary id must be an int, got {index!r}",
            TokenizerError,
        )
        if not 0 <= index < len(self.tokens):
            raise TokenizerError(f"id {index} is outside [0, {len(self.tokens)})")
        return self.tokens[index]

    def encode(self, tokens: Sequence[str], allow_unk: bool = True) -> tuple[int, ...]:
        """Map tokens to ids, substituting ``<unk>`` or raising as configured."""
        ids = []
        for token in tokens:
            if token in self._index:
                ids.append(self._index[token])
                continue
            require(
                allow_unk,
                f"{token!r} is not in the vocabulary and allow_unk is off",
                TokenizerError,
            )
            ids.append(self.unk_id)
        return tuple(ids)

    def decode(self, ids: Sequence[int]) -> tuple[str, ...]:
        """Map ids back to tokens."""
        return tuple(self.token_at(index) for index in ids)

    def digest(self) -> str:
        """A short hex digest of the whole ordered token list.

        Two vocabularies with the same digest are interchangeable for a trained
        model; anything else -- one extra piece, one reordering -- is not.  Reports
        quote this so a result can be tied to the exact table that produced it.
        """
        return hashlib.sha256("\n".join(self.tokens).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """A JSON-ready mapping, tagged with the schema version."""
        return {
            "schema": VOCAB_SCHEMA_VERSION,
            "reserved": list(self.reserved),
            "tokens": list(self.tokens),
            "digest": self.digest(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Vocabulary:
        """Rebuild a vocabulary, refusing a schema or digest that does not match."""
        version = payload.get("schema")
        require(
            version == VOCAB_SCHEMA_VERSION,
            f"vocabulary schema {version!r} is not supported; this build reads "
            f"{VOCAB_SCHEMA_VERSION}",
            SchemaError,
        )
        for name in ("reserved", "tokens"):
            require(name in payload, f"vocabulary payload is missing {name!r}", SchemaError)
        vocab = cls(tuple(payload["tokens"]), tuple(payload["reserved"]))
        recorded = payload.get("digest")
        require(
            recorded is None or recorded == vocab.digest(),
            f"vocabulary digest mismatch: the file says {recorded!r}, the tokens hash to "
            f"{vocab.digest()!r}",
            SchemaError,
        )
        return vocab
