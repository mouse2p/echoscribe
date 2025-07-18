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

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from echoscribe.errors import ConfigError, require

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
