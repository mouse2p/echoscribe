"""Byte-pair encoding, trained from scratch on the synthetic corpus.

BPE starts from single characters and repeatedly merges the most frequent adjacent
pair, recording each merge.  Encoding a word then means applying the learned merges
in rank order.  It is the simplest subword scheme that has the property an ASR
output layer actually needs: a finite alphabet that can still spell any word,
including one the training corpus never contained.

Two decisions here are worth calling out because they are where implementations
usually differ:

*words are spelled with an explicit end marker.*  A word is turned into its
characters plus ``"</w>"``, so a merge can only ever join a character to what
follows it inside the same word.  Without the marker, "the" and "then" would share
a prefix merge and the model could not tell a finished word from an unfinished one.

*the trainer takes no seed.*  Frequency ties are broken by the lexicographic order
of the pair, which makes the merge list a pure function of the word counts.  A
seeded tie-break would produce a different model per run for no benefit and make
"same corpus, same tokenizer" untestable.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from echoscribe.errors import ConfigError, SchemaError, require, require_positive
from echoscribe.text.normalize import NormalizeConfig, normalize_text
from echoscribe.text.segment import SegmentConfig, segment, token_texts

#: Marks the end of a word inside a symbol sequence.
WORD_END = "</w>"


def word_symbols(word: str) -> tuple[str, ...]:
    """Spell ``word`` as its characters followed by :data:`WORD_END`."""
    require(
        isinstance(word, str), f"a word must be a string, got {type(word).__name__}", ConfigError
    )
    require(bool(word), "cannot spell an empty word", ConfigError)
    return tuple(word) + (WORD_END,)


def count_pairs(words: Iterable[str]) -> Counter:
    """Count adjacent symbol pairs over a (possibly repeated) word list.

    Repeats matter: a word that occurs ten times contributes ten of each of its
    pairs, which is exactly the frequency signal the merge search needs.
    """
    counts: Counter = Counter()
    for word in words:
        symbols = word_symbols(word)
        counts.update(zip(symbols, symbols[1:], strict=False))
    return counts


def merge_pair(symbols: Sequence[str], pair: tuple[str, str]) -> tuple[str, ...]:
    """Apply one merge left to right, without overlapping matches."""
    result: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            result.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            result.append(symbols[index])
            index += 1
    return tuple(result)


def learn_bpe(words: Iterable[str], num_merges: int) -> tuple[tuple[str, str], ...]:
    """Learn at most ``num_merges`` merges from ``words``.

    Each round counts adjacent pairs weighted by word frequency, takes the most
    frequent one, and breaks a tie by choosing the lexicographically smaller pair.
    The tie-break is what makes the result reproducible: without it two runs over
    the same corpus can produce different merge orders whenever two pairs are
    equally common, which is often.

    Fewer merges are returned than requested when the corpus runs out of pairs to
    merge; that is a normal end state, not an error.
    """
    require_positive("num_merges", num_merges, ConfigError)
    frequencies = Counter(words)
    splits = {word: word_symbols(word) for word in frequencies}
    merges: list[tuple[str, str]] = []
    for _ in range(int(num_merges)):
        pairs: Counter = Counter()
        for word, frequency in frequencies.items():
            symbols = splits[word]
            for pair in zip(symbols, symbols[1:], strict=False):
                pairs[pair] += frequency
        if not pairs:
            break
        best = min(pairs.items(), key=lambda item: (-item[1], item[0]))[0]
        merges.append(best)
        splits = {word: merge_pair(symbols, best) for word, symbols in splits.items()}
    return tuple(merges)


#: Bumped when the serialized shape of a BPE model changes.
BPE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BpeModel:
    """A learned merge list, plus the encode and decode it defines."""

    merges: tuple[tuple[str, str], ...] = ()
    _ranks: dict[tuple[str, str], int] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        for pair in self.merges:
            require(
                isinstance(pair, tuple) and len(pair) == 2 and all(pair),
                f"a merge must be a pair of non-empty strings, got {pair!r}",
                ConfigError,
            )
        duplicates = [pair for pair, count in Counter(self.merges).items() if count > 1]
        require(not duplicates, f"the merge list repeats {duplicates[:4]}", ConfigError)
        object.__setattr__(self, "_ranks", {pair: rank for rank, pair in enumerate(self.merges)})

    @classmethod
    def learn(cls, words: Iterable[str], num_merges: int) -> BpeModel:
        """Train a model in one call."""
        return cls(learn_bpe(words, num_merges))

    def __len__(self) -> int:
        return len(self.merges)

    def rank_of(self, pair: tuple[str, str]) -> int | None:
        """Rank of a merge, or ``None`` when the pair was never merged."""
        return self._ranks.get(pair)

    def encode(self, word: str) -> tuple[str, ...]:
        """Split ``word`` into pieces by applying the lowest-rank merge available.

        Applying by rank rather than in list order matters: a later merge can create
        a pair that an earlier merge would have joined, and the rank order is what
        makes encoding agree with what training intended.
        """
        symbols = list(word_symbols(word))
        while len(symbols) > 1:
            best_rank = None
            best_at = -1
            for index in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_at = index
            if best_rank is None:
                break
            symbols[best_at : best_at + 2] = [symbols[best_at] + symbols[best_at + 1]]
        return tuple(symbols)

    @staticmethod
    def decode(pieces: Sequence[str]) -> str:
        """Rebuild one word from its pieces, dropping the end marker."""
        return "".join(pieces).replace(WORD_END, "")

    def digest(self) -> str:
        """A short hex digest of the merge list, in order.

        Two models with the same digest encode identically.  A report quotes this so
        a score can be tied to the tokenizer that produced it, and so re-training on
        a slightly different corpus is caught rather than silently substituted.
        """
        joined = "\n".join(f"{left}\x00{right}" for left, right in self.merges)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """A JSON-ready mapping of the model."""
        return {
            "schema": BPE_SCHEMA_VERSION,
            "merges": [[left, right] for left, right in self.merges],
            "digest": self.digest(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BpeModel:
        """Rebuild a model, refusing a schema or digest that does not match."""
        version = payload.get("schema")
        require(
            version == BPE_SCHEMA_VERSION,
            f"BPE schema {version!r} is not supported; this build reads {BPE_SCHEMA_VERSION}",
            SchemaError,
        )
        require("merges" in payload, "BPE payload is missing 'merges'", SchemaError)
        model = cls(tuple((left, right) for left, right in payload["merges"]))
        recorded = payload.get("digest")
        require(
            recorded is None or recorded == model.digest(),
            f"BPE digest mismatch: the file says {recorded!r}, the merges hash to "
            f"{model.digest()!r}",
            SchemaError,
        )
        return model


def tokenize(
    text: str,
    model: BpeModel,
    normalize: NormalizeConfig | None = None,
    segment_config: SegmentConfig | None = None,
) -> tuple[str, ...]:
    """Normalise, segment and subword-encode a whole utterance.

    Each segment is encoded on its own, so a merge can never cross a word boundary.
    That is also why a Han character stays a single piece: segmentation emits it as
    its own token, and BPE only ever learns merges *inside* a token.
    """
    normalized = normalize_text(text, normalize)
    pieces: list[str] = []
    for token in token_texts(segment(normalized, segment_config)):
        pieces.extend(model.encode(token))
    return tuple(pieces)


def detokenize(pieces: Sequence[str]) -> tuple[str, ...]:
    """Recover the word list from :func:`tokenize` output.

    The result is a tuple of words rather than a string, because how those words
    should be joined depends on their script -- see
    :func:`echoscribe.text.segment.join_for_display`.  Claiming a single string here
    would bake in an English punctuation convention.
    """
    words: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if piece.endswith(WORD_END):
            current.append(piece[: -len(WORD_END)])
            words.append("".join(current))
            current = []
        else:
            current.append(piece)
    require(not current, f"piece sequence ends without {WORD_END!r}: {tuple(current)}", ConfigError)
    return tuple(words)
