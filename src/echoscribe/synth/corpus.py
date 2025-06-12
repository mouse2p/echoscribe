"""Seeded text corpus generation for the synthetic laboratory.

A corpus here is a list of utterances, and an utterance is a token list plus the
transcript that token list implies.  Nothing is recorded and nothing is
transcribed by hand: the text is generated first and the audio is rendered from it
afterwards, which is why the transcript can be treated as exact ground truth.

Generation is driven entirely by :class:`~echoscribe.seeds.Seed` derivation, so a
corpus is identified by one integer.  Two runs with the same seed produce the same
token lists in the same order, and per-utterance seeds are forked so that changing
the utterance count does not reshuffle the ones before it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require, require_range
from echoscribe.seeds import DEFAULT_ROOT, Seed, new_rng
from echoscribe.synth.lexicon import LANGUAGES, Lexicon


@dataclass(frozen=True)
class CorpusConfig:
    """How many utterances to generate and what they are made of."""

    utterances: int = 24
    min_tokens: int = 2
    max_tokens: int = 7
    english_probability: float = 0.5
    seed: Seed = DEFAULT_ROOT

    def __post_init__(self) -> None:
        for name, value in (
            ("utterances", self.utterances),
            ("min_tokens", self.min_tokens),
            ("max_tokens", self.max_tokens),
        ):
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{name} must be a non-negative integer, got {value!r}",
                ConfigError,
            )
        require(self.utterances > 0, "a corpus needs at least one utterance", ConfigError)
        require(self.min_tokens > 0, "an utterance needs at least one token", ConfigError)
        require(
            self.max_tokens >= self.min_tokens,
            f"max_tokens ({self.max_tokens}) must be at least min_tokens ({self.min_tokens})",
            ConfigError,
        )
        require_range("english_probability", self.english_probability, 0.0, 1.0, ConfigError)
        require(isinstance(self.seed, Seed), f"seed must be a Seed, got {self.seed!r}", ConfigError)


def surface_text(tokens: Sequence[str], languages: Sequence[str]) -> str:
    """Render a token list the way it would be written.

    English words are separated by spaces and Chinese characters are not, with a
    space inserted at each script switch.  That is the convention the evaluation
    suite scores against, so it has to be produced here rather than re-derived by
    each consumer.
    """
    require(
        len(tokens) == len(languages),
        f"{len(tokens)} tokens cannot carry {len(languages)} language labels",
        ConfigError,
    )
    parts: list[str] = []
    previous: str | None = None
    for token, language in zip(tokens, languages, strict=True):
        needs_space = previous is not None and not (previous == "zh" and language == "zh")
        if needs_space:
            parts.append(" ")
        parts.append(token)
        previous = language
    return "".join(parts)


@dataclass(frozen=True)
class Utterance:
    """One generated utterance: its tokens, their scripts and its transcript."""

    utterance_id: str
    tokens: tuple[str, ...]
    languages: tuple[str, ...]
    text: str
    seed: Seed

    def __post_init__(self) -> None:
        require(bool(self.utterance_id), "an utterance needs an id", ConfigError)
        require(
            len(self.tokens) > 0, f"{self.utterance_id}: an utterance needs tokens", ConfigError
        )
        require(
            len(self.tokens) == len(self.languages),
            f"{self.utterance_id}: {len(self.tokens)} tokens but {len(self.languages)} labels",
            ConfigError,
        )
        for language in self.languages:
            require(
                language in LANGUAGES,
                f"{self.utterance_id}: unknown language label {language!r}",
                ConfigError,
            )

    @property
    def is_code_switched(self) -> bool:
        """True when more than one script appears in the utterance."""
        return len(set(self.languages)) > 1

    @property
    def num_tokens(self) -> int:
        """Number of surface tokens, which is what WER counts."""
        return len(self.tokens)

    def token_count(self, language: str) -> int:
        """How many of this utterance's tokens are in ``language``."""
        require(language in LANGUAGES, f"unknown language {language!r}", ConfigError)
        return sum(1 for label in self.languages if label == language)


def sample_tokens(
    rng: np.random.Generator,
    config: CorpusConfig,
    lexicon: Lexicon | None = None,
) -> tuple[str, ...]:
    """Draw one utterance's token list.

    Each token's language is drawn from a Bernoulli with ``english_probability``,
    then a token is picked uniformly from that language's vocabulary.  A language
    whose vocabulary is empty is dropped from the draw rather than raising, so a
    caller can generate an English-only corpus from a lexicon that has no
    characters in it.
    """
    table = Lexicon() if lexicon is None else lexicon
    pools = {"en": list(table.english), "zh": list(table.mandarin)}
    available = [language for language in LANGUAGES if pools[language]]
    require(bool(available), "the lexicon holds no tokens to sample from", ConfigError)
    weights = [
        config.english_probability if language == "en" else 1.0 - config.english_probability
        for language in available
    ]
    total = sum(weights)
    require(total > 0.0, "the language probabilities leave nothing to sample", ConfigError)
    probabilities = [weight / total for weight in weights]
    count = int(rng.integers(config.min_tokens, config.max_tokens + 1))
    choices = rng.choice(len(available), size=count, p=probabilities)
    return tuple(
        pools[available[int(index)]][int(rng.integers(len(pools[available[int(index)]])))]
        for index in choices
    )


@dataclass(frozen=True)
class Corpus:
    """A generated corpus and the configuration that produced it."""

    utterances: tuple[Utterance, ...]
    config: CorpusConfig

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self.utterances)

    def transcript(self) -> str:
        """Every transcript, one per line, in corpus order."""
        return "\n".join(utterance.text for utterance in self.utterances)

    def language_mix(self) -> dict[str, int]:
        """Token counts per language across the whole corpus."""
        counts = dict.fromkeys(LANGUAGES, 0)
        for utterance in self.utterances:
            for language in utterance.languages:
                counts[language] += 1
        return counts

    def code_switched_share(self) -> float:
        """Fraction of utterances that switch script at least once."""
        if not self.utterances:
            return 0.0
        switched = sum(1 for utterance in self.utterances if utterance.is_code_switched)
        return switched / len(self.utterances)


def generate_corpus(config: CorpusConfig, lexicon: Lexicon | None = None) -> Corpus:
    """Generate ``config.utterances`` utterances from ``config.seed``.

    Each utterance forks its own seed from the corpus seed, so appending utterances
    to a corpus leaves the ones already generated untouched -- which matters when a
    feature or model test caches results against an utterance id.
    """
    table = Lexicon() if lexicon is None else lexicon
    utterances = []
    for index in range(config.utterances):
        seed = config.seed.fork("utterance", index)
        tokens = sample_tokens(new_rng(seed), config, table)
        languages = tuple(table.classify(token) for token in tokens)
        utterances.append(
            Utterance(
                utterance_id=f"utt-{index:05d}",
                tokens=tokens,
                languages=languages,
                text=surface_text(tokens, languages),
                seed=seed,
            )
        )
    return Corpus(tuple(utterances), config)


DEFAULT_SPLIT_NAMES = ("train", "dev", "test")


def split_corpus(
    corpus: Corpus,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    names: Sequence[str] = DEFAULT_SPLIT_NAMES,
) -> dict[str, Corpus]:
    """Partition a corpus by a seeded shuffle, then contiguous slices.

    The shuffle is seeded from the corpus seed rather than from the caller's RNG so
    that the same corpus always splits the same way -- a decoder tuned on one run
    of the dev split has to see the same dev split next time.  Ratios are
    normalised, and each non-empty partition gets at least one utterance.
    """
    require(
        len(ratios) == len(names),
        f"{len(ratios)} ratios cannot name {len(names)} splits",
        ConfigError,
    )
    require(len(ratios) > 0, "a split needs at least one partition", ConfigError)
    require(
        all(ratio >= 0.0 for ratio in ratios),
        f"split ratios must not be negative, got {tuple(ratios)}",
        ConfigError,
    )
    total = float(sum(ratios))
    require(total > 0.0, "split ratios must sum to something positive", ConfigError)
    require(
        len(corpus) >= len(ratios),
        f"cannot split {len(corpus)} utterances into {len(ratios)} partitions",
        ConfigError,
    )
    order = np.argsort(new_rng(corpus.config.seed.fork("split")).random(len(corpus)))
    shuffled = [corpus.utterances[int(index)] for index in order]
    counts = _allocate(len(shuffled), [ratio / total for ratio in ratios])
    partitions: dict[str, Corpus] = {}
    cursor = 0
    for name, count in zip(names, counts, strict=True):
        partitions[name] = Corpus(tuple(shuffled[cursor : cursor + count]), corpus.config)
        cursor += count
    return partitions


def _allocate(total: int, shares: Sequence[float]) -> list[int]:
    """Turn normalised shares into counts that add up to ``total`` exactly.

    Largest-remainder rounding, so no utterance is lost or duplicated by the split.
    """
    exact = [total * share for share in shares]
    floors = [int(value) for value in exact]
    remainder = total - sum(floors)
    order = sorted(range(len(exact)), key=lambda index: exact[index] - floors[index], reverse=True)
    for index in order[:remainder]:
        floors[index] += 1
    return floors
