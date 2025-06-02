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

from collections.abc import Sequence
from dataclasses import dataclass

from echoscribe.errors import ConfigError, require, require_range
from echoscribe.seeds import DEFAULT_ROOT, Seed
from echoscribe.synth.lexicon import LANGUAGES


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
