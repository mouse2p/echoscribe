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

from dataclasses import dataclass

from echoscribe.errors import ConfigError, require, require_range
from echoscribe.seeds import DEFAULT_ROOT, Seed


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
