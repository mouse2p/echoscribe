"""Corpus generation is checked for reproducibility, coverage and its own limits."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.seeds import DEFAULT_ROOT, Seed
from echoscribe.synth.corpus import CorpusConfig


def test_defaults_describe_a_small_two_language_corpus() -> None:
    config = CorpusConfig()
    assert config.utterances == 24
    assert config.min_tokens == 2
    assert config.max_tokens == 7
    assert config.english_probability == 0.5
    assert config.seed is DEFAULT_ROOT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"utterances": 0},
        {"utterances": -1},
        {"utterances": 4.0},
        {"min_tokens": 0},
        {"max_tokens": 1, "min_tokens": 4},
        {"english_probability": -0.1},
        {"english_probability": 1.2},
        {"seed": 12345},
    ],
)
def test_invalid_corpus_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        CorpusConfig(**kwargs)


def test_a_single_language_corpus_is_allowed() -> None:
    assert CorpusConfig(english_probability=1.0).english_probability == 1.0
    assert CorpusConfig(english_probability=0.0).english_probability == 0.0


def test_a_fixed_token_count_is_allowed() -> None:
    config = CorpusConfig(min_tokens=4, max_tokens=4)
    assert config.min_tokens == config.max_tokens


def test_config_is_frozen() -> None:
    config = CorpusConfig()
    with pytest.raises(AttributeError):
        config.utterances = 3  # type: ignore[misc]


def test_an_explicit_seed_is_kept() -> None:
    seed = Seed(7, "custom")
    assert CorpusConfig(seed=seed).seed is seed
