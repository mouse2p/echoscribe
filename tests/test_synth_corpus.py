"""Corpus generation is checked for reproducibility, coverage and its own limits."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.seeds import DEFAULT_ROOT, Seed
from echoscribe.synth.corpus import CorpusConfig, Utterance, surface_text


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


def _utterance(tokens, languages) -> Utterance:
    return Utterance(
        "utt-00000",
        tuple(tokens),
        tuple(languages),
        surface_text(tokens, languages),
        Seed(1, "test"),
    )


def test_english_tokens_are_space_separated() -> None:
    assert surface_text(("hello", "world"), ("en", "en")) == "hello world"


def test_chinese_tokens_are_not_space_separated() -> None:
    assert surface_text(("你", "好"), ("zh", "zh")) == "你好"


def test_a_script_switch_inserts_one_space() -> None:
    assert surface_text(("hello", "你", "好", "world"), ("en", "zh", "zh", "en")) == (
        "hello 你好 world"
    )


def test_a_single_token_has_no_separator() -> None:
    assert surface_text(("你",), ("zh",)) == "你"
    assert surface_text(("hello",), ("en",)) == "hello"


def test_surface_text_rejects_a_label_count_mismatch() -> None:
    with pytest.raises(ConfigError, match="cannot carry"):
        surface_text(("a", "b"), ("en",))


def test_an_utterance_reports_whether_it_switches_script() -> None:
    assert _utterance(("hello", "你"), ("en", "zh")).is_code_switched
    assert not _utterance(("hello", "world"), ("en", "en")).is_code_switched


def test_an_utterance_counts_tokens_per_language() -> None:
    utterance = _utterance(("hello", "你", "好", "world"), ("en", "zh", "zh", "en"))
    assert utterance.num_tokens == 4
    assert utterance.token_count("en") == 2
    assert utterance.token_count("zh") == 2


def test_token_count_rejects_an_unknown_language() -> None:
    with pytest.raises(ConfigError, match="unknown language"):
        _utterance(("hello",), ("en",)).token_count("fr")


def test_an_utterance_keeps_its_transcript_consistent_with_its_tokens() -> None:
    utterance = _utterance(("start", "你"), ("en", "zh"))
    assert utterance.text == "start 你"
    assert utterance.tokens == ("start", "你")


@pytest.mark.parametrize(
    ("tokens", "languages"),
    [
        ((), ()),
        (("a",), ("en", "zh")),
        (("a", "b"), ("en", "fr")),
    ],
)
def test_malformed_utterances_are_rejected(tokens, languages) -> None:
    with pytest.raises(ConfigError):
        Utterance("utt", tokens, languages, "text", Seed(1, "x"))


def test_an_utterance_needs_an_id() -> None:
    with pytest.raises(ConfigError, match="needs an id"):
        Utterance("", ("a",), ("en",), "a", Seed(1, "x"))
