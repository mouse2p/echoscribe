"""Render configuration and acoustic conditions are validated up front."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.seeds import DEFAULT_ROOT
from echoscribe.synth.corpus import Utterance
from echoscribe.synth.formant import samples_for_duration
from echoscribe.synth.lexicon import Lexicon
from echoscribe.synth.render import (
    AcousticCondition,
    RenderConfig,
    render_token,
    render_utterance,
    utterance_audio_length,
)


def test_the_default_condition_is_clean() -> None:
    assert AcousticCondition().label == "clean"
    assert RenderConfig().condition.label == "clean"


def test_a_condition_label_names_every_active_stage() -> None:
    condition = AcousticCondition(noise_kind="pink", noise_snr_db=10.0, rt60_ms=300.0)
    assert condition.label == "rt60-300ms+pink-10dB"


def test_perturbation_appears_in_the_label() -> None:
    assert AcousticCondition(speed_factor=1.1).label == "speed-1.1"
    assert AcousticCondition(volume_db=-6.0).label == "volume--6dB"


def test_a_condition_round_trips_through_a_dict() -> None:
    condition = AcousticCondition(noise_kind="modulated", noise_snr_db=5.0, rt60_ms=250.0)
    assert AcousticCondition.from_dict(condition.to_dict()) == condition


def test_a_serialized_condition_rounds_at_the_production_point() -> None:
    payload = AcousticCondition(noise_snr_db=1.0 / 3.0).to_dict()
    assert payload["noise_snr_db"] == 0.333333


def test_a_condition_from_a_dict_rejects_a_missing_field() -> None:
    payload = AcousticCondition().to_dict()
    del payload["speed_factor"]
    with pytest.raises(ConfigError, match="missing 'speed_factor'"):
        AcousticCondition.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_kind": "babble"},
        {"rt60_ms": 0.0},
        {"rt60_ms": -10.0},
        {"speed_factor": 0.0},
    ],
)
def test_invalid_conditions_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        AcousticCondition(**kwargs)


def test_render_defaults_describe_a_16k_render() -> None:
    config = RenderConfig()
    assert config.sample_rate == 16000
    assert config.base_f0 == 120.0
    assert config.leading_silence_ms == 250.0
    assert config.boundary_silence_ms == 80.0
    assert config.peak_target == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": 16000.0},
        {"base_f0": 0.0},
        {"leading_silence_ms": -1.0},
        {"boundary_silence_ms": -0.5},
        {"peak_target": 0.0},
        {"peak_target": 1.5},
        {"condition": "clean"},
    ],
)
def test_invalid_render_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        RenderConfig(**kwargs)


def test_both_configs_are_frozen() -> None:
    with pytest.raises(AttributeError):
        RenderConfig().sample_rate = 8000  # type: ignore[misc]
    with pytest.raises(AttributeError):
        AcousticCondition().volume_db = 3.0  # type: ignore[misc]


def test_a_condition_is_hashable_so_it_can_key_a_cache() -> None:
    first = AcousticCondition(noise_kind="white", noise_snr_db=15.0)
    second = AcousticCondition(noise_kind="white", noise_snr_db=15.0)
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_the_dict_form_has_a_stable_key_order() -> None:
    payload = AcousticCondition(noise_kind="white").to_dict()
    assert list(payload) == [
        "noise_kind",
        "noise_snr_db",
        "rt60_ms",
        "speed_factor",
        "volume_db",
    ]


def test_a_render_config_carries_its_condition_through() -> None:
    condition = AcousticCondition(rt60_ms=400.0)
    assert RenderConfig(condition=condition).condition is condition


def test_the_default_condition_is_shared_but_immutable() -> None:
    assert RenderConfig().condition is RenderConfig().condition
    assert RenderConfig().condition.to_dict() == AcousticCondition().to_dict()


def test_a_condition_label_is_stable_for_the_same_settings() -> None:
    left = AcousticCondition(noise_kind="white", noise_snr_db=20.0, rt60_ms=150.0)
    right = AcousticCondition(noise_kind="white", noise_snr_db=20.0, rt60_ms=150.0)
    assert left.label == right.label == "rt60-150ms+white-20dB"


def _expected_length(phones, rate: int) -> int:
    return sum(samples_for_duration(phone.duration_ms, rate) for phone in phones)


def test_an_english_word_renders_to_its_phones_total_length() -> None:
    lexicon = Lexicon()
    config = RenderConfig()
    audio = render_token("hello", "en", DEFAULT_ROOT, config, lexicon)
    assert audio.num_frames == _expected_length(lexicon.word_phones("hello"), config.sample_rate)


def test_a_chinese_character_renders_initial_plus_final() -> None:
    lexicon = Lexicon()
    config = RenderConfig()
    audio = render_token("你", "zh", DEFAULT_ROOT, config, lexicon)
    assert audio.num_frames == _expected_length(lexicon.syllable_phones("你"), config.sample_rate)


def test_a_rendered_token_is_finite_and_quiet_enough_to_mix() -> None:
    audio = render_token("stream", "en", DEFAULT_ROOT, RenderConfig())
    assert np.all(np.isfinite(audio.samples))
    assert float(np.max(np.abs(audio.samples))) < 4.0


def test_the_same_seed_renders_the_same_audio() -> None:
    left = render_token("world", "en", DEFAULT_ROOT, RenderConfig())
    right = render_token("world", "en", DEFAULT_ROOT, RenderConfig())
    assert np.array_equal(left.samples, right.samples)


def test_a_different_seed_changes_a_turbulent_token() -> None:
    from echoscribe.seeds import Seed

    left = render_token("session", "en", Seed(1, "a"), RenderConfig())
    right = render_token("session", "en", Seed(2, "a"), RenderConfig())
    assert not np.array_equal(left.samples, right.samples)


def test_a_tone_changes_a_rendered_character() -> None:
    from echoscribe.synth.lexicon import MandarinSyllable

    lexicon = Lexicon(mandarin={"X": MandarinSyllable("X", "h", "a", 1)})
    rising = Lexicon(mandarin={"X": MandarinSyllable("X", "h", "a", 2)})
    config = RenderConfig()
    first = render_token("X", "zh", DEFAULT_ROOT, config, lexicon)
    second = render_token("X", "zh", DEFAULT_ROOT, config, rising)
    assert first.num_frames == second.num_frames
    assert not np.allclose(first.samples, second.samples)


def test_rendering_respects_the_configured_sample_rate() -> None:
    audio = render_token("code", "en", DEFAULT_ROOT, RenderConfig(sample_rate=8000))
    assert audio.sample_rate == 8000


def _utterance(tokens=("hello", "你", "好")) -> Utterance:
    lexicon = Lexicon()
    languages = tuple(lexicon.classify(token) for token in tokens)
    return Utterance("utt-test", tokens, languages, " ".join(tokens), DEFAULT_ROOT)


def test_the_rendered_length_matches_the_prediction() -> None:
    utterance = _utterance()
    config = RenderConfig()
    assert render_utterance(utterance, config).num_frames == utterance_audio_length(
        utterance, config, Lexicon()
    )


def test_the_leading_and_trailing_silence_are_exact() -> None:
    config = RenderConfig(leading_silence_ms=200.0, trailing_silence_ms=150.0)
    audio = render_utterance(_utterance(("code",)), config)
    lead = samples_for_duration(200.0, config.sample_rate)
    trail = samples_for_duration(150.0, config.sample_rate)
    assert np.all(audio.samples[:lead] == 0.0)
    assert np.all(audio.samples[-trail:] == 0.0)
    assert np.max(np.abs(audio.samples[lead:-trail])) > 0.1


def test_tokens_are_separated_by_exactly_the_boundary_silence() -> None:
    config = RenderConfig(leading_silence_ms=0.0, trailing_silence_ms=0.0, boundary_silence_ms=50.0)
    lexicon = Lexicon()
    audio = render_utterance(_utterance(("hello", "world")), config)
    first = _expected_length(lexicon.word_phones("hello"), config.sample_rate)
    second = _expected_length(lexicon.word_phones("world"), config.sample_rate)
    gap = samples_for_duration(50.0, config.sample_rate)
    assert audio.num_frames == first + gap + second
    assert np.all(audio.samples[first : first + gap] == 0.0)


def test_the_render_is_peak_normalised_to_the_target() -> None:
    config = RenderConfig(peak_target=0.7)
    audio = render_utterance(_utterance(), config)
    assert float(np.max(np.abs(audio.samples))) == pytest.approx(0.7, abs=1e-5)


def test_a_render_is_reproducible() -> None:
    utterance = _utterance()
    left = render_utterance(utterance, RenderConfig())
    right = render_utterance(utterance, RenderConfig())
    assert np.array_equal(left.samples, right.samples)


def test_a_longer_token_list_gives_a_longer_render() -> None:
    config = RenderConfig()
    short = render_utterance(_utterance(("hello",)), config)
    long = render_utterance(_utterance(("hello", "world", "你")), config)
    assert long.num_frames > short.num_frames


@pytest.mark.slow
def test_a_whole_generated_corpus_renders() -> None:
    from echoscribe.synth.corpus import CorpusConfig, generate_corpus

    corpus = generate_corpus(CorpusConfig(utterances=6, min_tokens=2, max_tokens=4))
    config = RenderConfig()
    for utterance in corpus:
        audio = render_utterance(utterance, config)
        assert audio.num_frames == utterance_audio_length(utterance, config, Lexicon())
        assert np.all(np.isfinite(audio.samples))
