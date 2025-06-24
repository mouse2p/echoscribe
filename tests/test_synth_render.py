"""Render configuration and acoustic conditions are validated up front."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.seeds import DEFAULT_ROOT
from echoscribe.synth.corpus import CorpusConfig, Utterance, generate_corpus
from echoscribe.synth.formant import samples_for_duration
from echoscribe.synth.lexicon import Lexicon
from echoscribe.synth.manifest import read_manifest, verify_manifest
from echoscribe.synth.noise import signal_to_noise_db
from echoscribe.synth.render import (
    AcousticCondition,
    RenderConfig,
    apply_condition,
    render_corpus,
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


def test_a_clean_condition_leaves_the_audio_alone() -> None:
    audio = render_utterance(_utterance(("hello",)), RenderConfig())
    assert np.array_equal(
        apply_condition(audio, AcousticCondition(), DEFAULT_ROOT).samples, audio.samples
    )


def test_reverberation_lengthens_the_utterance() -> None:
    audio = render_utterance(_utterance(("hello",)), RenderConfig())
    reverberant = apply_condition(audio, AcousticCondition(rt60_ms=300.0), DEFAULT_ROOT)
    assert reverberant.num_frames > audio.num_frames


@pytest.mark.parametrize("target", [20.0, 10.0])
def test_mixing_noise_measures_close_to_the_requested_snr(target: float) -> None:
    # A low peak target keeps the limiter out of the measurement: the SNR check
    # compares the degraded render against the clean one, which only works while
    # nothing has been rescaled.
    audio = render_utterance(_utterance(("hello", "world")), RenderConfig(peak_target=0.3))
    condition = AcousticCondition(noise_kind="white", noise_snr_db=target)
    noisy = apply_condition(audio, condition, DEFAULT_ROOT)
    measured = signal_to_noise_db(audio.samples, noisy.samples)
    assert measured == pytest.approx(target, abs=0.5)


def test_the_limiter_keeps_a_noisy_render_inside_the_pcm_range() -> None:
    audio = render_utterance(_utterance(("hello", "world")), RenderConfig(peak_target=0.95))
    condition = AcousticCondition(noise_kind="white", noise_snr_db=-10.0)
    degraded = apply_condition(audio, condition, DEFAULT_ROOT)
    assert float(np.max(np.abs(degraded.samples))) <= 1.0


def test_the_limiter_is_a_uniform_gain_so_the_snr_survives_it() -> None:
    condition = AcousticCondition(noise_kind="white", noise_snr_db=-10.0)
    # mix_at_snr scales the noise with the signal, so a quiet render and a loud
    # render of the same utterance differ only by a constant -- unless one of them
    # hits the limiter.  Proportionality is therefore the limiter's whole contract.
    quiet = render_utterance(_utterance(("hello", "world")), RenderConfig(peak_target=0.05))
    loud = render_utterance(_utterance(("hello", "world")), RenderConfig(peak_target=0.95))
    soft = apply_condition(quiet, condition, DEFAULT_ROOT).samples.astype(np.float64)
    hard = apply_condition(loud, condition, DEFAULT_ROOT).samples.astype(np.float64)
    assert float(np.max(np.abs(soft))) <= 1.0
    assert float(np.max(np.abs(hard))) == pytest.approx(1.0, abs=1e-5)
    ratio = float(np.dot(soft, hard) / np.dot(soft, soft))
    assert np.allclose(hard, soft * ratio, rtol=1e-4, atol=1e-6)


def test_speed_perturbation_changes_the_rendered_duration() -> None:
    audio = render_utterance(_utterance(("hello",)), RenderConfig())
    slower = apply_condition(audio, AcousticCondition(speed_factor=1.2), DEFAULT_ROOT)
    assert slower.num_frames == pytest.approx(audio.num_frames * 1.2, abs=3)


def test_a_condition_is_reproducible_from_the_seed() -> None:
    audio = render_utterance(_utterance(("hello",)), RenderConfig())
    condition = AcousticCondition(noise_kind="pink", noise_snr_db=8.0, rt60_ms=200.0)
    left = apply_condition(audio, condition, DEFAULT_ROOT)
    right = apply_condition(audio, condition, DEFAULT_ROOT)
    assert np.array_equal(left.samples, right.samples)


def test_a_different_seed_gives_different_noise() -> None:
    from echoscribe.seeds import Seed

    audio = render_utterance(_utterance(("hello",)), RenderConfig())
    condition = AcousticCondition(noise_kind="white", noise_snr_db=10.0)
    left = apply_condition(audio, condition, Seed(1, "a"))
    right = apply_condition(audio, condition, Seed(2, "a"))
    assert not np.array_equal(left.samples, right.samples)


def _small_corpus(count: int = 3):
    return generate_corpus(CorpusConfig(utterances=count, min_tokens=2, max_tokens=3))


def test_rendering_a_corpus_writes_wavs_and_a_manifest(tmp_path) -> None:
    records = render_corpus(_small_corpus(), RenderConfig(), tmp_path)
    assert len(records) == 3
    assert (tmp_path / "manifest.jsonl").is_file()
    for record in records:
        assert (tmp_path / record.wav_path).is_file()


def test_the_manifest_reads_back_as_written(tmp_path) -> None:
    records = render_corpus(_small_corpus(), RenderConfig(), tmp_path)
    assert read_manifest(tmp_path / "manifest.jsonl") == records


def test_a_rendered_corpus_verifies_clean(tmp_path) -> None:
    records = render_corpus(_small_corpus(), RenderConfig(), tmp_path)
    assert verify_manifest(tmp_path, records) == []


def test_tampering_with_one_wav_is_detected(tmp_path) -> None:
    records = render_corpus(_small_corpus(), RenderConfig(), tmp_path)
    target = tmp_path / records[1].wav_path
    target.write_bytes(target.read_bytes() + b"\x00\x00")
    assert verify_manifest(tmp_path, records) == [records[1].utterance_id]


def test_records_carry_the_planted_transcript(tmp_path) -> None:
    corpus = _small_corpus()
    records = render_corpus(corpus, RenderConfig(), tmp_path)
    assert [record.text for record in records] == [u.text for u in corpus]
    assert [record.tokens for record in records] == [u.tokens for u in corpus]


def test_records_carry_the_acoustic_condition(tmp_path) -> None:
    config = RenderConfig(condition=AcousticCondition(noise_kind="pink", noise_snr_db=12.0))
    records = render_corpus(_small_corpus(2), config, tmp_path)
    assert all(record.condition["noise_kind"] == "pink" for record in records)
    assert all(record.condition["noise_snr_db"] == 12.0 for record in records)


def test_rendering_the_same_corpus_twice_gives_the_same_bytes(tmp_path) -> None:
    config = RenderConfig()
    left = render_corpus(_small_corpus(2), config, tmp_path / "a")
    right = render_corpus(_small_corpus(2), config, tmp_path / "b")
    assert [record.checksum for record in left] == [record.checksum for record in right]


def test_record_durations_match_the_written_audio(tmp_path) -> None:
    from echoscribe.audio import read_wav

    records = render_corpus(_small_corpus(2), RenderConfig(), tmp_path)
    for record in records:
        audio = read_wav(tmp_path / record.wav_path)
        assert audio.num_frames == record.num_samples
        assert audio.duration == pytest.approx(record.duration, abs=1e-9)
        assert audio.sample_rate == record.sample_rate
