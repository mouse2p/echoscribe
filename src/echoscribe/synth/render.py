"""End-to-end rendering: a transcript becomes a waveform and a manifest line.

This is the module that ties the laboratory together.  It takes an
:class:`~echoscribe.synth.corpus.Utterance`, looks its tokens up in the lexicon,
renders each syllable or word with the phone synthesizer, joins them with silence,
optionally degrades the result, and writes a wav plus a manifest record.

The pipeline order is fixed and matters:

1. render clean audio from the phone sequence;
2. peak-normalise, so the degradation stage starts from a known level;
3. apply the room, because reverberation happens before the microphone's noise;
4. mix noise at the requested SNR, measured against the reverberant signal;
5. apply speed and volume perturbation last, so a corpus augmented by perturbation
   keeps the SNR and RT60 it was specified with.

Every stage is optional except the first two, and the settings that were used are
written into the manifest so a result can always be attributed to a condition.

The order is fixed rather than configurable on purpose.  A caller who mixed noise
before the room would get a reverberant noise floor, which is a legitimate acoustic
condition but a different one, and a result labelled "20 dB SNR" would no longer
mean what it says.  Keeping the order in one place means the label on a result and
the signal it describes cannot drift apart.

The limiter at the end is the one exception to "degrade, then write".  It exists
because 16-bit PCM cannot hold a sample above 1.0, and silently clipping would
introduce distortion the manifest does not record.  Scaling the whole signal down
instead is inaudible in the features and leaves the SNR untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echoscribe.audio import write_wav
from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.seeds import Seed, new_rng
from echoscribe.synth.corpus import Corpus, Utterance
from echoscribe.synth.formant import normalize_peak, render_sequence, samples_for_duration
from echoscribe.synth.lexicon import Lexicon
from echoscribe.synth.manifest import (
    ManifestRecord,
    checksum_of_file,
    write_manifest,
)
from echoscribe.synth.noise import NOISE_KINDS, generate_noise, mix_at_snr
from echoscribe.synth.perturb import speed_perturb, volume_perturb
from echoscribe.synth.room import RoomConfig, apply_room
from echoscribe.synth.tone import tone_by_index
from echoscribe.waveform import Audio, concatenate, silence


@dataclass(frozen=True)
class AcousticCondition:
    """The degradation applied to one rendered utterance."""

    noise_kind: str | None = None
    noise_snr_db: float = 20.0
    rt60_ms: float | None = None
    speed_factor: float = 1.0
    volume_db: float = 0.0

    def __post_init__(self) -> None:
        if self.noise_kind is not None:
            require(
                self.noise_kind in NOISE_KINDS,
                f"noise_kind must be one of {NOISE_KINDS} or None, got {self.noise_kind!r}",
                ConfigError,
            )
        if self.rt60_ms is not None:
            require_positive("rt60_ms", self.rt60_ms, ConfigError)
        require_positive("speed_factor", self.speed_factor, ConfigError)

    @property
    def label(self) -> str:
        """A short human-readable name for the condition, used in reports."""
        parts = []
        if self.rt60_ms is not None:
            parts.append(f"rt60-{int(self.rt60_ms)}ms")
        if self.noise_kind is not None:
            parts.append(f"{self.noise_kind}-{self.noise_snr_db:g}dB")
        if self.speed_factor != 1.0:
            parts.append(f"speed-{self.speed_factor:g}")
        if self.volume_db != 0.0:
            parts.append(f"volume-{self.volume_db:g}dB")
        return "+".join(parts) if parts else "clean"

    def to_dict(self) -> dict:
        """A JSON-ready mapping of every field, in a fixed key order."""
        return {
            "noise_kind": self.noise_kind,
            "noise_snr_db": round(float(self.noise_snr_db), 6),
            "rt60_ms": None if self.rt60_ms is None else round(float(self.rt60_ms), 6),
            "speed_factor": round(float(self.speed_factor), 6),
            "volume_db": round(float(self.volume_db), 6),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> AcousticCondition:
        """Rebuild a condition from :meth:`to_dict`."""
        for name in ("noise_kind", "noise_snr_db", "rt60_ms", "speed_factor", "volume_db"):
            require(name in payload, f"acoustic condition is missing {name!r}", ConfigError)
        return cls(
            noise_kind=payload["noise_kind"],
            noise_snr_db=float(payload["noise_snr_db"]),
            rt60_ms=payload["rt60_ms"],
            speed_factor=float(payload["speed_factor"]),
            volume_db=float(payload["volume_db"]),
        )


@dataclass(frozen=True)
class RenderConfig:
    """How an utterance is turned into audio."""

    sample_rate: int = 16000
    base_f0: float = 120.0
    leading_silence_ms: float = 250.0
    trailing_silence_ms: float = 250.0
    boundary_silence_ms: float = 80.0
    peak_target: float = 0.9
    condition: AcousticCondition = AcousticCondition()

    def __post_init__(self) -> None:
        require(
            isinstance(self.sample_rate, int)
            and not isinstance(self.sample_rate, bool)
            and self.sample_rate > 0,
            f"sample_rate must be a positive integer, got {self.sample_rate!r}",
            ConfigError,
        )
        require_positive("base_f0", self.base_f0, ConfigError)
        for name, value in (
            ("leading_silence_ms", self.leading_silence_ms),
            ("trailing_silence_ms", self.trailing_silence_ms),
            ("boundary_silence_ms", self.boundary_silence_ms),
        ):
            require(
                isinstance(value, int | float) and value >= 0.0,
                f"{name} must be a non-negative number, got {value!r}",
                ConfigError,
            )
        require(
            0.0 < self.peak_target <= 1.0,
            f"peak_target must lie in (0, 1], got {self.peak_target!r}",
            ConfigError,
        )
        require(
            isinstance(self.condition, AcousticCondition),
            f"condition must be an AcousticCondition, got {self.condition!r}",
            ConfigError,
        )


def render_token(
    token: str,
    language: str,
    seed: Seed,
    config: RenderConfig,
    lexicon: Lexicon | None = None,
) -> Audio:
    """Render one surface token -- an English word or one Chinese character.

    A Chinese character is rendered under its lexical tone; an English word is
    rendered at a flat ``base_f0``.  The turbulence generator is forked from the
    token's own seed, so re-rendering one token does not disturb the noise in any
    other.
    """
    table = Lexicon() if lexicon is None else lexicon
    rng = new_rng(seed.fork("source"))
    if language == "en":
        phones = table.word_phones(token)
        samples = render_sequence(phones, config.sample_rate, config.base_f0, None, rng)
    else:
        syllable = table.mandarin[token]
        phones = table.syllable_phones(token)
        samples = render_sequence(
            phones, config.sample_rate, config.base_f0, tone_by_index(syllable.tone), rng
        )
    return Audio(samples.astype(np.float32), config.sample_rate)


def render_utterance(
    utterance: Utterance,
    config: RenderConfig,
    lexicon: Lexicon | None = None,
) -> Audio:
    """Render a whole utterance: leading silence, tokens, trailing silence.

    Boundary silence is not cosmetic.  An endpointer needs quiet to detect the
    start of speech, a feature pipeline needs a frame or two to warm up, and a
    decoder needs trailing silence to emit its final hypothesis.  The lengths are
    therefore part of the render configuration rather than something a caller has
    to remember to pad.

    The result is peak-normalised, so the degradation stage that follows always
    starts from a known level.
    """
    rate = config.sample_rate
    pieces = [silence(config.leading_silence_ms / 1000.0, rate)]
    for index, (token, language) in enumerate(
        zip(utterance.tokens, utterance.languages, strict=True)
    ):
        if index:
            pieces.append(silence(config.boundary_silence_ms / 1000.0, rate))
        pieces.append(
            render_token(token, language, utterance.seed.fork("token", index), config, lexicon)
        )
    pieces.append(silence(config.trailing_silence_ms / 1000.0, rate))
    joined = concatenate(pieces)
    scaled = normalize_peak(joined.samples.astype(np.float64), config.peak_target)
    return Audio(scaled.astype(np.float32), rate)


def utterance_audio_length(utterance: Utterance, config: RenderConfig, lexicon: Lexicon) -> int:
    """Predict an utterance's rendered length without rendering it.

    Useful for a latency budget: a streaming test can size its chunk schedule from
    the number alone.
    """
    rate = config.sample_rate
    total = samples_for_duration(config.leading_silence_ms, rate)
    total += samples_for_duration(config.trailing_silence_ms, rate)
    for index, token in enumerate(utterance.tokens):
        if index:
            total += samples_for_duration(config.boundary_silence_ms, rate)
        phones = lexicon.phones_for(token)
        total += sum(samples_for_duration(phone.duration_ms, rate) for phone in phones)
    return total


def apply_condition(audio: Audio, condition: AcousticCondition, seed: Seed) -> Audio:
    """Degrade ``audio`` according to ``condition``, in the documented order.

    Room first, then noise, then perturbation.  A room colours the signal before a
    microphone adds its floor, and speed or volume changes are a property of the
    recording rather than of the acoustics, so they come last.  Mixing noise before
    the room would reverberate the noise too, which is a different -- and much less
    common -- condition.

    A final limiter scales down only if the chain would clip.  It scales signal and
    noise together, so the SNR the caller asked for survives it.
    """
    result = audio
    if condition.rt60_ms is not None:
        room = RoomConfig(rt60_ms=condition.rt60_ms, seed=seed.fork("room"))
        result = apply_room(result, room)
    if condition.noise_kind is not None:
        noise = generate_noise(
            condition.noise_kind,
            result.num_frames,
            new_rng(seed.fork("noise")),
            result.sample_rate,
        )
        result = mix_at_snr(result, noise, condition.noise_snr_db)
    if condition.speed_factor != 1.0:
        result = speed_perturb(result, condition.speed_factor)
    if condition.volume_db != 0.0:
        result = volume_perturb(result, condition.volume_db).audio
    peak = float(np.max(np.abs(result.samples))) if result.num_frames else 0.0
    if peak > 1.0:
        limited = result.samples.astype(np.float64) / peak
        result = Audio(limited.astype(np.float32), result.sample_rate)
    return result


MANIFEST_NAME = "manifest.jsonl"
WAV_DIRECTORY = "wav"


def render_corpus(
    corpus: Corpus,
    config: RenderConfig,
    directory: str | Path,
    lexicon: Lexicon | None = None,
) -> list[ManifestRecord]:
    """Render every utterance to ``directory/wav`` and write ``directory/manifest.jsonl``.

    Each wav is checksummed after it is written, so the manifest describes the bytes
    that are actually on disk rather than the array that was in memory.  The corpus
    digest is returned through :func:`write_manifest` for a caller that wants to
    record it; the records themselves come back so a test can inspect them without
    re-reading the file.
    """
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    records = []
    for utterance in corpus:
        clean = render_utterance(utterance, config, lexicon)
        audio = apply_condition(clean, config.condition, utterance.seed)
        relative = f"{WAV_DIRECTORY}/{utterance.utterance_id}.wav"
        write_wav(base / relative, audio)
        records.append(
            ManifestRecord(
                utterance_id=utterance.utterance_id,
                text=utterance.text,
                tokens=utterance.tokens,
                languages=utterance.languages,
                wav_path=relative,
                sample_rate=audio.sample_rate,
                num_samples=audio.num_frames,
                duration=audio.duration,
                seed=utterance.seed.value,
                checksum=checksum_of_file(base / relative),
                condition=config.condition.to_dict(),
            )
        )
    write_manifest(base / MANIFEST_NAME, records)
    return records
