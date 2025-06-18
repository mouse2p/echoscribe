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
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.synth.noise import NOISE_KINDS


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
