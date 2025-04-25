"""The phone inventory the formant synthesizer renders.

The entries below are *invented*.  Formant frequencies, bandwidths and durations
were chosen so that each phone is spectrally distinguishable from its neighbours
in a log-mel filterbank, not measured from a speaker, and they are deliberately
coarse: a three-formant description of a vowel is a cartoon of articulation.
Nothing here is a claim about how real speech is produced.

Two inventories live side by side because the evaluation suite is code-switch
aware:

``ENGLISH_PHONES``
    an ARPAbet-style symbol set covering the vowels and consonants needed to spell
    a small list of common English words.
``INITIALS`` / ``FINALS``
    the consonant onset and the vowel nucleus of a Mandarin syllable, kept apart
    because a syllable's tone is realised on the final, not on the initial.
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import ConfigError, require, require_in

#: Articulatory classes the synthesizer knows how to render.
MANNERS = ("silence", "vowel", "nasal", "plosive", "fricative", "approximant")

#: Formant description used for a voiced phone: ``(centre_hz, bandwidth_hz)``.
Formant = tuple[float, float]


@dataclass(frozen=True)
class Phone:
    """One synthetic sound: how long it lasts and what resonates in it."""

    symbol: str
    manner: str
    voiced: bool
    duration_ms: float
    formants: tuple[Formant, ...]
    gain: float = 1.0
    turbulent: bool = False

    def __post_init__(self) -> None:
        require(bool(self.symbol), "a phone symbol must not be empty", ConfigError)
        require_in("manner", self.manner, MANNERS, ConfigError)
        require(self.duration_ms > 0.0, f"{self.symbol}: duration must be positive", ConfigError)
        require(0.0 < self.gain <= 1.0, f"{self.symbol}: gain must lie in (0, 1]", ConfigError)
        require(len(self.formants) >= 1, f"{self.symbol}: needs at least one formant", ConfigError)
        for index, (centre, bandwidth) in enumerate(self.formants):
            require(
                centre > 0.0, f"{self.symbol}: formant {index} centre must be positive", ConfigError
            )
            require(
                bandwidth > 0.0,
                f"{self.symbol}: formant {index} bandwidth must be positive",
                ConfigError,
            )

    @property
    def is_silent(self) -> bool:
        """True for the silence phone, which carries no source at all."""
        return self.manner == "silence"
