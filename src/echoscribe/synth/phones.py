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


#: Vowel rows: symbol, three (centre, bandwidth) formant pairs, duration in ms.
#: The values are invented -- spread out enough that a 40-bin log-mel filterbank
#: at 16 kHz puts each vowel in a different corner of the F1/F2 plane.
_VOWEL_ROWS = (
    ("aa", (720.0, 90.0), (1240.0, 100.0), (2520.0, 150.0), 130.0),
    ("ae", (660.0, 90.0), (1720.0, 110.0), (2560.0, 150.0), 120.0),
    ("ah", (620.0, 90.0), (1180.0, 100.0), (2480.0, 150.0), 95.0),
    ("ao", (560.0, 85.0), (880.0, 95.0), (2420.0, 150.0), 130.0),
    ("aw", (700.0, 90.0), (1120.0, 100.0), (2460.0, 150.0), 140.0),
    ("ay", (680.0, 90.0), (1620.0, 110.0), (2540.0, 150.0), 140.0),
    ("eh", (520.0, 80.0), (1820.0, 110.0), (2520.0, 150.0), 100.0),
    ("er", (480.0, 80.0), (1360.0, 130.0), (1700.0, 160.0), 110.0),
    ("ih", (400.0, 75.0), (1920.0, 110.0), (2560.0, 150.0), 80.0),
    ("iy", (280.0, 65.0), (2280.0, 120.0), (2960.0, 160.0), 120.0),
    ("ow", (440.0, 80.0), (880.0, 95.0), (2380.0, 150.0), 130.0),
    ("uh", (430.0, 80.0), (1120.0, 100.0), (2440.0, 150.0), 80.0),
    ("uw", (310.0, 70.0), (880.0, 95.0), (2340.0, 150.0), 120.0),
)

#: Voiceless and voiced consonant rows: symbol, manner, voiced, formants,
#: duration, gain, and whether a turbulent noise source is mixed in.
_CONSONANT_ROWS = (
    ("sil", "silence", False, ((200.0, 150.0),), 120.0, 0.001, False),
    ("p", "plosive", False, ((700.0, 160.0), (1500.0, 320.0)), 55.0, 0.45, True),
    ("t", "plosive", False, ((1800.0, 220.0), (3200.0, 420.0)), 50.0, 0.5, True),
    ("k", "plosive", False, ((1500.0, 240.0), (2400.0, 420.0)), 55.0, 0.5, True),
    ("b", "plosive", True, ((300.0, 120.0), (1100.0, 260.0)), 60.0, 0.45, True),
    ("d", "plosive", True, ((400.0, 130.0), (1700.0, 300.0)), 58.0, 0.5, True),
    ("g", "plosive", True, ((400.0, 130.0), (2100.0, 320.0)), 62.0, 0.5, True),
    ("f", "fricative", False, ((5200.0, 1600.0),), 90.0, 0.3, True),
    ("s", "fricative", False, ((5600.0, 1200.0),), 100.0, 0.4, True),
    ("sh", "fricative", False, ((3200.0, 1100.0),), 100.0, 0.4, True),
    ("th", "fricative", False, ((4400.0, 1500.0),), 90.0, 0.3, True),
    ("hh", "fricative", False, ((1200.0, 700.0), (2200.0, 900.0)), 70.0, 0.25, True),
    ("v", "fricative", True, ((4800.0, 1500.0),), 85.0, 0.28, True),
    ("z", "fricative", True, ((5200.0, 1200.0),), 90.0, 0.32, True),
    ("m", "nasal", True, ((280.0, 90.0), (1100.0, 200.0), (2200.0, 320.0)), 75.0, 0.6, False),
    ("n", "nasal", True, ((320.0, 90.0), (1500.0, 220.0), (2600.0, 320.0)), 70.0, 0.6, False),
    ("ng", "nasal", True, ((300.0, 90.0), (1000.0, 200.0), (2100.0, 300.0)), 75.0, 0.6, False),
    ("l", "approximant", True, ((380.0, 80.0), (1200.0, 160.0), (2600.0, 240.0)), 70.0, 0.7, False),
    ("r", "approximant", True, ((340.0, 80.0), (1100.0, 150.0), (1650.0, 200.0)), 70.0, 0.7, False),
    ("w", "approximant", True, ((300.0, 70.0), (700.0, 110.0), (2300.0, 240.0)), 70.0, 0.7, False),
    ("y", "approximant", True, ((280.0, 70.0), (2200.0, 140.0), (2900.0, 200.0)), 65.0, 0.7, False),
)


def _build_vowels() -> dict[str, Phone]:
    return {
        symbol: Phone(symbol, "vowel", True, duration, (first, second, third), 1.0, False)
        for symbol, first, second, third, duration in _VOWEL_ROWS
    }


def _build_consonants() -> dict[str, Phone]:
    return {
        symbol: Phone(symbol, manner, voiced, duration, formants, gain, turbulent)
        for symbol, manner, voiced, formants, duration, gain, turbulent in _CONSONANT_ROWS
    }


#: Every English-style phone the synthesizer can render, keyed by symbol.
ENGLISH_PHONES: dict[str, Phone] = {**_build_vowels(), **_build_consonants()}
