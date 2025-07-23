"""Modelling units: what a decoder actually emits, per language.

The two vocabularies in the laboratory need different units, and pretending
otherwise is where a code-switch system quietly loses accuracy:

*an English word* is spelled as its phones.  A phone-level unit set keeps the
inventory small (a few dozen) and every word is a sequence of them, so a new word
is representable without a new output symbol.
*a Mandarin syllable* is one ``initial . final . tone`` unit.  Folding the tone
into the unit -- rather than emitting it as a separate symbol -- is what makes the
unit set match the lexicon: ``ma1`` and ``ma3`` are different words, and a decoder
that emits ``m``, ``a``, then a tone has to get the *alignment* right as well as the
identity, which is a harder problem for no gain.

Symbols use ``.`` as a separator and ``_`` for a zero initial, so ``n.i.3`` is the
third tone of ``ni`` and ``_.a.1`` is the first tone of a syllable with no onset.
Both parse back unambiguously, which the numeric-only form does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from echoscribe.errors import ConfigError, TokenizerError, require, require_in
from echoscribe.synth.lexicon import Lexicon, MandarinSyllable
from echoscribe.text.vocab import Vocabulary

#: Separates the three parts of a tone-language unit symbol.
UNIT_SEPARATOR = "."

#: Stands in for a syllable with no initial consonant.
ZERO_INITIAL_MARK = "_"

#: Display diacritics for the four lexical tones, in tone order.
TONE_MARKS = {1: "\u02c9", 2: "\u02ca", 3: "\u02c7", 4: "\u02cb"}

#: Granularities :func:`syllable_units` can emit.
UNIT_LEVELS = ("syllable", "phone")


@dataclass(frozen=True)
class ToneUnit:
    """One tone-language syllable as a modelling unit."""

    initial: str
    final: str
    tone: int

    def __post_init__(self) -> None:
        require(bool(self.final), "a tone unit needs a final", ConfigError)
        require_in("tone", self.tone, tuple(TONE_MARKS), ConfigError)
        require(
            UNIT_SEPARATOR not in self.initial and UNIT_SEPARATOR not in self.final,
            f"a unit part must not contain {UNIT_SEPARATOR!r}",
            ConfigError,
        )

    @property
    def symbol(self) -> str:
        """The parseable symbol, e.g. ``n.i.3`` or ``_.a.1``."""
        initial = self.initial or ZERO_INITIAL_MARK
        return UNIT_SEPARATOR.join((initial, self.final, str(self.tone)))

    @property
    def display(self) -> str:
        """The symbol with a tone diacritic instead of a digit, for reports."""
        return f"{self.initial}{self.final}{TONE_MARKS[self.tone]}"

    @classmethod
    def of(cls, syllable: MandarinSyllable) -> ToneUnit:
        """Build the unit for one lexicon syllable."""
        return cls(syllable.initial, syllable.final, syllable.tone)

    @classmethod
    def parse(cls, symbol: str) -> ToneUnit:
        """Inverse of :attr:`symbol`."""
        parts = symbol.split(UNIT_SEPARATOR)
        require(
            len(parts) == 3,
            f"{symbol!r} is not a unit symbol; expected initial{UNIT_SEPARATOR}final"
            f"{UNIT_SEPARATOR}tone",
            TokenizerError,
        )
        initial = "" if parts[0] == ZERO_INITIAL_MARK else parts[0]
        require(bool(parts[1]), f"{symbol!r} has an empty final", TokenizerError)
        require(
            parts[2].isdigit() and int(parts[2]) in TONE_MARKS,
            f"{symbol!r} has tone {parts[2]!r}, which is not one of {tuple(TONE_MARKS)}",
            TokenizerError,
        )
        return cls(initial, parts[1], int(parts[2]))


def syllable_units(syllable: MandarinSyllable, level: str = "syllable") -> tuple[str, ...]:
    """The modelling units of one tone-language syllable.

    ``"syllable"`` emits one toned unit; ``"phone"`` emits the initial (when there
    is one) and the toned final separately, which is the finer decomposition a
    phone-level acoustic model would use.
    """
    require_in("level", level, UNIT_LEVELS, ConfigError)
    unit = ToneUnit.of(syllable)
    if level == "syllable":
        return (unit.symbol,)
    final = ToneUnit("", syllable.final, syllable.tone).symbol
    return (syllable.initial, final) if syllable.initial else (final,)


def utterance_units(
    tokens: Sequence[str],
    languages: Sequence[str],
    lexicon: Lexicon | None = None,
    level: str = "syllable",
) -> tuple[str, ...]:
    """The modelling-unit sequence of a code-switched utterance.

    English words expand to their phone symbols and Chinese characters to their tone
    units, in transcript order.  The result is what a decoder is trained to emit and
    what an alignment maps back onto words.
    """
    require_in("level", level, UNIT_LEVELS, ConfigError)
    require(
        len(tokens) == len(languages),
        f"{len(tokens)} tokens cannot carry {len(languages)} language labels",
        ConfigError,
    )
    table = Lexicon() if lexicon is None else lexicon
    units: list[str] = []
    for token, language in zip(tokens, languages, strict=True):
        if language == "en":
            units.extend(table.english[token])
        else:
            units.extend(syllable_units(table.mandarin[token], level))
    return tuple(units)


def unit_vocabulary(lexicon: Lexicon | None = None, level: str = "syllable") -> Vocabulary:
    """The decoder output alphabet implied by a lexicon.

    Built from the lexicon tables rather than from a corpus, so the vocabulary
    covers every word the laboratory can render even if a particular corpus never
    uses one.  A corpus-trained subword vocabulary is a different thing and lives
    in the tokenizer modules.
    """
    require_in("level", level, UNIT_LEVELS, ConfigError)
    table = Lexicon() if lexicon is None else lexicon
    pieces: list[str] = []
    for spelling in table.english.values():
        pieces.extend(spelling)
    for syllable in table.mandarin.values():
        pieces.extend(syllable_units(syllable, level))
    return Vocabulary.build(pieces)
