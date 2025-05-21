"""The lexicon: how a surface token becomes a phone sequence.

Two vocabularies live here because the evaluation suite is code-switch aware and
the renderer has to handle both in one utterance:

``ENGLISH_SPELLINGS``
    a hand-written list of common English words spelled in the ARPAbet-style
    symbols of :data:`echoscribe.synth.phones.ENGLISH_PHONES`.  These are
    broad transcriptions chosen for a synthesizer with no ``ch``, ``jh``, ``dh``
    or ``zh``, so a few words are spelled more plainly than a dictionary would.
``MANDARIN_SYLLABLES``
    single characters mapped to an initial, a final and a lexical tone.  Finals
    are restricted to the fifteen in the synthesizer's table, so syllables whose
    real nucleus is a diphthong or a nasal coda outside that set are approximated.

Both lists are small on purpose.  The laboratory needs enough vocabulary for a
decoder to have to choose, and every entry has to be checked against the phone
inventory -- it does not need ten thousand words nobody will render.

Two approximations are worth naming, because they are the reason a transcript can
be exact while the audio is not a faithful pronunciation:

* the English phone set has no affricates (``ch``, ``jh``) and no ``dh``, so words
  that would need them are simply absent from the vocabulary rather than misspelt;
* the Mandarin final set has fifteen entries, so ``shuo`` renders on ``o``,
  ``xiao`` on ``ao`` and ``hui`` on ``ei``.

A recognizer evaluated here is therefore learning a slightly idealised mapping.
That is acceptable -- the transcript is still the exact ground truth for the audio
that was generated -- but it is another reason no number from this package should
be read as an accuracy claim about real speech.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from echoscribe.errors import TokenizerError, require
from echoscribe.synth.phones import FINALS, INITIALS, Phone, english_phone

#: Broad phone spellings for the English-style vocabulary.
ENGLISH_SPELLINGS: dict[str, tuple[str, ...]] = {
    "hello": ("hh", "ah", "l", "ow"),
    "world": ("w", "er", "l", "d"),
    "time": ("t", "ay", "m"),
    "meeting": ("m", "iy", "t", "ih", "ng"),
    "please": ("p", "l", "iy", "z"),
    "thank": ("th", "ae", "ng", "k"),
    "you": ("y", "uw"),
    "good": ("g", "uh", "d"),
    "morning": ("m", "ao", "r", "n", "ih", "ng"),
    "today": ("t", "ah", "d", "ay"),
    "system": ("s", "ih", "s", "t", "ah", "m"),
    "network": ("n", "eh", "t", "w", "er", "k"),
    "model": ("m", "aa", "d", "l"),
    "data": ("d", "aa", "t", "ah"),
    "file": ("f", "ay", "l"),
    "open": ("ow", "p", "ah", "n"),
    "start": ("s", "t", "aa", "r", "t"),
    "stop": ("s", "t", "aa", "p"),
    "listen": ("l", "ih", "s", "ah", "n"),
    "sound": ("s", "aw", "n", "d"),
    "signal": ("s", "ih", "g", "n", "ah", "l"),
    "filter": ("f", "ih", "l", "t", "er"),
    "sample": ("s", "ae", "m", "p", "l"),
    "test": ("t", "eh", "s", "t"),
    "code": ("k", "ow", "d"),
    "record": ("r", "ah", "k", "ao", "r", "d"),
    "input": ("ih", "n", "p", "uh", "t"),
    "output": ("aw", "t", "p", "uh", "t"),
    "level": ("l", "eh", "v", "l"),
    "value": ("v", "ae", "l", "y", "uw"),
    "number": ("n", "ah", "m", "b", "er"),
    "report": ("r", "ih", "p", "ao", "r", "t"),
    "show": ("sh", "ow"),
    "read": ("r", "iy", "d"),
    "write": ("r", "ay", "t"),
    "learn": ("l", "er", "n"),
    "machine": ("m", "ah", "sh", "iy", "n"),
    "window": ("w", "ih", "n", "d", "ow"),
    "answer": ("ae", "n", "s", "er"),
    "better": ("b", "eh", "t", "er"),
    "working": ("w", "er", "k", "ih", "ng"),
    "service": ("s", "er", "v", "ih", "s"),
    "session": ("s", "eh", "sh", "ah", "n"),
    "stream": ("s", "t", "r", "iy", "m"),
    "buffer": ("b", "ah", "f", "er"),
}


def english_vocabulary() -> tuple[str, ...]:
    """Every English word the laboratory can render, in table order."""
    return tuple(ENGLISH_SPELLINGS)


@dataclass(frozen=True)
class MandarinSyllable:
    """One character's pronunciation: an optional initial, a final and a tone."""

    character: str
    initial: str
    final: str
    tone: int

    def __post_init__(self) -> None:
        require(bool(self.character), "a syllable needs a character", TokenizerError)
        require(
            1 <= self.tone <= 4,
            f"{self.character}: tone must be 1-4, got {self.tone}",
            TokenizerError,
        )

    @property
    def romanization(self) -> str:
        """The syllable spelled as ``initial + final + tone``, e.g. ``ni3``."""
        return f"{self.initial}{self.final}{self.tone}"


#: Character-keyed pronunciations.  Finals are restricted to the synthesizer's
#: table, so ``shuo1`` is rendered on ``o`` rather than ``uo`` and ``xiao3`` on
#: ``ao`` rather than ``iao``.  Every syllable carries one of the four lexical
#: tones; a character that is neutral in running speech is given its citation
#: tone so the four-way contrast stays present in the corpus.
MANDARIN_SYLLABLES: dict[str, MandarinSyllable] = {
    syllable.character: syllable
    for syllable in (
        MandarinSyllable("你", "n", "i", 3),
        MandarinSyllable("好", "h", "ao", 3),
        MandarinSyllable("他", "t", "a", 1),
        MandarinSyllable("她", "t", "a", 1),
        MandarinSyllable("们", "m", "en", 2),
        MandarinSyllable("这", "zh", "e", 4),
        MandarinSyllable("那", "n", "a", 4),
        MandarinSyllable("看", "k", "an", 4),
        MandarinSyllable("来", "l", "ai", 2),
        MandarinSyllable("去", "q", "v", 4),
        MandarinSyllable("说", "sh", "o", 1),
        MandarinSyllable("会", "h", "ei", 4),
        MandarinSyllable("多", "d", "o", 1),
        MandarinSyllable("大", "d", "a", 4),
        MandarinSyllable("小", "x", "ao", 3),
        MandarinSyllable("中", "zh", "ong", 1),
        MandarinSyllable("人", "r", "en", 2),
        MandarinSyllable("时", "sh", "i", 2),
        MandarinSyllable("可", "k", "e", 3),
        MandarinSyllable("很", "h", "en", 3),
        MandarinSyllable("不", "b", "u", 4),
        MandarinSyllable("在", "z", "ai", 4),
        MandarinSyllable("声", "sh", "eng", 1),
        MandarinSyllable("话", "h", "a", 4),
    )
}


def mandarin_vocabulary() -> tuple[str, ...]:
    """Every character the laboratory can render, in table order."""
    return tuple(MANDARIN_SYLLABLES)


#: Script labels the lexicon assigns to a token.
LANGUAGES = ("en", "zh")


@dataclass(frozen=True)
class Lexicon:
    """Token lookup for both vocabularies behind one interface.

    The defaults go through ``default_factory`` rather than naming the tables
    directly.  ``dataclasses`` rejects an unhashable default, and copying also
    stops one caller's edits from leaking into every other ``Lexicon`` in the
    process.
    """

    english: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(ENGLISH_SPELLINGS))
    mandarin: dict[str, MandarinSyllable] = field(default_factory=lambda: dict(MANDARIN_SYLLABLES))

    def classify(self, token: str) -> str:
        """Return ``"en"`` or ``"zh"`` for a token, based on which table holds it."""
        if token in self.english:
            return "en"
        if token in self.mandarin:
            return "zh"
        raise TokenizerError(
            f"unknown token {token!r}; the lexicon holds {len(self.english)} English words "
            f"and {len(self.mandarin)} characters"
        )

    def word_phones(self, word: str) -> tuple[Phone, ...]:
        """Phones of one English word."""
        require(
            word in self.english, f"not an English word in this lexicon: {word!r}", TokenizerError
        )
        return tuple(english_phone(symbol) for symbol in self.english[word])

    def syllable_phones(self, character: str) -> tuple[Phone, ...]:
        """Phones of one Mandarin syllable: its initial, if any, then its final."""
        require(
            character in self.mandarin,
            f"not a character in this lexicon: {character!r}",
            TokenizerError,
        )
        syllable = self.mandarin[character]
        onset = (INITIALS[syllable.initial],) if syllable.initial else ()
        return onset + (FINALS[syllable.final],)

    def phones_for(self, token: str) -> tuple[Phone, ...]:
        """Phones of one token, whichever vocabulary it belongs to."""
        if self.classify(token) == "en":
            return self.word_phones(token)
        return self.syllable_phones(token)


@dataclass(frozen=True)
class TokenSpan:
    """Where one surface token's phones sit in a flattened phone sequence."""

    token: str
    language: str
    start: int
    length: int

    @property
    def stop(self) -> int:
        """Exclusive end index of the span."""
        return self.start + self.length


@dataclass(frozen=True)
class PhoneSequence:
    """A flattened phone stream plus the spans that produced it.

    Keeping the spans next to the phones is what lets a decoder's token alignment
    be turned back into a word alignment later: without them the boundary between
    ``hello`` and ``world`` is unrecoverable, because the phones are concatenated
    with no separator.
    """

    phones: tuple[Phone, ...]
    spans: tuple[TokenSpan, ...]

    def symbols(self) -> tuple[str, ...]:
        """The phone symbols, which is the unit sequence a decoder consumes."""
        return tuple(phone.symbol for phone in self.phones)

    def tokens(self) -> tuple[str, ...]:
        """The surface tokens, in order."""
        return tuple(span.token for span in self.spans)

    def phones_of(self, token: str) -> tuple[Phone, ...]:
        """Phones belonging to one token of the sequence."""
        for span in self.spans:
            if span.token == token:
                return self.phones[span.start : span.stop]
        raise TokenizerError(f"{token!r} is not part of this sequence")


def transcribe(tokens: Sequence[str], lexicon: Lexicon | None = None) -> PhoneSequence:
    """Flatten a token sequence into phones, recording each token's span."""
    table = Lexicon() if lexicon is None else lexicon
    require(len(tokens) > 0, "cannot transcribe an empty token sequence", TokenizerError)
    phones: list[Phone] = []
    spans: list[TokenSpan] = []
    for token in tokens:
        language = table.classify(token)
        resolved = table.phones_for(token)
        spans.append(TokenSpan(token, language, len(phones), len(resolved)))
        phones.extend(resolved)
    return PhoneSequence(tuple(phones), tuple(spans))


@dataclass(frozen=True)
class Coverage:
    """How much of a token list this lexicon can actually render."""

    total: int
    known: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def ratio(self) -> float:
        """Fraction of tokens that are known; 1.0 for an empty list."""
        return len(self.known) / self.total if self.total else 1.0

    @property
    def complete(self) -> bool:
        """True when nothing has to be dropped or replaced."""
        return not self.unknown


def coverage_of(tokens: Sequence[str], lexicon: Lexicon | None = None) -> Coverage:
    """Split ``tokens`` into those the lexicon knows and those it does not.

    A corpus generator calls this before rendering so that an out-of-vocabulary
    word is reported once, up front, rather than surfacing as an exception halfway
    through writing a manifest.
    """
    table = Lexicon() if lexicon is None else lexicon
    known: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        try:
            table.classify(token)
        except TokenizerError:
            unknown.append(token)
        else:
            known.append(token)
    return Coverage(len(tokens), tuple(known), tuple(unknown))
