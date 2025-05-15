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
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import TokenizerError, require

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
