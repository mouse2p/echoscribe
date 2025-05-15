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
