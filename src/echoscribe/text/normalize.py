"""Unicode normalisation for transcripts and hypotheses.

A recognizer that compares raw strings will score two identical utterances as
different because one used a fullwidth comma, one capitalised a word, or one was
typed on a keyboard that emits a combining acute accent instead of a precomposed
``é``.  Normalisation removes those differences *before* anything is counted, so an
error rate measures word choice rather than keyboard layout.

The stage is deliberately conservative: it folds case, folds CJK punctuation to its
ASCII equivalent, collapses runs of whitespace and -- optionally -- strips Latin
accent marks.  It does not translate, stem, selectively lowercase proper nouns, or
drop punctuation that a scorer still needs to see.

Why it has to run before anything else: every later stage makes a decision that
normalisation would otherwise have made for it.  A segmenter splits on whitespace,
so an ideographic space that was never folded produces one token instead of two.  A
vocabulary lookup is exact, so a fullwidth digit misses its entry.  A scorer counts
tokens, so a difference in capitalisation becomes a substitution error that has
nothing to do with recognition.

The same configuration must be applied to the reference and to the hypothesis.
Scoring a normalised hypothesis against a raw reference measures the normaliser
rather than the recognizer, which is why :class:`NormalizeConfig` belongs in a
run's recorded metadata instead of being a per-call argument a scorer can forget.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from echoscribe.errors import ConfigError, require, require_in
from echoscribe.text.script import script_of

#: Unicode normalisation forms :func:`normalize_text` accepts.
NORMALIZATION_FORMS = ("NFC", "NFKC", "NFD", "NFKD")


@dataclass(frozen=True)
class NormalizeConfig:
    """Which normalisation stages to apply."""

    form: str = "NFKC"
    lower: bool = True
    fold_punctuation: bool = True
    collapse_spaces: bool = True
    strip_accents: bool = False

    def __post_init__(self) -> None:
        require_in("form", self.form, NORMALIZATION_FORMS, ConfigError)


def collapse_whitespace(text: str) -> str:
    """Reduce every run of whitespace to one space and trim both ends."""
    return " ".join(text.split())


def normalize_text(text: str, config: NormalizeConfig | None = None) -> str:
    """Apply the Unicode form and every enabled stage to ``text``.

    The form comes first because compatibility folding is what turns a fullwidth
    ``Ａ`` into an ``A`` and an ideographic space into a regular one; the later
    stages then see ordinary ASCII where a CJK keyboard put something else.
    """
    require(
        isinstance(text, str),
        f"normalize_text needs a string, got {type(text).__name__}",
        ConfigError,
    )
    settings = NormalizeConfig() if config is None else config
    result = unicodedata.normalize(settings.form, text)
    for name, stage in NORMALIZATION_STAGES:
        if getattr(settings, name):
            result = stage(result)
    return result


#: Punctuation that NFKC leaves alone but a scorer should treat as its ASCII
#: equivalent.  The ideographic comma and full stop have no compatibility
#: decomposition at all, so folding them is a decision this toolkit makes
#: explicitly rather than something Unicode does for it.
PUNCTUATION_FOLD = {
    "\u3001": ",",  # ideographic comma
    "\u3002": ".",  # ideographic full stop
    "\u00b7": " ",  # middle dot, used to separate syllables in transliteration
    "\u2014": "-",  # em dash
    "\u2013": "-",  # en dash
    "\u2026": "...",  # horizontal ellipsis
    "\u300c": '"',  # left corner bracket
    "\u300d": '"',  # right corner bracket
    "\u300e": '"',  # left white corner bracket
    "\u300f": '"',  # right white corner bracket
    "\uff5e": "~",  # fullwidth tilde, which NFKC maps to a different tilde
}


def fold_punctuation(text: str) -> str:
    """Replace the CJK and typographic punctuation that NFKC does not touch."""
    return "".join(PUNCTUATION_FOLD.get(character, character) for character in text)


#: Combining-mark category that a Latin accent decomposes into.  ``Mc`` and ``Me``
#: are deliberately absent: in most scripts those marks are letters in their own
#: right, not decoration.
COMBINING_CATEGORIES = ("Mn",)


def strip_accents(text: str, latin_only: bool = True) -> str:
    """Remove combining accent marks, by default only after a Latin base letter.

    The Latin-only restriction is the important part.  Decomposing and dropping
    every ``Mn`` character is the usual one-liner, and it silently destroys scripts
    where those marks are phonemic -- an Arabic fatha or a Devanagari nukta changes
    the pronunciation.  Restricting the removal to marks that follow a Latin base
    keeps ``café`` folding to ``cafe`` without touching anything else.
    """
    decomposed = unicodedata.normalize("NFD", text)
    kept = []
    previous = ""
    for character in decomposed:
        combining = unicodedata.category(character) in COMBINING_CATEGORIES
        if combining and (not latin_only or (previous and script_of(previous) == "latin")):
            continue
        kept.append(character)
        previous = character
    return unicodedata.normalize("NFC", "".join(kept))


#: The stages :func:`normalize_text` applies, in order, paired with the
#: :class:`NormalizeConfig` field that switches each one on.  Declaring the order
#: as data is what makes it testable: a stage that was silently reordered would
#: otherwise show up only as a wrong score.
NORMALIZATION_STAGES = (
    ("fold_punctuation", fold_punctuation),
    ("lower", str.lower),
    ("strip_accents", strip_accents),
    ("collapse_spaces", collapse_whitespace),
)
