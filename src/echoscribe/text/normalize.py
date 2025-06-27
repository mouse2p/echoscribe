"""Unicode normalisation for transcripts and hypotheses.

A recognizer that compares raw strings will score two identical utterances as
different because one used a fullwidth comma, one capitalised a word, or one was
typed on a keyboard that emits a combining acute accent instead of a precomposed
``é``.  Normalisation removes those differences *before* anything is counted, so an
error rate measures word choice rather than keyboard layout.

The stage is deliberately conservative: it folds case, collapses runs of whitespace
and -- with the stages added below -- folds CJK punctuation and strips Latin
accents.  It does not translate, stem, selectively lowercase proper nouns, or drop
punctuation that a scorer still needs to see.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from echoscribe.errors import ConfigError, require, require_in

#: Unicode normalisation forms :func:`normalize_text` accepts.
NORMALIZATION_FORMS = ("NFC", "NFKC", "NFD", "NFKD")


@dataclass(frozen=True)
class NormalizeConfig:
    """Which normalisation stages to apply."""

    form: str = "NFKC"
    lower: bool = True
    collapse_spaces: bool = True

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
    if settings.lower:
        result = result.lower()
    if settings.collapse_spaces:
        result = collapse_whitespace(result)
    return result
