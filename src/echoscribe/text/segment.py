"""Segmentation: turning a normalised string into the tokens a scorer counts.

The rule is script-dependent, and that is the whole reason this module exists.  For
a word language the unit that carries meaning is the whitespace-delimited word, so
``hello world`` is two tokens.  For a character language there is no whitespace and
no cheap word segmentation that is not itself a model, so the unit is the
character: ``你好`` is two tokens.  A hybrid score has to count both in the same
utterance, which means the boundary decision cannot be deferred to the scorer.

Two details matter more than they look:

* an apostrophe or hyphen *inside* a Latin run belongs to the word, so ``don't`` is
  one token and ``code-switch`` is one token unless splitting is asked for;
* punctuation is dropped by default.  A transcript's commas are not recognition
  errors, and counting them would make the score depend on the transcriber's
  punctuation policy rather than on the words.
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import ConfigError, require

#: Scripts whose runs are emitted as whole words.
WORD_SCRIPTS = ("latin", "digit")

#: Punctuation that continues a word rather than ending it.
INTERNAL_WORD_PUNCTUATION = ("-", "'")


@dataclass(frozen=True)
class SegmentConfig:
    """How aggressively to split a normalised string."""

    keep_punctuation: bool = False
    split_on_hyphen: bool = False
    min_token_length: int = 1

    def __post_init__(self) -> None:
        require(
            isinstance(self.min_token_length, int)
            and not isinstance(self.min_token_length, bool)
            and self.min_token_length >= 1,
            f"min_token_length must be an integer >= 1, got {self.min_token_length!r}",
            ConfigError,
        )


@dataclass(frozen=True)
class Token:
    """One scoreable unit, with where it came from."""

    text: str
    script: str
    start: int

    def __post_init__(self) -> None:
        require(bool(self.text), "a token must not be empty", ConfigError)
        require(self.start >= 0, f"token start must not be negative, got {self.start}", ConfigError)

    @property
    def stop(self) -> int:
        """Exclusive end offset in the source string."""
        return self.start + len(self.text)

    @property
    def is_word(self) -> bool:
        """True when the token is a whole word rather than a single character."""
        return self.script in WORD_SCRIPTS
