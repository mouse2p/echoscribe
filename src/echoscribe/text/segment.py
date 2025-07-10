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
from echoscribe.text.script import script_of

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


def segment(text: str, config: SegmentConfig | None = None) -> tuple[Token, ...]:
    """Split ``text`` into the tokens a scorer should count.

    The scan is character-by-character rather than run-by-run because the decision
    at a punctuation mark depends on what is on *both* sides of it: ``don't`` keeps
    its apostrophe, ``hello, world`` does not keep its comma.
    """
    require(
        isinstance(text, str), f"segment needs a string, got {type(text).__name__}", ConfigError
    )
    settings = SegmentConfig() if config is None else config
    tokens: list[Token] = []
    index = 0
    length = len(text)
    while index < length:
        script = script_of(text[index])
        if script == "space":
            index += 1
            continue
        if script not in WORD_SCRIPTS:
            if script == "punctuation" and not settings.keep_punctuation:
                index += 1
                continue
            tokens.append(Token(text[index], script, index))
            index += 1
            continue
        end = _word_end(text, index, script, settings)
        tokens.append(Token(text[index:end], script, index))
        index = end
    return tuple(token for token in tokens if len(token.text) >= settings.min_token_length)


def _word_end(text: str, start: int, script: str, settings: SegmentConfig) -> int:
    """Exclusive end of the word starting at ``start``."""
    length = len(text)
    end = start + 1
    while end < length:
        following = script_of(text[end])
        if following == script:
            end += 1
            continue
        if _continues_a_word(text, end, following, script, settings):
            end += 1
            continue
        break
    return end


def _continues_a_word(
    text: str, position: int, script: str, word_script: str, settings: SegmentConfig
) -> bool:
    """True when the punctuation at ``position`` is internal to a word."""
    if settings.split_on_hyphen:
        return False
    if script != "punctuation" or text[position] not in INTERNAL_WORD_PUNCTUATION:
        return False
    following = position + 1
    return following < len(text) and script_of(text[following]) == word_script
