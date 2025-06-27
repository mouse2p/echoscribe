"""Script classification, which is what makes code-switch scoring possible.

A hybrid WER/CER score has to decide, per character, whether it belongs to a word
language or a character language.  That decision cannot be made by looking for
spaces -- Chinese text has none -- and cannot be made by a tokenizer that has
already merged things.  It is made here, from the Unicode block a character sits
in, before anything else touches the string.

Six classes are enough for an English-Chinese recognizer, and having an explicit
``other`` class is what stops an unexpected character from being silently scored as
Latin.
"""

from __future__ import annotations

import unicodedata

from echoscribe.errors import TokenizerError, require

#: Script classes :func:`script_of` can return.
SCRIPTS = ("han", "latin", "digit", "punctuation", "space", "other")

#: Unicode blocks treated as Han ideographs.
HAN_RANGES = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
)

#: Blocks treated as Latin, including the extended ranges European names need.
LATIN_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),  # Latin-1 Supplement letters through Latin Extended-B
    (0x1E00, 0x1EFF),  # Latin Extended Additional
)


def script_of(character: str) -> str:
    """Return the script class of a single character."""
    require(
        len(character) == 1,
        f"script_of classifies one character, got {len(character)}: {character!r}",
        TokenizerError,
    )
    code = ord(character)
    if character.isspace():
        return "space"
    if _in_ranges(code, HAN_RANGES):
        return "han"
    if _in_ranges(code, LATIN_RANGES):
        return "latin"
    if character.isdigit():
        return "digit"
    if unicodedata.category(character)[0] in ("P", "S"):
        return "punctuation"
    return "other"


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    """True when ``code`` falls inside one of the inclusive ranges."""
    return any(low <= code <= high for low, high in ranges)


def is_han(character: str) -> bool:
    """Convenience predicate the scoring code uses on every character."""
    return script_of(character) == "han"
