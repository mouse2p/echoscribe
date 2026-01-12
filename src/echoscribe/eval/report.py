"""Deterministic SVG reports, written by hand rather than by a plotting library.

Every figure in this package is a string of SVG built from small ordered
pieces, so the same inputs produce the same bytes on every run.  A report can
be committed as a golden file, diffed in review and regenerated in CI without
a display server or a heavyweight dependency.
"""

from __future__ import annotations

import math

from echoscribe.errors import ConfigError, require

ESCAPES: tuple[tuple[str, str], ...] = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&quot;"),
    ("'", "&apos;"),
)


def escape_text(text: str) -> str:
    """Escape the five characters XML reserves, in the one order that is safe.

    ``&`` must be replaced first; every other replacement introduces new ``&``
    characters that a later pass would otherwise escape a second time.  The
    function is deliberately *not* idempotent -- escaping already-escaped text
    is a bug in the caller, and ``&amp;`` turning into ``&amp;amp;`` is what
    correctly reporting that bug looks like.
    """
    require(
        isinstance(text, str),
        f"escape_text needs a string, got {type(text).__name__}",
        ConfigError,
    )
    escaped = text
    for raw, entity in ESCAPES:
        escaped = escaped.replace(raw, entity)
    return escaped


def format_length(value: float) -> str:
    """Render an SVG coordinate: two decimals at most, no trailing zeros.

    Coordinates pass through ``round`` before they are printed, so a value
    that differs only in bits beyond the second decimal -- the kind of
    difference floating-point arithmetic produces between platforms -- still
    renders identically everywhere.  A tiny negative that rounds to zero
    prints as ``0``, never ``-0``.
    """
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"an SVG length must be a number, got {value!r}",
        ConfigError,
    )
    number = float(value)
    require(math.isfinite(number), f"an SVG length must be finite, got {value!r}", ConfigError)
    rounded = round(number, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")
