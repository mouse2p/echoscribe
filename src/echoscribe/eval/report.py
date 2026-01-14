"""Deterministic SVG reports, written by hand rather than by a plotting library.

Every figure in this package is a string of SVG built from small ordered
pieces, so the same inputs produce the same bytes on every run.  A report can
be committed as a golden file, diffed in review and regenerated in CI without
a display server or a heavyweight dependency.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from echoscribe.errors import ConfigError, require

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
SVG_VERSION = "1.1"

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


AttributePair = tuple[str, str | int | float]

FORBIDDEN_NAME_CHARACTERS = " >/\"'"


@dataclass(frozen=True)
class SvgElement:
    """One node of an SVG tree: a tag, ordered attributes and children.

    Attributes keep the order they were given, because the render is a byte
    contract and mapping iteration order is an implementation detail a golden
    file should not depend on.  Children are nested elements or plain text,
    and text is escaped on the way out, never on the way in.
    """

    tag: str
    attributes: tuple[AttributePair, ...] = ()
    children: tuple[SvgElement | str, ...] = ()

    def __post_init__(self) -> None:
        require(
            isinstance(self.tag, str)
            and self.tag
            and not any(character in self.tag for character in FORBIDDEN_NAME_CHARACTERS),
            f"{self.tag!r} is not a usable SVG tag name",
            ConfigError,
        )
        names = [name for name, _ in self.attributes]
        require(
            len(set(names)) == len(names),
            f"duplicate SVG attribute in {names!r}",
            ConfigError,
        )
        for name, value in self.attributes:
            require(
                isinstance(name, str)
                and name
                and not any(character in name for character in FORBIDDEN_NAME_CHARACTERS),
                f"{name!r} is not a usable SVG attribute name",
                ConfigError,
            )
            require(
                isinstance(value, (str, int, float)) and not isinstance(value, bool),
                f"attribute {name!r} must be a string or a number, got {value!r}",
                ConfigError,
            )
        for child in self.children:
            require(
                isinstance(child, (SvgElement, str)),
                f"an SVG child must be an element or text, got {child!r}",
                ConfigError,
            )

    def render(self, indent: int = 0) -> str:
        """This element as SVG text, nested children indented by two spaces."""
        pad = "  " * indent
        rendered_attributes = "".join(
            f' {name}="{attribute_value(value)}"' for name, value in self.attributes
        )
        opening = f"<{self.tag}{rendered_attributes}"
        if not self.children:
            return f"{pad}{opening}/>"
        if all(isinstance(child, str) for child in self.children):
            text = "".join(escape_text(child) for child in self.children)
            return f"{pad}{opening}>{text}</{self.tag}>"
        child_pad = "  " * (indent + 1)
        parts = [
            child.render(indent + 1)
            if isinstance(child, SvgElement)
            else f"{child_pad}{escape_text(child)}"
            for child in self.children
        ]
        return f"{pad}{opening}>\n" + "\n".join(parts) + f"\n{pad}</{self.tag}>"


def attribute_value(value: str | int | float) -> str:
    """One attribute value as it appears inside the rendered quotes."""
    if isinstance(value, str):
        return escape_text(value)
    if isinstance(value, int):
        return str(value)
    return format_length(value)


def svg_document(
    width: int,
    height: int,
    children: Sequence[SvgElement | str] = (),
    title: str = "",
    description: str = "",
) -> str:
    """A complete SVG document: XML declaration, sized root, optional title.

    ``title`` and ``description`` become the first children of the root, which
    is where SVG expects them and what makes the file self-describing when it
    is opened in a browser or diffed by hand.
    """
    for name, value in (("width", width), ("height", height)):
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"the document {name} must be a positive integer, got {value!r}",
            ConfigError,
        )
    header: list[SvgElement] = []
    if title:
        header.append(SvgElement("title", (), (title,)))
    if description:
        header.append(SvgElement("desc", (), (description,)))
    root = SvgElement(
        "svg",
        (
            ("xmlns", SVG_NAMESPACE),
            ("version", SVG_VERSION),
            ("width", width),
            ("height", height),
            ("viewBox", f"0 0 {width} {height}"),
        ),
        (*header, *children),
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{root.render()}\n'


def digest_report(svg: str) -> str:
    """The SHA-256 of a document, the handle golden tests compare with.

    Digesting the exact bytes that would be written means a golden test fails
    on any change at all -- a reordered attribute, one coordinate off in the
    last decimal.  That is the point: report drift should be a decision, never
    an accident.
    """
    require(
        isinstance(svg, str),
        f"a report digest needs a string, got {type(svg).__name__}",
        ConfigError,
    )
    return hashlib.sha256(svg.encode("utf-8")).hexdigest()
