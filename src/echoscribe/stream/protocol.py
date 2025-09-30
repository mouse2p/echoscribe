"""The partial/final hypothesis protocol.

A streaming consumer does not get one answer, it gets a sequence of them, and the
sequence has structure worth naming explicitly rather than leaving to convention:

``partial``
    the best hypothesis for the audio consumed so far.  A partial is *retractable*
    in principle -- a decoder that looks ahead can change its mind -- but the greedy
    search used here never does, so partials form a growing prefix chain.
``final``
    emitted once, when the stream is closed.  After a final there is nothing more to
    say about this utterance.
``endpoint``
    emitted when an endpointer decides the speaker has stopped, which can happen
    while audio is still arriving.
``reset``
    emitted after a barge-in or an explicit restart.  It carries an empty hypothesis
    and means everything before it is void.

Every event carries the frame index it was produced at, so a consumer can compute
how much audio it had seen rather than having to trust an arrival timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import StreamError, require, require_in

#: Kinds of event a stream can emit.
EVENT_KINDS = ("partial", "final", "endpoint", "reset")


@dataclass(frozen=True)
class Hypothesis:
    """What the decoder believes so far."""

    symbols: tuple[int, ...] = ()
    score: float = 0.0
    frame: int = 0
    is_final: bool = False

    def __post_init__(self) -> None:
        require(self.frame >= 0, f"frame must not be negative, got {self.frame}", StreamError)
        require(
            all(symbol >= 0 for symbol in self.symbols),
            f"a hypothesis cannot contain a negative symbol: {self.symbols}",
            StreamError,
        )

    @property
    def is_empty(self) -> bool:
        """True when nothing has been emitted yet."""
        return not self.symbols

    def extends(self, earlier: Hypothesis) -> bool:
        """True when this hypothesis starts with everything ``earlier`` had.

        The prefix property is what makes a partial safe to display: a consumer that
        rendered ``earlier`` can append to it rather than rewriting.
        """
        return self.symbols[: len(earlier.symbols)] == earlier.symbols


@dataclass(frozen=True)
class StreamEvent:
    """One thing a stream told its consumer."""

    kind: str
    frame: int
    hypothesis: Hypothesis

    def __post_init__(self) -> None:
        require_in("kind", self.kind, EVENT_KINDS, StreamError)
        require(self.frame >= 0, f"frame must not be negative, got {self.frame}", StreamError)

    @property
    def is_terminal(self) -> bool:
        """True for the events after which no further partial should be expected."""
        return self.kind in ("final", "reset")
