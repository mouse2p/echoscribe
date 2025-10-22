"""Deterministic replay of a recorded stream.

A streaming decoder is the hardest thing in this package to debug, because the
interesting failures depend on *when* frames arrived and not only on what they
contained.  Reproducing one from a bug report means reproducing the chunking, and
that is what a recording is for.

:class:`StreamRecording` stores the appended logit blocks in order, together with the
configuration the stream ran under.  :func:`replay` feeds them back through a fresh
decoder, and :func:`trace_digest` reduces the resulting events to one hash -- so
"the replay matched" is a single comparable string rather than a diff of nested
tuples.

The guarantee is exact, not approximate: the logits are stored as raw float64 bytes
rather than as decimal text, so nothing is lost to formatting, and the decoder is
deterministic by construction.  Two runs over the same recording that disagree are a
bug in the decoder, never a difference in the data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from echoscribe.errors import StreamError, require, require_positive
from echoscribe.stream.protocol import StreamEvent
from echoscribe.types import FEATURE_DTYPE

#: Bumped when the serialized shape of a recording changes.
RECORDING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StreamRecording:
    """The logit blocks a stream was fed, plus the settings it ran under."""

    logits: np.ndarray
    block_sizes: tuple[int, ...]
    num_symbols: int
    chunk_frames: int
    max_symbols_per_frame: int = 3

    def __post_init__(self) -> None:
        values = np.asarray(self.logits, dtype=FEATURE_DTYPE)
        require(
            values.ndim == 2,
            f"a recording stores a (frames, symbols) matrix, got shape {values.shape}",
            StreamError,
        )
        require(
            values.shape[1] == self.num_symbols,
            f"the recording has {values.shape[1]} symbols per frame but declares "
            f"{self.num_symbols}",
            StreamError,
        )
        require(
            sum(self.block_sizes) == values.shape[0],
            f"the block sizes add to {sum(self.block_sizes)} but the recording holds "
            f"{values.shape[0]} frames",
            StreamError,
        )
        require_positive("chunk_frames", self.chunk_frames, StreamError)
        object.__setattr__(self, "logits", values)

    @property
    def num_frames(self) -> int:
        """Frames the recording covers."""
        return int(self.logits.shape[0])

    def blocks(self) -> Iterator[np.ndarray]:
        """Yield the appended blocks in the order they originally arrived."""
        cursor = 0
        for size in self.block_sizes:
            yield self.logits[cursor : cursor + size]
            cursor += size

    def digest(self) -> str:
        """A short hex digest of the recording's bytes and settings."""
        payload = hashlib.sha256()
        payload.update(np.ascontiguousarray(self.logits).tobytes())
        payload.update(repr(self.block_sizes).encode())
        payload.update(f"{self.num_symbols}|{self.chunk_frames}".encode())
        return payload.hexdigest()[:16]


def trace_digest(events: Sequence[StreamEvent]) -> str:
    """One hash over an event trace, so two runs can be compared directly."""
    payload = hashlib.sha256()
    for event in events:
        payload.update(event.kind.encode())
        payload.update(b"|")
        payload.update(str(event.frame).encode())
        payload.update(b"|")
        payload.update(",".join(str(symbol) for symbol in event.hypothesis.symbols).encode())
        payload.update(b"\n")
    return payload.hexdigest()
