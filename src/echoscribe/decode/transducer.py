"""RNN-T (transducer) decoding.

A transducer differs from CTC in one structural way: it decides *per frame* whether
to emit a label and stay, or to emit nothing and advance.  CTC fixes the alignment
length to the frame count and recovers labels by collapsing; a transducer can emit
several labels from one frame or none from many, which is what makes it usable
label-synchronously on a stream.

The decoder here talks to an abstract joint rather than to a network:

    ``joint(frame, history) -> log-probabilities over the alphabet``

where ``history`` is the tuple of labels emitted so far and index ``blank_id`` means
"emit nothing, advance the frame".  Everything below only needs that call, so the
same greedy search drives a torch model and a hand-written table.

:data:`TableJoint` is the hand-written one.  It is a *fixture*: its logits come from
an array indexed by ``(frame, number of labels emitted so far)``, so its predictor
depends only on how many labels have been produced, not on which.  That is not a
model of a real predictor -- a real one conditions on the label identity, which is
what makes it a language model -- but it is enough to exercise the search, the cache
and the latency accounting exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from echoscribe.errors import DecodeError, require
from echoscribe.numeric import log_softmax
from echoscribe.types import FEATURE_DTYPE


@runtime_checkable
class Joint(Protocol):
    """What a transducer decoder needs from an acoustic-plus-predictor model."""

    @property
    def num_frames(self) -> int:
        """Frames of encoder output available."""

    @property
    def num_symbols(self) -> int:
        """Size of the output alphabet, blank included."""

    def __call__(self, frame: int, history: tuple[int, ...]) -> np.ndarray:
        """Log-probabilities over the alphabet at one ``(frame, history)`` state."""


@dataclass(frozen=True)
class TableJoint:
    """A scripted joint reading logits from a ``(frames, labels + 1, symbols)`` table.

    The second axis is the number of labels emitted so far, so a table built with
    ``labels == 4`` can drive any decode that emits at most four symbols.  Asking for
    a state the table does not cover is an error rather than a silent clamp: a
    decoder that ran off the end of a fixture should say so.
    """

    logits: np.ndarray
    blank_id: int = 0

    def __post_init__(self) -> None:
        values = np.asarray(self.logits, dtype=FEATURE_DTYPE)
        require(
            values.ndim == 3,
            f"a joint table must be (frames, labels + 1, symbols), got shape {values.shape}",
            DecodeError,
        )
        require(
            bool(np.all(np.isfinite(values))),
            "a joint table must hold only finite logits",
            DecodeError,
        )
        require(
            values.shape[2] >= 2,
            f"an alphabet of {values.shape[2]} symbols cannot hold a blank and a label",
            DecodeError,
        )
        require(
            0 <= self.blank_id < values.shape[2],
            f"blank id {self.blank_id} is outside the {values.shape[2]}-symbol alphabet",
            DecodeError,
        )
        object.__setattr__(self, "logits", values)

    @property
    def num_frames(self) -> int:
        """Frames the table covers."""
        return int(self.logits.shape[0])

    @property
    def num_symbols(self) -> int:
        """Alphabet size, blank included."""
        return int(self.logits.shape[2])

    @property
    def max_labels(self) -> int:
        """How many labels a decode may emit before running out of predictor states."""
        return int(self.logits.shape[1]) - 1

    def __call__(self, frame: int, history: tuple[int, ...]) -> np.ndarray:
        """Log-probabilities for one state, normalised over the alphabet."""
        require(
            0 <= frame < self.num_frames,
            f"frame {frame} is outside the {self.num_frames}-frame table",
            DecodeError,
        )
        require(
            len(history) <= self.max_labels,
            f"the table covers {self.max_labels} emitted labels, not {len(history)}",
            DecodeError,
        )
        return log_softmax(self.logits[frame, len(history)], axis=-1)
