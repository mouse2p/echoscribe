"""A joint whose frame count grows as encoder output arrives.

The batch decoders take a joint with a fixed ``num_frames``.  A stream cannot offer
that: the encoder has only produced the frames the audio has reached.  This class is
the adapter -- it satisfies the same
:class:`~echoscribe.decode.transducer.Joint` protocol, and its frame count grows
every time encoder output is appended.

An optional ``predictor`` turns it into a transducer joint rather than a CTC one.  It
is called with the emitted-label history and returns an additive log-prior over the
alphabet, which is shallow fusion: the acoustic evidence and the label model are
combined by addition in the log domain and renormalised before the argmax.  With no
predictor the history is ignored and the joint is exactly a CTC posterior stream.

Rows are normalised once, when they are appended.  Doing it per lookup would repeat
the same work every time a decoder step revisits a frame, and would make the stored
rows unusable for anything else.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from echoscribe.errors import DecodeError, StreamError, require, require_positive
from echoscribe.numeric import log_softmax
from echoscribe.types import FEATURE_DTYPE

#: Adds a label-history prior to one frame of acoustic logits.
Predictor = Callable[[Sequence[int]], np.ndarray]


class AppendableJoint:
    """A growing joint that satisfies the batch decoders' protocol."""

    def __init__(self, num_symbols: int, predictor: Predictor | None = None) -> None:
        require_positive("num_symbols", num_symbols, StreamError)
        require(
            isinstance(num_symbols, int) and not isinstance(num_symbols, bool),
            f"num_symbols must be an integer, got {num_symbols!r}",
            StreamError,
        )
        self._num_symbols = int(num_symbols)
        self._predictor = predictor
        self._rows: list[np.ndarray] = []

    @property
    def num_frames(self) -> int:
        """Frames appended so far."""
        return len(self._rows)

    @property
    def num_symbols(self) -> int:
        """Alphabet size, blank included."""
        return self._num_symbols

    def append(self, logits: np.ndarray) -> int:
        """Normalise and store new frames; return the resulting frame count."""
        matrix = np.asarray(logits, dtype=FEATURE_DTYPE)
        require(
            matrix.ndim == 2,
            f"append needs a (frames, symbols) matrix, got shape {matrix.shape}",
            StreamError,
        )
        require(
            matrix.shape[1] == self._num_symbols,
            f"appended rows have {matrix.shape[1]} symbols but the joint has {self._num_symbols}",
            StreamError,
        )
        require(
            bool(np.all(np.isfinite(matrix))),
            "appended logits must be finite",
            StreamError,
        )
        self._rows.extend(log_softmax(matrix, axis=-1))
        return len(self._rows)

    def __call__(self, frame: int, history: tuple[int, ...]) -> np.ndarray:
        """Log-probabilities for one frame, biased by the predictor if there is one."""
        require(
            0 <= frame < len(self._rows),
            f"frame {frame} is outside the {len(self._rows)} frames appended so far",
            DecodeError,
        )
        row = self._rows[frame]
        if self._predictor is None:
            return row
        prior = np.asarray(self._predictor(tuple(history)), dtype=FEATURE_DTYPE)
        require(
            prior.shape == (self._num_symbols,),
            f"the predictor returned shape {prior.shape}, expected {self._num_symbols} symbols",
            DecodeError,
        )
        return log_softmax(row + prior)

    def reset(self) -> None:
        """Forget every frame, as after a barge-in."""
        self._rows.clear()
