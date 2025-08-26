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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from echoscribe.decode.tokens import DecodedToken
from echoscribe.errors import DecodeError, require, require_positive
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


@dataclass(frozen=True)
class TransducerConfig:
    """How far a greedy transducer decode is allowed to run."""

    blank_id: int = 0
    max_symbols_per_frame: int = 10
    eos_id: int | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.max_symbols_per_frame, int)
            and not isinstance(self.max_symbols_per_frame, bool),
            f"max_symbols_per_frame must be an integer, got {self.max_symbols_per_frame!r}",
            DecodeError,
        )
        require_positive("max_symbols_per_frame", self.max_symbols_per_frame, DecodeError)
        require(
            isinstance(self.blank_id, int) and not isinstance(self.blank_id, bool),
            f"blank_id must be an integer, got {self.blank_id!r}",
            DecodeError,
        )


@dataclass(frozen=True)
class TransducerResult:
    """What a transducer decode produced, and when."""

    tokens: tuple[DecodedToken, ...]
    frames: tuple[int, ...]
    score: float
    num_frames: int
    frames_consumed: int = 0

    @property
    def stopped_early(self) -> bool:
        """True when decoding ended before the joint ran out of frames."""
        return self.frames_consumed < self.num_frames

    @property
    def symbols(self) -> tuple[int, ...]:
        """The emitted labels, in order."""
        return tuple(token.symbol for token in self.tokens)

    @property
    def emissions_per_frame(self) -> tuple[int, ...]:
        """How many labels each frame produced."""
        counts = [0] * self.num_frames
        for frame in self.frames:
            counts[frame] += 1
        return tuple(counts)


def transducer_greedy(joint: Joint, config: TransducerConfig | None = None) -> TransducerResult:
    """Decode greedily: emit the argmax label, or advance the frame on a blank.

    The inner loop is what makes a transducer a transducer.  At each state the
    decoder asks the joint for a distribution, takes its argmax, and either emits
    that label *and stays on the same frame* or consumes a blank and moves on.  A
    frame can therefore produce several labels, which CTC cannot do.

    ``max_symbols_per_frame`` bounds the inner loop.  Without it a joint that keeps
    preferring a label over the blank would spin forever on one frame, and a
    streaming decoder has to promise a latency bound regardless of what the model
    says.

    When ``config.eos_id`` is set, emitting that symbol ends the decode.  The symbol
    is scored but never emitted, and the frames the decoder never reached are
    reported through :attr:`TransducerResult.frames_consumed`, which is what a
    latency budget reads.
    """
    settings = TransducerConfig() if config is None else config
    require(
        0 <= settings.blank_id < joint.num_symbols,
        f"blank id {settings.blank_id} is outside the {joint.num_symbols}-symbol alphabet",
        DecodeError,
    )
    labels: list[int] = []
    frames: list[int] = []
    label_scores: list[float] = []
    total = 0.0
    frame = 0
    finished = False
    while frame < joint.num_frames and not finished:
        emitted = 0
        while True:
            logits = joint(frame, tuple(labels))
            best = int(np.argmax(logits))
            total += float(logits[best])
            if best == settings.blank_id:
                frame += 1
                break
            if settings.eos_id is not None and best == settings.eos_id:
                # The end-of-sequence symbol is consumed -- it contributes to the
                # path score -- but it is not a label, so it never reaches the
                # output.  Stopping here is what lets a stream end on the model's
                # own evidence rather than on a silence timer.
                finished = True
                break
            labels.append(best)
            frames.append(frame)
            label_scores.append(float(logits[best]))
            emitted += 1
            if emitted >= settings.max_symbols_per_frame:
                frame += 1
                break
    return TransducerResult(
        tokens=_spans(labels, frames, joint.num_frames, label_scores),
        frames=tuple(frames),
        score=total,
        num_frames=joint.num_frames,
        frames_consumed=frame,
    )


def _spans(
    labels: Sequence[int],
    frames: Sequence[int],
    num_frames: int,
    scores: Sequence[float],
) -> tuple[DecodedToken, ...]:
    """Turn emission frames into half-open spans with a score each.

    A token's span runs from the frame that produced it to the frame that produced
    the *next* one, so two labels emitted from the same frame give the first of them
    a zero-width span.  Zero width is the honest answer: the decoder has no evidence
    that the earlier label occupied any time at all.

    Per-token scores count labels only.  :attr:`TransducerResult.score` also counts
    the blanks that advanced time, because that is the log-probability of the whole
    path through the lattice.  The two answer different questions and neither
    replaces the other.
    """
    tokens: list[DecodedToken] = []
    for position, frame in enumerate(frames):
        end = frames[position + 1] if position + 1 < len(frames) else num_frames
        tokens.append(DecodedToken(labels[position], frame, max(end, frame), scores[position]))
    return tuple(tokens)
