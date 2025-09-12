"""Label-synchronous chunked decoding.

A streaming decoder cannot wait for the whole utterance, so it consumes encoder
output a chunk at a time and has to be able to stop, hand back a partial hypothesis,
and pick up exactly where it left off.  "Label-synchronous" names the loop structure
that makes this possible: the outer loop is over *frames* and the inner loop is over
*labels emitted from the current frame*, so the decoder never needs to know in
advance how many labels the utterance will contain.

:func:`decode_chunk` is the unit of work.  It advances at most ``frames_per_chunk``
frames and returns the tokens it produced plus a :class:`ChunkState` carrying the
emitted-label history.  That history is the predictor's input, so losing it would
make the next chunk's logits wrong rather than merely less accurate.

:func:`decode_all_chunks` runs the same loop to the end of the audio and is the
offline equivalent of :func:`echoscribe.decode.transducer.transducer_greedy`.  That
the two agree is the property the tests pin, and it is why the chunked path can be
trusted to give the same answer as the batch one on a whole utterance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.decode.tokens import DecodedToken
from echoscribe.decode.transducer import Joint, TransducerResult, emission_spans
from echoscribe.errors import DecodeError, require, require_positive


@dataclass(frozen=True)
class ChunkConfig:
    """How much audio one decoding step is allowed to consume."""

    frames_per_chunk: int = 8
    max_symbols_per_frame: int = 3
    blank_id: int = 0
    eos_id: int | None = None

    def __post_init__(self) -> None:
        require_positive("frames_per_chunk", self.frames_per_chunk, DecodeError)
        require_positive("max_symbols_per_frame", self.max_symbols_per_frame, DecodeError)
        require(
            self.blank_id >= 0,
            f"blank_id must not be negative, got {self.blank_id}",
            DecodeError,
        )


@dataclass(frozen=True)
class ChunkState:
    """Everything a resumed decode needs: history, position, score and status."""

    labels: tuple[int, ...] = ()
    frame: int = 0
    score: float = 0.0
    finished: bool = False


@dataclass(frozen=True)
class ChunkResult:
    """One chunk's output plus the state to resume from."""

    tokens: tuple[DecodedToken, ...]
    state: ChunkState
    frames_consumed: int

    @property
    def symbols(self) -> tuple[int, ...]:
        """The labels this chunk emitted."""
        return tuple(token.symbol for token in self.tokens)


def decode_chunk(
    joint: Joint,
    state: ChunkState | None = None,
    config: ChunkConfig | None = None,
) -> ChunkResult:
    """Decode one chunk, starting from ``state`` and stopping at the chunk boundary.

    A chunk ends when any of these happens: the frame budget is used up, the joint
    runs out of frames, or the end-of-sequence symbol is emitted.  The returned
    state records which, so a caller can tell "paused" from "finished" without
    looking at the audio.
    """
    settings = ChunkConfig() if config is None else config
    current = ChunkState() if state is None else state
    require(
        0 <= current.frame <= joint.num_frames,
        f"chunk state frame {current.frame} is outside [0, {joint.num_frames}]",
        DecodeError,
    )
    require(
        0 <= settings.blank_id < joint.num_symbols,
        f"blank id {settings.blank_id} is outside the {joint.num_symbols}-symbol alphabet",
        DecodeError,
    )
    if current.finished:
        return ChunkResult((), current, 0)
    stop = min(current.frame + settings.frames_per_chunk, joint.num_frames)
    labels = list(current.labels)
    emitted_labels: list[int] = []
    emitted_frames: list[int] = []
    scores: list[float] = []
    total = current.score
    frame = current.frame
    finished = False
    while frame < stop and not finished:
        emitted = 0
        while True:
            logits = joint(frame, tuple(labels))
            best = int(np.argmax(logits))
            total += float(logits[best])
            if best == settings.blank_id:
                frame += 1
                break
            if settings.eos_id is not None and best == settings.eos_id:
                finished = True
                break
            labels.append(best)
            emitted_labels.append(best)
            emitted_frames.append(frame)
            scores.append(float(logits[best]))
            emitted += 1
            if emitted >= settings.max_symbols_per_frame:
                frame += 1
                break
    # Spans inside a chunk end at the chunk boundary: the next chunk's first token
    # starts there, so the two halves of the timeline meet without overlapping.
    boundary = max(frame, current.frame)
    return ChunkResult(
        emission_spans(emitted_labels, emitted_frames, boundary, scores),
        ChunkState(tuple(labels), boundary, total, finished),
        boundary - current.frame,
    )


def decode_all_chunks(joint: Joint, config: ChunkConfig | None = None) -> TransducerResult:
    """Run chunked decoding to the end and report it like a batch decode."""
    settings = ChunkConfig() if config is None else config
    state = ChunkState()
    tokens: list[DecodedToken] = []
    while state.frame < joint.num_frames and not state.finished:
        result = decode_chunk(joint, state, settings)
        tokens.extend(result.tokens)
        state = result.state
    return TransducerResult(
        tokens=tuple(tokens),
        frames=tuple(token.start for token in tokens),
        score=state.score,
        num_frames=joint.num_frames,
        frames_consumed=state.frame,
    )
