"""The streaming decode loop.

:func:`StreamDecoder.push_logits` is the only entry point a caller needs during an
utterance: it appends encoder output, decodes every chunk that has become complete,
and returns the events that produced.  :func:`StreamDecoder.finish` decodes the tail
and closes the utterance; :func:`StreamDecoder.reset` throws everything away, which
is what a barge-in does.

The design decision worth stating is that a chunk is decoded only once *all* of its
frames have arrived.  Decoding a partial chunk would produce a hypothesis that has to
be retracted when the rest of the chunk lands, and retraction is exactly what a
user-visible transcript should not do.  The cost is one chunk of latency, which the
schedule already accounts for.

What the reported latency covers, and what it does not:

* it covers the chunk duration and the right context, because those are audio that
  has to exist before a frame can be decoded at all;
* it covers the frame length, because a frame is not complete until its last sample
  has arrived;
* it does *not* cover encoder compute, joint evaluation, network transport or the
  time a consumer spends rendering.  Those are real and they are usually larger.

So :attr:`~echoscribe.stream.chunks.ChunkSchedule.algorithmic_latency_ms` is a floor
on end-to-end latency, and the only part of it this package can guarantee.  Reporting
it as "the latency" without that caveat would overstate what has been measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.decode.chunked import ChunkConfig, ChunkResult, ChunkState, decode_chunk
from echoscribe.decode.tokens import DecodedToken
from echoscribe.stream.chunks import ChunkSchedule
from echoscribe.stream.endpoint import BlankRatioEndpointer
from echoscribe.stream.joint import AppendableJoint
from echoscribe.stream.protocol import Hypothesis, StreamEvent


@dataclass(frozen=True)
class StreamConfig:
    """How the runtime turns appended frames into events."""

    schedule: ChunkSchedule = ChunkSchedule()
    max_symbols_per_frame: int = 3
    blank_id: int = 0
    eos_id: int | None = None

    def chunk_config(self) -> ChunkConfig:
        """The decoder settings that match this stream configuration."""
        return ChunkConfig(
            frames_per_chunk=self.schedule.frames_per_chunk,
            max_symbols_per_frame=self.max_symbols_per_frame,
            blank_id=self.blank_id,
            eos_id=self.eos_id,
        )


class StreamDecoder:
    """Decodes a growing joint one chunk at a time."""

    def __init__(
        self,
        joint: AppendableJoint,
        config: StreamConfig | None = None,
        endpointer: BlankRatioEndpointer | None = None,
    ) -> None:
        self.joint = joint
        self.config = StreamConfig() if config is None else config
        self.endpointer = endpointer
        self._state = ChunkState()
        self._tokens: list[DecodedToken] = []
        self._closed = False

    @property
    def hypothesis(self) -> Hypothesis:
        """The current hypothesis, final or not."""
        return Hypothesis(
            symbols=tuple(token.symbol for token in self._tokens),
            score=self._state.score,
            frame=self._state.frame,
            is_final=self._closed,
        )

    @property
    def is_closed(self) -> bool:
        """True once :meth:`finish` has run and :meth:`reset` has not."""
        return self._closed

    def push_logits(self, logits: np.ndarray) -> tuple[StreamEvent, ...]:
        """Append encoder output and decode every chunk it completes."""
        appended = self.joint.append(logits)
        events = list(self._drain())
        events.extend(self._watch(appended))
        return tuple(events)

    def _watch(self, appended: int) -> list[StreamEvent]:
        """Feed newly appended frames to the endpointer, if there is one.

        The endpointer sees the *acoustic* posteriors rather than the biased ones a
        decoder might use, so a hotword list cannot make an utterance end sooner.
        """
        if self.endpointer is None or self._closed:
            return []
        events = []
        start = self.joint.num_frames - appended
        for frame in range(start, self.joint.num_frames):
            row = np.exp(self.joint(frame, ()))
            if self.endpointer.push_posteriors(row):
                self._closed = True
                events.append(StreamEvent("endpoint", frame + 1, self.hypothesis))
                break
        return events

    def finish(self) -> tuple[StreamEvent, ...]:
        """Decode the tail and close the utterance.

        Returns every event the close produced, which is usually just the final one
        but includes any whole chunk that had not been decoded yet.  The final event
        is always last, and its hypothesis is the one to transcribe.
        """
        events = list(self._drain())
        self._absorb(decode_chunk(self.joint, self._state, self._tail_config()))
        self._closed = True
        events.append(StreamEvent("final", self._state.frame, self.hypothesis))
        return tuple(events)

    def reset(self) -> StreamEvent:
        """Throw everything away, as after a barge-in."""
        self.joint.reset()
        self._state = ChunkState()
        self._tokens = []
        self._closed = False
        if self.endpointer is not None:
            self.endpointer.reset()
        return StreamEvent("reset", 0, self.hypothesis)

    def _tail_config(self) -> ChunkConfig:
        """A chunk config wide enough to consume whatever frames are left."""
        base = self.config.chunk_config()
        remaining = max(self.joint.num_frames - self._state.frame, 1)
        return ChunkConfig(
            frames_per_chunk=remaining,
            max_symbols_per_frame=base.max_symbols_per_frame,
            blank_id=base.blank_id,
            eos_id=base.eos_id,
        )

    def _drain(self) -> tuple[StreamEvent, ...]:
        """Decode whole chunks until the next one would be incomplete."""
        events: list[StreamEvent] = []
        size = self.config.schedule.frames_per_chunk
        while (
            not self._closed
            and not self._state.finished
            and self.joint.num_frames - self._state.frame >= size
        ):
            result = decode_chunk(self.joint, self._state, self.config.chunk_config())
            self._absorb(result)
            events.append(StreamEvent("partial", self._state.frame, self.hypothesis))
        return tuple(events)

    def _absorb(self, result: ChunkResult) -> None:
        """Fold one chunk's output into the running hypothesis."""
        self._tokens.extend(result.tokens)
        self._state = result.state
