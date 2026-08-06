"""Latency-budgeted chunk scheduling.

A streaming decoder consumes audio in chunks.  Three numbers decide what that costs:

``frames_per_chunk``
    how many frames are decoded at once.  Larger chunks amortise the per-call
    overhead and give the encoder more context, and they delay every hypothesis by
    the chunk duration.
``right_context_frames``
    how many *future* frames a chunk is allowed to look at.  This is the only part
    of the budget that requires audio the speaker has not finished producing yet, so
    it is the part a live system actually feels.
``frame_shift_ms`` / ``frame_length_ms``
    the analysis grid the frames sit on.

From those, the algorithmic latency of a chunk follows exactly, and
:attr:`ChunkSchedule.algorithmic_latency_ms` states it: the time from the start of
the chunk's first frame to the moment its last frame can be computed, which is

    ``(frames_per_chunk + right_context_frames - 1) * frame_shift_ms + frame_length_ms``

That number is a floor.  It excludes the encoder's own compute time, any network
hop and the decoder's search, all of which a real system adds on top.  What it does
give is a budget that can be checked: a schedule whose latency exceeds the product
requirement is wrong before a single sample is processed.
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import ConfigError, require, require_positive


@dataclass(frozen=True)
class ChunkPlan:
    """One chunk: which frames it decodes and when they become available."""

    index: int
    first_frame: int
    end_frame: int
    available_at_ms: float

    def __post_init__(self) -> None:
        require(
            0 <= self.first_frame <= self.end_frame,
            f"chunk {self.index} covers an invalid frame range "
            f"({self.first_frame}, {self.end_frame})",
            ConfigError,
        )

    @property
    def num_frames(self) -> int:
        """Frames this chunk decodes."""
        return self.end_frame - self.first_frame


@dataclass(frozen=True)
class ChunkSchedule:
    """A chunk size, a right context and the frame grid they sit on."""

    sample_rate: int = 16000
    frames_per_chunk: int = 10
    right_context_frames: int = 4
    frame_shift_ms: float = 10.0
    frame_length_ms: float = 25.0

    def __post_init__(self) -> None:
        require(
            isinstance(self.sample_rate, int)
            and not isinstance(self.sample_rate, bool)
            and self.sample_rate > 0,
            f"sample_rate must be a positive integer, got {self.sample_rate!r}",
            ConfigError,
        )
        require_positive("frames_per_chunk", self.frames_per_chunk, ConfigError)
        require(
            self.right_context_frames >= 0,
            f"right_context_frames must not be negative, got {self.right_context_frames}",
            ConfigError,
        )
        require_positive("frame_shift_ms", self.frame_shift_ms, ConfigError)
        require_positive("frame_length_ms", self.frame_length_ms, ConfigError)
        require(
            self.frame_length_ms >= self.frame_shift_ms,
            "a frame must be at least as long as the shift between frames",
            ConfigError,
        )

    @property
    def chunk_ms(self) -> float:
        """Audio duration one chunk decodes."""
        return self.frames_per_chunk * self.frame_shift_ms

    @property
    def right_context_ms(self) -> float:
        """Future audio one chunk is allowed to wait for."""
        return self.right_context_frames * self.frame_shift_ms

    @property
    def algorithmic_latency_ms(self) -> float:
        """Delay between a frame arriving and it being decodable."""
        span = self.frames_per_chunk + self.right_context_frames - 1
        return span * self.frame_shift_ms + self.frame_length_ms

    @property
    def frame_shift_samples(self) -> int:
        """Frame hop in samples at this schedule's sample rate."""
        return _samples_from_ms(self.frame_shift_ms, self.sample_rate)

    @property
    def frame_length_samples(self) -> int:
        """Frame length in samples at this schedule's sample rate."""
        return _samples_from_ms(self.frame_length_ms, self.sample_rate)

    def chunk_count(self, num_frames: int) -> int:
        """How many chunks ``num_frames`` needs."""
        require(num_frames >= 0, f"num_frames must not be negative, got {num_frames}", ConfigError)
        return -(-num_frames // self.frames_per_chunk)

    def available_at_ms(self, frame_index: int) -> float:
        """Audio time that must have arrived before ``frame_index`` can be decoded.

        ``frame_index`` is exclusive: it is the frame *after* the last one decoded,
        which is what a chunk plan reports.
        """
        require_positive("frame_index", frame_index, ConfigError)
        last = frame_index - 1 + self.right_context_frames
        return last * self.frame_shift_ms + self.frame_length_ms

    def samples_required(self, frame_index: int) -> int:
        """Sample count that must have arrived before ``frame_index`` is decodable."""
        require_positive("frame_index", frame_index, ConfigError)
        last = frame_index - 1 + self.right_context_frames
        return last * self.frame_shift_samples + self.frame_length_samples

    def plan(self, num_frames: int) -> tuple[ChunkPlan, ...]:
        """The chunk plan for an utterance of ``num_frames`` frames.

        The last chunk is short when the frame count is not a multiple of the chunk
        size; it still reports the right context it would like, so a caller can see
        that the tail of an utterance is decoded with less lookahead than the middle.
        """
        plans: list[ChunkPlan] = []
        first = 0
        while first < num_frames:
            end = min(first + self.frames_per_chunk, num_frames)
            plans.append(ChunkPlan(len(plans), first, end, self.available_at_ms(end)))
            first = end
        return tuple(plans)


def _samples_from_ms(milliseconds: float, sample_rate: int) -> int:
    """Whole samples covered by a duration in milliseconds, rounded half up."""
    return int((milliseconds * sample_rate + 500.0) // 1000.0)
