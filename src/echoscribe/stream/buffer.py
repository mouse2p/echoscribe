"""A frame buffer for streaming feature extraction.

Framing needs ``frame_length`` samples to produce one frame, but audio arrives in
chunks whose size has nothing to do with that length.  Something has to remember the
samples that arrived but could not yet complete a frame, and that something is this
buffer.

The contract worth testing is the obvious one and the one most often broken: feeding
a waveform in arbitrary chunks has to produce exactly the frames that feeding it all
at once produces, in the same order.  Anything else means the features depend on the
network packet size, which is not a property a recognizer should have.

The buffer requires ``snip_edges=True``.  The padded mode centres frame zero on
sample zero, which needs samples from *before* the start of the utterance -- the one
thing a stream can never provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from echoscribe.errors import StreamError, require
from echoscribe.framing import FrameConfig, extract_frames
from echoscribe.types import FEATURE_DTYPE, FloatArray


@dataclass
class FrameBuffer:
    """Pending samples plus the count of frames already emitted."""

    config: FrameConfig
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=FEATURE_DTYPE))
    frames_emitted: int = 0

    def __post_init__(self) -> None:
        require(
            self.config.snip_edges,
            "a streaming frame buffer needs snip_edges=True; the padded mode centres "
            "frame zero on sample zero, which a stream cannot provide",
            StreamError,
        )
        require(
            self.frames_emitted >= 0,
            f"frames_emitted must not be negative, got {self.frames_emitted}",
            StreamError,
        )
        self.pending = np.asarray(self.pending, dtype=FEATURE_DTYPE)

    @property
    def pending_samples(self) -> int:
        """Samples held but not yet framed."""
        return int(self.pending.size)

    def push(self, samples: np.ndarray) -> FloatArray:
        """Add samples and return every frame they completed."""
        incoming = np.asarray(samples, dtype=FEATURE_DTYPE)
        require(
            incoming.ndim == 1,
            f"push needs 1-D samples, got shape {incoming.shape}",
            StreamError,
        )
        buffer = np.concatenate([self.pending, incoming]) if incoming.size else self.pending
        length, shift = self.config.frame_length, self.config.frame_shift
        count = (buffer.size - length) // shift + 1 if buffer.size >= length else 0
        frames = extract_frames(buffer, self.config) if count else np.zeros((0, length))
        # The next un-emitted frame starts at count * shift, so everything from
        # there on is still pending: dropping the whole consumed window would throw
        # away the length - shift samples of overlap the next frame needs.
        self.pending = buffer[count * shift :]
        self.frames_emitted += int(frames.shape[0])
        return frames

    def flush(self) -> FloatArray:
        """Zero-pad the tail and return the frames it completes.

        A stream ends with fewer than ``frame_length`` samples pending more often than
        not, and those samples carry the last phone of the last word.  Padding them
        recovers one more frame; a caller that would rather drop the tail simply does
        not call this.
        """
        if self.pending.size == 0:
            return np.zeros((0, self.config.frame_length), dtype=FEATURE_DTYPE)
        shortfall = self.config.frame_length - self.pending.size
        padded = np.concatenate([self.pending, np.zeros(max(shortfall, 0))])
        frames = extract_frames(padded, self.config)
        self.frames_emitted += int(frames.shape[0])
        self.pending = np.zeros(0, dtype=FEATURE_DTYPE)
        return frames

    def reset(self) -> None:
        """Drop pending samples and the frame count, as after a barge-in."""
        self.pending = np.zeros(0, dtype=FEATURE_DTYPE)
        self.frames_emitted = 0
