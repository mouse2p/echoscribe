"""The streaming runtime.

Everything in this package exists to answer one question a batch decoder never has
to: *how late is this hypothesis?*  A recognizer that sees the whole utterance can
be as accurate as it likes; one that has to answer while the speaker is still
talking trades accuracy for latency, and the trade has to be stated in milliseconds
rather than hoped for.

The modules split along that concern:

:mod:`~echoscribe.stream.chunks`
    turns a latency budget into a chunk plan;
:mod:`~echoscribe.stream.buffer`
    carries the samples that have not yet completed a frame across chunk boundaries;
:mod:`~echoscribe.stream.protocol`
    the partial/final hypothesis events a consumer sees;
:mod:`~echoscribe.stream.joint`
    a joint whose frame count grows as encoder output arrives;
:mod:`~echoscribe.stream.runtime`
    the decode loop that ties them together.
"""

from echoscribe.stream.buffer import FrameBuffer
from echoscribe.stream.chunks import ChunkPlan, ChunkSchedule
from echoscribe.stream.joint import AppendableJoint
from echoscribe.stream.protocol import EVENT_KINDS, Hypothesis, StreamEvent
from echoscribe.stream.runtime import StreamConfig, StreamDecoder

__all__ = [
    "EVENT_KINDS",
    "AppendableJoint",
    "ChunkPlan",
    "ChunkSchedule",
    "FrameBuffer",
    "Hypothesis",
    "StreamConfig",
    "StreamDecoder",
    "StreamEvent",
]
