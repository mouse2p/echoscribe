#!/usr/bin/env python3
"""Stream one scripted utterance through the chunked runtime, then replay it.

Two runs over the same fixture logits:

1. a full stream decode, recorded chunk by chunk and *replayed* through a
   fresh decoder -- the event traces must be byte-identical, which is the
   determinism promise the streaming runtime makes;
2. a second decode with a blank-ratio endpointer attached, which closes the
   utterance inside the trailing silence instead of running to the end.

The logits are scripted fixtures with a planted transcript; the example
exercises scheduling, endpointing and replay, not recognition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echoscribe.decode.posteriors import scripted_posteriors
from echoscribe.stream.chunks import ChunkSchedule
from echoscribe.stream.endpoint import BlankRatioConfig, BlankRatioEndpointer
from echoscribe.stream.joint import AppendableJoint
from echoscribe.stream.replay import record, replay, trace_digest
from echoscribe.stream.runtime import StreamConfig, StreamDecoder
from echoscribe.stream.session import Session, UtteranceRecord, latency_report, write_session

UNITS = (3, 1, 4, 1, 5)
NUM_SYMBOLS = 6
LEAD_BLANKS = 25
UNIT_FRAMES = 60
TAIL_BLANKS = 45
CHUNK_FRAMES = 16
FRAME_SHIFT_MS = 10.0


def sparse_path(units: tuple[int, ...], frames: int, blank: int = 0) -> list[int]:
    """One frame per unit, blanks elsewhere."""
    gap = frames // len(units)
    path = [blank] * frames
    for index, unit in enumerate(units):
        path[index * gap] = unit
    return path


def blocks(count: int, size: int) -> list[int]:
    """Chunk sizes covering ``count`` frames."""
    sizes = [size] * (count // size)
    if count % size:
        sizes.append(count % size)
    return sizes


def main() -> int:
    """Stream, replay, endpoint, and write the session."""
    parser = argparse.ArgumentParser(description="streaming session with replay")
    parser.add_argument("--workdir", required=True, help="directory for the session file")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    path = [0] * LEAD_BLANKS + sparse_path(UNITS, UNIT_FRAMES) + [0] * TAIL_BLANKS
    logits = scripted_posteriors(path, num_symbols=NUM_SYMBOLS)
    total_frames = len(path)
    sizes = blocks(total_frames, CHUNK_FRAMES)
    budget_ms = CHUNK_FRAMES * FRAME_SHIFT_MS
    print(f"stream: {total_frames} frames in {len(sizes)} chunks of <= {CHUNK_FRAMES} frames")
    print(
        f"latency budget: one chunk of look-ahead = {budget_ms:.0f} ms "
        f"at {FRAME_SHIFT_MS:.0f} ms shift"
    )

    # Run 1: plain streaming decode, recorded and replayed.
    joint = AppendableJoint(NUM_SYMBOLS)
    config = StreamConfig(
        schedule=ChunkSchedule(frames_per_chunk=CHUNK_FRAMES), max_symbols_per_frame=1
    )
    decoder = StreamDecoder(joint, config)
    events = []
    cursor = 0
    for size in sizes:
        events.extend(decoder.push_logits(logits[cursor : cursor + size]))
        cursor += size
    events.extend(decoder.finish())
    hypothesis = decoder.hypothesis
    print(f"live hypothesis:     {hypothesis.symbols}")
    assert hypothesis.symbols == UNITS, "the stream must replay the planted units"

    recording = record(logits, sizes, CHUNK_FRAMES, max_symbols_per_frame=1)
    replayed = replay(recording)
    live_digest, replayed_digest = trace_digest(events), trace_digest(replayed)
    print(f"live trace digest:    {live_digest}")
    print(f"replayed trace digest: {replayed_digest}")
    assert live_digest == replayed_digest, "replay must reproduce the trace exactly"
    print("the replayed stream produced an identical event trace")

    # Run 2: the same stream with a blank-ratio endpointer attached.
    endpointer = BlankRatioEndpointer(
        BlankRatioConfig(window_frames=20, ratio_threshold=0.9, min_speech_frames=3)
    )
    joint_b = AppendableJoint(NUM_SYMBOLS)
    decoder_b = StreamDecoder(joint_b, config, endpointer)
    endpoint_frame = None
    pushed = 0
    events_b = []
    for size in sizes:
        if decoder_b.is_closed:
            break
        events_b.extend(decoder_b.push_logits(logits[pushed : pushed + size]))
        pushed += size
    if not decoder_b.is_closed:
        events_b.extend(decoder_b.finish())
        pushed = total_frames
    for event in events_b:
        if event.kind == "endpoint":
            endpoint_frame = event.frame
    hypothesis_b = decoder_b.hypothesis
    print(f"endpointed at frame: {endpoint_frame} of {pushed} pushed")
    assert endpoint_frame is not None, "the trailing silence must fire the endpointer"
    assert hypothesis_b.symbols == UNITS, "endpointing must not truncate the units"

    session = Session(
        session_id="example-streaming-session",
        metadata={
            "example": "streaming_session",
            "chunk_frames": CHUNK_FRAMES,
            "planted_units": list(UNITS),
            "posteriors": "scripted fixtures",
        },
        utterances=(
            UtteranceRecord(
                utterance_id="scripted-000",
                hypothesis=hypothesis_b.symbols,
                reference=" ".join(str(unit) for unit in UNITS),
                text=" ".join(str(unit) for unit in hypothesis_b.symbols),
                frames=pushed,
                events=len(events_b),
                endpoint_frames=endpoint_frame,
                score=round(float(hypothesis_b.score), 12),
            ),
        ),
    )
    target = workdir / "session.jsonl"
    write_session(target, session)
    print(f"latency report: {latency_report(session, FRAME_SHIFT_MS)}")
    print(f"session written to {target}")
    print("note: scripted fixture logits with a planted transcript, not recognition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
