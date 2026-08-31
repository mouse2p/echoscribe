"""``echoscribe stream``: chunked transducer decoding of a prepared corpus.

The stream runtime is transducer-shaped: it consumes per-frame logits and
emits symbols without collapsing repeats, so this command drives it with the
*sparse* fixture path -- one frame per unit, blanks between -- and a
per-frame symbol cap of one.  The cap is the honest setting for a fixture
whose logits do not depend on the emitted history: with a cap above one the
same frame would keep arguing for the same symbol, and a history-blind
joint has no way to say "already emitted".  Raise the cap only against a
joint that actually conditions on history.

The session file this writes is the library's own format, readable with
:func:`~echoscribe.stream.session.read_session`, so anything a test can ask
of a session, a user can ask of this file.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from echoscribe.cli.common import EXIT_OK
from echoscribe.cli.scripted import load_scripted_corpus, sparse_path
from echoscribe.decode.posteriors import scripted_posteriors
from echoscribe.stream.chunks import ChunkSchedule
from echoscribe.stream.joint import AppendableJoint
from echoscribe.stream.runtime import StreamConfig, StreamDecoder
from echoscribe.stream.session import Session, UtteranceRecord, write_session


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``stream`` to the top-level parser."""
    parser = subparsers.add_parser(
        "stream",
        help="stream-decode a prepared corpus chunk by chunk",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory prepared by synth and feats")
    parser.add_argument(
        "--frames-per-chunk", type=int, default=16, help="frames the scheduler releases at once"
    )
    parser.add_argument(
        "--max-symbols", type=int, default=1, help="per-frame symbol cap; see the module note"
    )
    parser.add_argument("--out", default="session.jsonl", help="session file to write")
    parser.set_defaults(handler=run)


def session_id_for(identifiers: list[str]) -> str:
    """A deterministic session id: no clocks, no randomness, no collisions to hide."""
    digest = hashlib.sha256(",".join(identifiers).encode("utf-8")).hexdigest()[:12]
    return f"cli-stream-{digest}"


def run(args: argparse.Namespace) -> int:
    """Stream every utterance through a fresh decoder and write one session."""
    utterances, vocabulary = load_scripted_corpus(args.workdir)
    schedule = ChunkSchedule(frames_per_chunk=args.frames_per_chunk)
    print(
        f"echoscribe stream: chunks of {schedule.chunk_ms:.0f} ms, "
        f"right context {schedule.right_context_ms:.0f} ms, "
        f"cap {args.max_symbols} symbol per frame"
    )
    records: list[UtteranceRecord] = []
    exact = 0
    for item in utterances:
        logits = scripted_posteriors(
            sparse_path(item.unit_ids, item.num_frames), num_symbols=item.num_symbols
        )
        joint = AppendableJoint(item.num_symbols)
        decoder = StreamDecoder(
            joint,
            StreamConfig(
                schedule=schedule,
                max_symbols_per_frame=args.max_symbols,
            ),
        )
        events = 0
        for start in range(0, item.num_frames, args.frames_per_chunk):
            events += len(decoder.push_logits(logits[start : start + args.frames_per_chunk]))
        events += len(decoder.finish())
        hypothesis = decoder.hypothesis
        matched = hypothesis.symbols == item.unit_ids
        exact += int(matched)
        records.append(
            UtteranceRecord(
                utterance_id=item.utterance_id,
                hypothesis=hypothesis.symbols,
                reference=item.text,
                text=" ".join(vocabulary.token_at(symbol) for symbol in hypothesis.symbols),
                frames=item.num_frames,
                events=events,
                endpoint_frames=None,
                score=round(float(hypothesis.score), 12),
            )
        )
    session = Session(
        session_id=session_id_for([item.utterance_id for item in utterances]),
        metadata={
            "command": "stream",
            "frames_per_chunk": args.frames_per_chunk,
            "max_symbols_per_frame": args.max_symbols,
            "posteriors": "scripted fixtures",
        },
        utterances=tuple(records),
    )
    target = Path(args.workdir) / args.out
    write_session(target, session)
    print(
        f"echoscribe stream: {len(records)} utterances, "
        f"{exact} replayed their planted transcript exactly"
    )
    print(f"  session: {target}")
    print("  note: scripted fixture logits, not the output of a trained model")
    return EXIT_OK
