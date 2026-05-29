"""``echoscribe bench``: measure decoding real-time factor over a corpus.

RTF here means exactly what :mod:`echoscribe.eval.bench` says it means: wall
seconds this process spent decoding, over seconds of audio decoded, on this
machine, median of the measured repeats.  The audio duration is derived from
the fixture frame grid -- 10 ms per frame, the shift ``feats`` framed with --
so the denominator is a fact about the corpus rather than another
measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echoscribe.cli.common import EXIT_OK, write_json
from echoscribe.cli.scripted import load_scripted_corpus
from echoscribe.decode.ctc_greedy import ctc_greedy
from echoscribe.decode.ctc_prefix import ctc_prefix_search
from echoscribe.eval.bench import measure_rtf

FRAME_SHIFT_SECONDS = 0.01
METHODS = ("greedy", "prefix")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``bench`` to the top-level parser."""
    parser = subparsers.add_parser(
        "bench",
        help="measure decode real-time factor over a prepared corpus",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory prepared by synth and feats")
    parser.add_argument("--method", choices=METHODS, default="greedy", help="decoder to time")
    parser.add_argument("--beam", type=int, default=5, help="beam width for the prefix method")
    parser.add_argument("--warmups", type=int, default=1, help="untimed runs before measuring")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs to keep")
    parser.add_argument("--out", default="bench.json", help="result file to write")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Time the decoder over the whole corpus and write one bench result."""
    utterances, _ = load_scripted_corpus(args.workdir)
    audio_seconds = sum(item.num_frames for item in utterances) * FRAME_SHIFT_SECONDS
    if args.method == "greedy":

        def decode_all() -> object:
            return [ctc_greedy(item.posteriors).symbols for item in utterances]

    else:

        def decode_all() -> object:
            return [
                ctc_prefix_search(item.posteriors, beam_size=args.beam).symbols
                for item in utterances
            ]

    result = measure_rtf(
        f"ctc-{args.method}-decode",
        decode_all,
        audio_seconds,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    target = write_json(Path(args.workdir) / args.out, result.to_dict())
    print(
        f"echoscribe bench: RTF {result.rtf} (best {result.rtf_best}) decoding "
        f"{audio_seconds:.2f} s of audio with {result.repeats} repeats"
    )
    print(f"  result: {target}")
    print("  note: single-process CPU timing on this machine, not a hardware claim")
    return EXIT_OK
