"""``echoscribe demo``: the whole offline loop, once, with its numbers.

Synthesize a corpus, extract its features, decode it two ways (batch CTC and
chunked streaming), score both granularities, time the decoder, and render
one SVG report -- then print the numbers those stages actually measured.
Every figure comes from a file this run wrote into the work directory; the
command invents nothing and rounds nothing twice.

The default path needs no torch: decoding runs on scripted fixture
posteriors built from the planted transcripts.  What the demo demonstrates
is therefore the *pipeline* -- every stage handing the next one artifacts it
can read -- and the printed error rates certify the mechanics against known
answers rather than measuring a recognizer.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from echoscribe.cli.bench import run as run_bench
from echoscribe.cli.common import EXIT_OK, fail, read_json
from echoscribe.cli.decode import run as run_decode
from echoscribe.cli.evaluate import run as run_eval
from echoscribe.cli.feats import run as run_feats
from echoscribe.cli.report import run as run_report
from echoscribe.cli.stream import run as run_stream
from echoscribe.cli.synth import run as run_synth
from echoscribe.eval.bench import BenchResult
from echoscribe.eval.report import digest_report


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``demo`` to the top-level parser."""
    parser = subparsers.add_parser(
        "demo",
        help="run the full offline loop and print what it measured",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory for every stage's files")
    parser.add_argument("--utterances", type=int, default=8, help="corpus size to synthesize")
    parser.add_argument("--seed", type=int, default=2025, help="corpus seed")
    parser.add_argument("--repeats", type=int, default=3, help="timed decode repeats")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Run every stage in order, then summarize the files they wrote."""
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stages = [
        (
            "synth",
            run_synth,
            Namespace(
                workdir=str(workdir),
                utterances=args.utterances,
                seed=args.seed,
                min_tokens=1,
                max_tokens=3,
                english_probability=0.5,
            ),
        ),
        (
            "feats",
            run_feats,
            Namespace(
                workdir=str(workdir),
                num_mel_bins=40,
                deltas=False,
                delta_order=2,
                out="features.npz",
            ),
        ),
        (
            "decode",
            run_decode,
            Namespace(workdir=str(workdir), method="greedy", beam=5, out="decode.jsonl"),
        ),
        (
            "stream",
            run_stream,
            Namespace(
                workdir=str(workdir), frames_per_chunk=16, max_symbols=1, out="session.jsonl"
            ),
        ),
        (
            "eval-units",
            run_eval,
            Namespace(
                workdir=str(workdir),
                level="units",
                metric="wer",
                decode_file="decode.jsonl",
                out="scores-units.json",
            ),
        ),
        (
            "eval-text",
            run_eval,
            Namespace(
                workdir=str(workdir),
                level="text",
                metric="hybrid",
                decode_file="decode.jsonl",
                out="scores-text.json",
            ),
        ),
        (
            "bench",
            run_bench,
            Namespace(
                workdir=str(workdir),
                method="greedy",
                beam=5,
                warmups=1,
                repeats=args.repeats,
                out="bench.json",
            ),
        ),
        (
            "report",
            run_report,
            Namespace(workdir=str(workdir), label="demo", out="report.svg"),
        ),
    ]
    for name, handler, namespace in stages:
        code = handler(namespace)
        if code != EXIT_OK:
            return fail(f"the demo failed inside its {name} stage")
    units = read_json(workdir / "scores-units.json")
    text_scores = read_json(workdir / "scores-text.json")
    bench_payload = read_json(workdir / "bench.json")
    if not isinstance(units, dict) or not isinstance(text_scores, dict):
        return fail("the demo's own score files are unreadable")
    if not isinstance(bench_payload, dict):
        return fail("the demo's own bench file is unreadable")
    bench = BenchResult.from_dict(bench_payload)
    document = (workdir / "report.svg").read_text(encoding="utf-8", newline="\n")
    print()
    print("demo summary -- every number below was measured in this run")
    print(
        f"  unit error rate:   {units['totals']['error_rate']:.4f} "
        f"over {units['totals']['reference_length']} units"
    )
    print(
        f"  hybrid text score: {text_scores['totals']['error_rate']:.4f} "
        f"over {text_scores['totals']['reference_length']} mixed tokens"
    )
    print(f"  decode RTF:        {bench.rtf} (median of {bench.repeats} repeats)")
    print(f"  report digest:     {digest_report(document)}")
    print()
    print("The audio is synthetic and the transcripts are planted. These numbers")
    print("validate the mechanics -- features, decoding, streaming, scoring,")
    print("reporting -- against known answers; they are not recognizer accuracy.")
    return EXIT_OK
