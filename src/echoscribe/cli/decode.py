"""``echoscribe decode``: CTC decoding of a prepared corpus.

Two methods share one door.  ``greedy`` takes the per-frame argmax, collapses
repeats and drops blanks -- the baseline a streaming system actually runs.
``prefix`` keeps the beam_size most probable collapsed sequences instead,
which costs more and buys an n-best list; on the unambiguous fixtures the two
must agree, and a test says so.

What the agreement is worth: on a fixture whose argmax path is planted,
greedy and prefix *have* to coincide, so a disagreement is a bug in one of
the two searches rather than a modelling question.  On a trained model's
posteriors the same check says nothing -- there the methods differ by design
-- which is why the invariant lives in the tests over fixtures and not in
the command's output.

Each row also carries surface text.  A row whose hypothesis matches the
planted units carries the reference text forward: the same unit sequence
renders to the same surface text under the lexicon, so this is a
reconstruction, not a peek at the answer.  A row that does *not* match has no
reliable surface form -- modelling units do not invert to words -- so its
text is the unit sequence joined with spaces, and a text-level score reads
that as the error it is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echoscribe.cli.common import EXIT_OK, write_jsonl
from echoscribe.cli.scripted import load_scripted_corpus
from echoscribe.decode.ctc_greedy import ctc_greedy
from echoscribe.decode.ctc_prefix import ctc_prefix_search

METHODS = ("greedy", "prefix")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``decode`` to the top-level parser."""
    parser = subparsers.add_parser(
        "decode",
        help="CTC-decode a prepared corpus of scripted posteriors",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory prepared by synth and feats")
    parser.add_argument("--method", choices=METHODS, default="greedy", help="decoding method")
    parser.add_argument("--beam", type=int, default=5, help="beam width for the prefix method")
    parser.add_argument("--out", default="decode.jsonl", help="hypothesis file to write")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Decode every utterance and write one JSONL line per hypothesis."""
    utterances, vocabulary = load_scripted_corpus(args.workdir)
    rows: list[dict] = []
    exact = 0
    for item in utterances:
        if args.method == "greedy":
            symbols = ctc_greedy(item.posteriors).symbols
        else:
            symbols = ctc_prefix_search(item.posteriors, beam_size=args.beam).symbols
        matched = symbols == item.unit_ids
        exact += int(matched)
        rows.append(
            {
                "utterance_id": item.utterance_id,
                "method": args.method,
                "reference_units": list(item.units),
                "hypothesis_units": [vocabulary.token_at(symbol) for symbol in symbols],
                "hypothesis_ids": list(symbols),
                "exact_match": matched,
                "reference_text": item.text,
                "hypothesis_text": item.text
                if matched
                else " ".join(vocabulary.token_at(symbol) for symbol in symbols),
            }
        )
    target = write_jsonl(Path(args.workdir) / args.out, rows)
    print(
        f"echoscribe decode ({args.method}): {len(rows)} utterances, "
        f"{exact} replayed their planted transcript exactly"
    )
    print(f"  hypotheses: {target}")
    print("  note: scripted fixture posteriors, not the output of a trained model")
    return EXIT_OK
