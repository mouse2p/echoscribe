"""``echoscribe bias``: what a hotword bonus changes, on an ambiguous joint.

Biasing is only visible where the unbiased decode could go either way, so
this command builds exactly that situation as a scripted transducer joint:
one frame where two symbols sit closer together than the bonus, then blanks.
It runs the decode twice -- with and without the hotword -- and reports both,
because a biased hypothesis with nothing to compare it to is not a result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from echoscribe.cli.common import EXIT_OK, fail, write_jsonl
from echoscribe.cli.scripted import load_scripted_corpus
from echoscribe.decode.bias import BiasConfig, bias_greedy
from echoscribe.decode.transducer import TableJoint, TransducerConfig
from echoscribe.errors import ConfigError, TokenizerError, require
from echoscribe.text.vocab import Vocabulary

FIXTURE_FRAMES = 6


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``bias`` to the top-level parser."""
    parser = subparsers.add_parser(
        "bias",
        help="show what a hotword bonus changes on an ambiguous scripted joint",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory prepared by synth and feats")
    parser.add_argument("--hotword", required=True, help="a modelling-unit token to bias toward")
    parser.add_argument("--bonus", type=float, default=2.0, help="log-score bonus for the hotword")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="how much the unbiased joint prefers its rival symbol",
    )
    parser.add_argument("--out", default="bias.jsonl", help="report file to write")
    parser.set_defaults(handler=run)


def ambiguous_joint(vocabulary: Vocabulary, primary: int, rival: int, margin: float) -> TableJoint:
    """A joint whose first frame prefers ``primary`` over ``rival`` by ``margin``."""
    require(margin > 0.0, f"the margin must be positive, got {margin!r}", ConfigError)
    table = np.zeros((FIXTURE_FRAMES, 2, len(vocabulary)))
    table[:, :, 0] = 5.0  # blanks dominate everywhere...
    table[0, 0, 0] = -5.0  # ...except on the first frame before any label
    table[0, 0, primary] = margin
    table[0, 0, rival] = 0.0
    return TableJoint(logits=table)


def run(args: argparse.Namespace) -> int:
    """Decode the fixture with and without the hotword and report the diff."""
    utterances, vocabulary = load_scripted_corpus(args.workdir)
    try:
        hotword_id = vocabulary.id_of(args.hotword)
    except TokenizerError:
        return fail(f"{args.hotword!r} is not in the modelling-unit vocabulary")
    candidates = [symbol for item in utterances for symbol in item.unit_ids if symbol != hotword_id]
    if not candidates:
        return fail("the corpus uses no unit other than the hotword; nothing to prefer")
    joint = ambiguous_joint(vocabulary, candidates[0], hotword_id, args.margin)
    report = bias_greedy(
        joint,
        [[hotword_id]],
        TransducerConfig(),
        BiasConfig(bonus=args.bonus),
    )
    row = {
        "hotword": args.hotword,
        "hotword_id": hotword_id,
        "rival_id": candidates[0],
        "bonus": args.bonus,
        "margin": args.margin,
        "unbiased_ids": list(report.unbiased_symbols),
        "biased_ids": list(report.biased_symbols),
        "flipped": report.changed,
        "hotwords_matched": report.hotwords_matched(),
    }
    target = write_jsonl(Path(args.workdir) / args.out, [row])
    verdict = "flipped the decode" if report.changed else "left the decode unchanged"
    print(
        f"echoscribe bias: hotword {args.hotword!r} at bonus {args.bonus} over "
        f"margin {args.margin} {verdict}"
    )
    print(f"  report: {target}")
    print("  note: the joint is a scripted two-symbol fixture, not a trained model")
    return EXIT_OK
