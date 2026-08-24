"""``echoscribe synth``: generate and render a seeded synthetic corpus.

The work directory afterwards holds ``wav/`` with one file per utterance and
``manifest.jsonl`` describing them -- text, tokens, durations, checksums --
which is every input the later stages need and the reference transcripts the
scoring stage reads.
"""

from __future__ import annotations

import argparse

from echoscribe.cli.common import EXIT_OK, fail
from echoscribe.seeds import Seed
from echoscribe.synth.corpus import CorpusConfig, generate_corpus
from echoscribe.synth.manifest import manifest_digest
from echoscribe.synth.render import AcousticCondition, RenderConfig, render_corpus


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``synth`` to the top-level parser."""
    parser = subparsers.add_parser(
        "synth",
        help="render a seeded synthetic corpus into a work directory",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument(
        "--workdir", required=True, help="directory to fill with wav/ and manifest.jsonl"
    )
    parser.add_argument(
        "--utterances", type=int, default=24, help="how many utterances to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="root seed; the same seed renders the same corpus"
    )
    parser.add_argument("--min-tokens", type=int, default=1, help="fewest tokens per utterance")
    parser.add_argument("--max-tokens", type=int, default=7, help="most tokens per utterance")
    parser.add_argument(
        "--english-probability",
        type=float,
        default=0.5,
        help="chance each token is English rather than Chinese",
    )
    parser.add_argument(
        "--noise-kind",
        choices=("white", "pink", "modulated"),
        default=None,
        help="mix this noise into every rendered utterance",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="planted mix SNR in dB; needs --noise-kind",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Generate, render and summarize one corpus."""
    corpus = generate_corpus(
        CorpusConfig(
            utterances=args.utterances,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            english_probability=args.english_probability,
            seed=Seed(args.seed, "cli.synth"),
        )
    )
    if args.snr is not None and args.noise_kind is None:
        return fail("--snr needs --noise-kind; a clean corpus has no SNR to set")
    condition = (
        AcousticCondition()
        if args.noise_kind is None
        else AcousticCondition(
            noise_kind=args.noise_kind,
            noise_snr_db=args.snr if args.snr is not None else 20.0,
        )
    )
    records = render_corpus(corpus, RenderConfig(condition=condition), args.workdir)
    seconds = sum(record.duration for record in records)
    print(
        f"echoscribe synth: {len(records)} utterances, {seconds:.2f} s of audio, "
        f"manifest digest {manifest_digest(records)}"
    )
    print(f"  workdir: {args.workdir}")
    if args.noise_kind is not None:
        print(f"  condition: {args.noise_kind} noise at {condition.noise_snr_db:.1f} dB SNR")
    return EXIT_OK
