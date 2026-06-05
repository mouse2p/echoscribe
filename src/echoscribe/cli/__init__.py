"""The ``echoscribe`` command line.

One contract for every subcommand: parse its own arguments, do one coherent
job against files under a work directory, print a short human summary to
stdout, and return an exit code -- 0 when the job succeeded, 1 when a job
failed on valid usage, 2 when the usage itself was wrong.  Library errors
surface as one line on stderr instead of a traceback, because a traceback
answers "what broke in the code" when the user asked "what do I do now".

The work directory is the contract *between* subcommands: ``synth`` fills it
with ``wav/`` and ``manifest.jsonl``, ``feats`` adds ``features.npz``, and
the decoding, scoring and reporting stages read their inputs from exactly
those paths.  Nothing is hidden in the user's home directory, every stage
can be rerun on its own, and deleting the directory deletes the run.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from echoscribe import __version__
from echoscribe.cli import (
    bench,
    bias,
    decode,
    evaluate,
    feats,
    report,
    stream,
    synth,
)
from echoscribe.cli.common import EXIT_FAILURE, EXIT_USAGE
from echoscribe.errors import EchoscribeError


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser, subcommands included."""
    parser = argparse.ArgumentParser(
        prog="echoscribe",
        description="Offline workbench for streaming speech-recognition mechanics.",
    )
    parser.add_argument("--version", action="version", version=f"echoscribe {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    bench.register(subparsers)
    bias.register(subparsers)
    decode.register(subparsers)
    evaluate.register(subparsers)
    feats.register(subparsers)
    report.register(subparsers)
    stream.register(subparsers)
    synth.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI invocation and return its exit code.

    ``argv=None`` reads ``sys.argv``; every test in the suite passes an
    explicit list through this same door, which is the console script's.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return handler(args)
    except EchoscribeError as error:
        print(f"echoscribe: error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    except OSError as error:
        print(f"echoscribe: error: {error}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
