"""``echoscribe eval``: score decode output against the planted transcripts.

Two granularities, because a code-switched corpus has two.  ``--level units``
scores the modelling-unit sequences the decoder actually emits; ``--level
text`` scores surface text with plain ``wer``, plain ``cer``, or the
``hybrid`` metric that measures each script run with the metric it deserves.

On the scripted fixtures an exact decode scores zero at every level, and
that zero is worth stating plainly: it certifies the join, the alignment and
the arithmetic between a hypothesis file and a manifest -- a scoring
pipeline that misjoins utterances or miscounts edits does not score zero --
and it says nothing about recognition, because nothing was recognized.  A
nonzero score from a real system's hypotheses is the number that means
something; this command computes both the same way.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from echoscribe.cli.common import EXIT_OK, fail, read_jsonl, write_json
from echoscribe.errors import ConfigError, require
from echoscribe.eval.codeswitch import score_mixed
from echoscribe.eval.wer import (
    CorpusReport,
    character_sequence,
    format_corpus_summary,
    score_corpus,
    word_sequence,
)
from echoscribe.synth.manifest import read_manifest
from echoscribe.text.units import utterance_units

LEVELS = ("units", "text")
METRICS = ("wer", "cer", "hybrid")
LABELS = {"units": "UER", "wer": "WER", "cer": "CER", "hybrid": "CS"}


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``eval`` to the top-level parser."""
    parser = subparsers.add_parser(
        "eval",
        help="score decode output against the planted transcripts",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory prepared by synth and decode")
    parser.add_argument("--level", choices=LEVELS, default="units", help="scoring granularity")
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="wer",
        help="text-level metric; ignored at the units level",
    )
    parser.add_argument("--decode-file", default="decode.jsonl", help="hypotheses to score")
    parser.add_argument("--out", default="scores.json", help="score file to write")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Join hypotheses to references, score them, and write one score file."""
    root = Path(args.workdir)
    manifest_path = root / "manifest.jsonl"
    require(
        manifest_path.exists(),
        f"no manifest at {manifest_path}; run `echoscribe synth` first",
        ConfigError,
    )
    decode_path = root / args.decode_file
    if not decode_path.exists():
        return fail(f"no decode file at {decode_path}; run `echoscribe decode` first")
    records = read_manifest(manifest_path)
    rows = {str(row["utterance_id"]): row for row in read_jsonl(decode_path)}
    missing = [record.utterance_id for record in records if record.utterance_id not in rows]
    if missing:
        return fail(
            f"the decode file has no hypothesis for {len(missing)} utterances, "
            f"starting with {missing[0]!r}"
        )
    field = "hypothesis_units" if args.level == "units" else "hypothesis_text"
    if any(field not in row for row in rows.values()):
        return fail(f"decode rows carry no {field!r}; rerun `echoscribe decode`")
    pairs: list[tuple[Sequence[str], Sequence[str]]]
    if args.level == "units":
        pairs = [
            (
                list(utterance_units(record.tokens, record.languages)),
                list(rows[record.utterance_id]["hypothesis_units"]),
            )
            for record in records
        ]
        corpus = score_corpus(pairs)
        label = LABELS["units"]
    elif args.metric == "hybrid":
        reports = [
            score_mixed(record.text, str(rows[record.utterance_id]["hypothesis_text"])).overall
            for record in records
        ]
        corpus = CorpusReport(tuple(reports))
        label = LABELS["hybrid"]
    else:
        split = word_sequence if args.metric == "wer" else character_sequence
        pairs = [
            (split(record.text), split(str(rows[record.utterance_id]["hypothesis_text"])))
            for record in records
        ]
        corpus = score_corpus(pairs)
        label = LABELS[args.metric]
    payload = {
        "level": args.level,
        "metric": "units" if args.level == "units" else args.metric,
        "totals": {
            "substitutions": corpus.substitutions,
            "deletions": corpus.deletions,
            "insertions": corpus.insertions,
            "errors": corpus.errors,
            "reference_length": corpus.reference_length,
            "error_rate": corpus.error_rate,
        },
        "utterances": [
            {
                "utterance_id": record.utterance_id,
                "errors": report.errors,
                "reference_length": report.reference_length,
                "error_rate": report.error_rate,
            }
            for record, report in zip(records, corpus.reports, strict=True)
        ],
        "ground_truth": "planted transcripts from the synthetic laboratory",
    }
    target = write_json(root / args.out, payload)
    print(f"echoscribe eval: {format_corpus_summary(corpus, label=label)}")
    print(f"  scores: {target}")
    print("  ground truth: planted transcripts, not human labels of real speech")
    return EXIT_OK
