#!/usr/bin/env python3
"""Score code-switched pairs and render one deterministic SVG report.

The hypothesis pairs below are *hand-written demonstrations* with planted
errors -- they did not come out of a recognizer, and the scores say nothing
about recognition.  What the example exercises is the scoring machinery:
script-run attribution, the hybrid metric, and a report whose bytes are a
pure function of its inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echoscribe.eval.codeswitch import format_code_switch, score_mixed
from echoscribe.eval.edit import align
from echoscribe.eval.report import (
    alignment_section,
    digest_report,
    render_report,
    table_section,
    write_report,
)
from echoscribe.eval.wer import cer, format_summary, wer
from echoscribe.meta import run_metadata

PAIRS = (
    ("u1", "the meeting is at three", "the meeting is at tree"),
    ("u2", "今天天气不错", "今天气不错"),
    ("u3", "打开the document now", "打开document now"),
    ("u4", "please call me tomorrow", "please call me tomorrow"),
)


def main() -> int:
    """Score every pair, print the lines, render the report."""
    parser = argparse.ArgumentParser(description="code-switch scoring report")
    parser.add_argument("--workdir", required=True, help="directory for the SVG report")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("per-metric lines, one per hand-written pair:")
    print(f"  u1 WER: {format_summary(wer(PAIRS[0][1], PAIRS[0][2]))}")
    print(f"  u2 CER: {format_summary(cer(PAIRS[1][1], PAIRS[1][2]), label='CER')}")

    rows: list[tuple[str, str]] = []
    reports = []
    for identifier, reference, hypothesis in PAIRS:
        mixed = score_mixed(reference, hypothesis)
        reports.append(mixed)
        rows.append((identifier, format_code_switch(mixed)))
        print(f"  {identifier} CS:  {rows[-1][1]}")

    sections = [
        table_section(rows, title="hybrid scores"),
        alignment_section(
            align(*[list(side) for side in _mixed_pairs(PAIRS[2][1], PAIRS[2][2])]).pairs(),
            title="mixed-token alignment of u3",
        ),
    ]
    metadata = run_metadata(
        "codeswitch-example",
        seed=0,
        pairs=len(PAIRS),
        metric="hybrid",
    )
    document = render_report(
        "code-switch score report",
        sections,
        metadata,
        description="hand-written hypothesis pairs with planted errors",
    )
    target = write_report(workdir / "codeswitch_report.svg", document, overwrite=True)
    print(f"report written to {target}")
    print(f"report digest: {digest_report(document)}")
    total_errors = sum(report.overall.errors for report in reports)
    total_length = sum(report.overall.reference_length for report in reports)
    print(f"corpus totals: {total_errors} errors over {total_length} mixed tokens")
    print("note: hand-written pairs with planted errors, not recognizer output")
    return 0


def _mixed_pairs(reference: str, hypothesis: str) -> tuple[list[str], list[str]]:
    """Token lists for the alignment grid, using the scorer's own tokenizer."""
    from echoscribe.eval.codeswitch import mixed_token_sequence

    return (
        [token for token, _ in mixed_token_sequence(reference)],
        [token for token, _ in mixed_token_sequence(hypothesis)],
    )


if __name__ == "__main__":
    raise SystemExit(main())
