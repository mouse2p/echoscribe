"""Evaluation: turning a hypothesis and a reference into numbers.

Everything in this package is pure arithmetic on token sequences, which is exactly
why it can be tested exactly.  A recognizer's accuracy is a property of the
recognizer; a scorer's correctness is a property of the scorer, and the two must not
be confused.  A WER of 12% means something only if the alignment, the tokenisation
and the error counting behind it are right, and none of those can be checked by
looking at the number.

The modules:

:mod:`~echoscribe.eval.edit`
    edit-distance alignment and its substitution/insertion/deletion breakdown;
:mod:`~echoscribe.eval.wer`
    word and character error rates over that alignment;
:mod:`~echoscribe.eval.codeswitch`
    the hybrid score that measures each script run with the metric it deserves;
:mod:`~echoscribe.eval.bench`
    real-time-factor measurement with warmup and repeats;
:mod:`~echoscribe.eval.report`
    deterministic SVG output, so a report is a file that can be diffed.
"""

from echoscribe.eval.bench import (
    BENCH_SCHEMA_VERSION,
    BenchResult,
    TimingStatistics,
    measure_rtf,
    timing_statistics,
)
from echoscribe.eval.codeswitch import (
    CodeSwitchReport,
    ScriptScore,
    count_code_switches,
    format_code_switch,
    mixed_token_sequence,
    score_mixed,
)
from echoscribe.eval.edit import (
    EDIT_OPERATIONS,
    ERROR_OPERATIONS,
    Alignment,
    EditOp,
    align,
    brute_force_distance,
    edit_distance,
)
from echoscribe.eval.report import (
    Section,
    SvgElement,
    alignment_section,
    digest_report,
    escape_text,
    format_length,
    histogram_counts,
    histogram_section,
    metadata_line,
    read_report,
    render_report,
    svg_document,
    table_section,
    write_report,
)
from echoscribe.eval.wer import (
    CorpusReport,
    ScoreReport,
    cer,
    character_sequence,
    format_summary,
    score_corpus,
    score_tokens,
    wer,
    word_sequence,
)

__all__ = [
    "BENCH_SCHEMA_VERSION",
    "EDIT_OPERATIONS",
    "ERROR_OPERATIONS",
    "Alignment",
    "BenchResult",
    "CodeSwitchReport",
    "CorpusReport",
    "EditOp",
    "ScoreReport",
    "ScriptScore",
    "Section",
    "SvgElement",
    "TimingStatistics",
    "align",
    "alignment_section",
    "brute_force_distance",
    "cer",
    "character_sequence",
    "count_code_switches",
    "digest_report",
    "edit_distance",
    "escape_text",
    "format_code_switch",
    "format_length",
    "format_summary",
    "histogram_counts",
    "histogram_section",
    "measure_rtf",
    "metadata_line",
    "mixed_token_sequence",
    "read_report",
    "render_report",
    "score_corpus",
    "score_mixed",
    "score_tokens",
    "svg_document",
    "table_section",
    "timing_statistics",
    "wer",
    "word_sequence",
    "write_report",
]
