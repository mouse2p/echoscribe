"""Text handling for a code-switch aware recognizer.

The order of operations here is fixed and it matters: normalise first, then split
into script runs, then segment, then map to modelling units.  Each stage assumes
the one before it removed a class of ambiguity -- a fullwidth ``A`` and an ASCII
``A`` have to be the same token by the time a vocabulary is consulted, and a
Chinese character must not be glued to an English word by the time a scorer counts
errors.
"""

from echoscribe.text.normalize import (
    NORMALIZATION_FORMS,
    NORMALIZATION_STAGES,
    PUNCTUATION_FOLD,
    NormalizeConfig,
    collapse_whitespace,
    fold_punctuation,
    normalize_text,
    strip_accents,
)
from echoscribe.text.script import SCRIPTS, ScriptRun, is_han, script_of, script_runs
from echoscribe.text.segment import (
    WORD_SCRIPTS,
    SegmentationSummary,
    SegmentConfig,
    Token,
    count_by_script,
    join_for_display,
    script_share,
    segment,
    summarize,
    token_texts,
)

__all__ = [
    "NORMALIZATION_FORMS",
    "NORMALIZATION_STAGES",
    "PUNCTUATION_FOLD",
    "SCRIPTS",
    "WORD_SCRIPTS",
    "NormalizeConfig",
    "SegmentConfig",
    "SegmentationSummary",
    "Token",
    "ScriptRun",
    "collapse_whitespace",
    "count_by_script",
    "join_for_display",
    "fold_punctuation",
    "is_han",
    "normalize_text",
    "script_of",
    "script_share",
    "segment",
    "summarize",
    "script_runs",
    "strip_accents",
    "token_texts",
]
