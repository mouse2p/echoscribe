"""Text handling for a code-switch aware recognizer.

The order of operations here is fixed and it matters: normalise first, then split
into script runs, then segment, then map to modelling units.  Each stage assumes
the one before it removed a class of ambiguity -- a fullwidth ``A`` and an ASCII
``A`` have to be the same token by the time a vocabulary is consulted, and a
Chinese character must not be glued to an English word by the time a scorer counts
errors.
"""

from echoscribe.text.bpe import (
    BPE_SCHEMA_VERSION,
    WORD_END,
    BpeModel,
    count_pairs,
    detokenize,
    learn_bpe,
    merge_pair,
    tokenize,
    word_symbols,
)
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
from echoscribe.text.unigram import (
    UnigramModel,
    best_segmentation,
    candidate_pieces,
    corpus_log_likelihood,
    estimate_log_probabilities,
    learn_unigram,
)
from echoscribe.text.units import (
    TONE_MARKS,
    UNIT_LEVELS,
    ToneUnit,
    syllable_units,
    unit_vocabulary,
    utterance_units,
)
from echoscribe.text.vocab import DEFAULT_RESERVED, VOCAB_SCHEMA_VERSION, Vocabulary

__all__ = [
    "NORMALIZATION_FORMS",
    "NORMALIZATION_STAGES",
    "PUNCTUATION_FOLD",
    "SCRIPTS",
    "UNIT_LEVELS",
    "VOCAB_SCHEMA_VERSION",
    "UnigramModel",
    "Vocabulary",
    "WORD_END",
    "WORD_SCRIPTS",
    "BPE_SCHEMA_VERSION",
    "BpeModel",
    "DEFAULT_RESERVED",
    "NormalizeConfig",
    "SegmentConfig",
    "SegmentationSummary",
    "TONE_MARKS",
    "Token",
    "ToneUnit",
    "ScriptRun",
    "best_segmentation",
    "candidate_pieces",
    "collapse_whitespace",
    "corpus_log_likelihood",
    "count_by_script",
    "count_pairs",
    "detokenize",
    "join_for_display",
    "learn_unigram",
    "learn_bpe",
    "merge_pair",
    "estimate_log_probabilities",
    "fold_punctuation",
    "is_han",
    "normalize_text",
    "script_of",
    "script_share",
    "segment",
    "summarize",
    "syllable_units",
    "script_runs",
    "strip_accents",
    "token_texts",
    "unit_vocabulary",
    "tokenize",
    "utterance_units",
    "word_symbols",
]
