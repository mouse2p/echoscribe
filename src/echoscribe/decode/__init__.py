"""Decoding: turning per-frame posteriors into a token sequence.

Every module here consumes the same input -- a ``(frames, symbols)`` matrix of log
probabilities -- and produces the same kind of output: tokens, their scores and the
frames they came from.  CTC and transducer decoding differ in how the frame axis and
the token axis are allowed to move relative to each other, and nothing else, so they
share the validation, the timing and the log-domain arithmetic.

The decoders are pure NumPy.  A trained model is only needed to *produce* the
posterior matrix; once one exists -- from a model, or from
:func:`echoscribe.decode.posteriors.scripted_posteriors` -- decoding is arithmetic
that can be checked exactly.

What a scripted posterior is, and what it is not
------------------------------------------------

:func:`echoscribe.decode.posteriors.scripted_posteriors` produces a matrix whose
answer is known by construction.  A decoder tested against it is being tested on its
own bookkeeping -- does greedy collapse the blanks correctly, does prefix search
merge equal prefixes, does the transducer stay monotonic -- and nothing else.  It
says nothing about accuracy on speech, because there is no speech and no model
involved.

That is the deal every test in this package makes, and it is why the fixtures live
in one module rather than being scattered through the decoder tests: a reader can
see exactly which numbers were planted and which were measured.
"""

from echoscribe.decode.alignment import WordSpan, align_words
from echoscribe.decode.bias import (
    BiasConfig,
    BiasedJoint,
    BiasReport,
    TrieNode,
    advance,
    apply_bias,
    bias_greedy,
    build_trie,
    collect_bonus,
)
from echoscribe.decode.cache import JointCache
from echoscribe.decode.chunked import (
    ChunkConfig,
    ChunkResult,
    ChunkState,
    decode_all_chunks,
    decode_chunk,
)
from echoscribe.decode.confidence import (
    CONFIDENCE_FLOOR,
    CalibrationBin,
    TokenConfidence,
    bucket_counts,
    calibration_bins,
    expected_calibration_error,
    probabilities_of,
    reliability_curve,
    token_confidences,
    utterance_confidence,
)
from echoscribe.decode.ctc_greedy import GreedyResult, ctc_greedy
from echoscribe.decode.ctc_prefix import (
    PrefixBeamResult,
    PrefixHypothesis,
    ctc_prefix_search,
)
from echoscribe.decode.posteriors import (
    check_log_posteriors,
    frame_index,
    frame_times,
    masked_posteriors,
    scripted_posteriors,
    to_log_posteriors,
)
from echoscribe.decode.tokens import DecodedToken, span_gaps, symbols_of, total_score
from echoscribe.decode.transducer import (
    Joint,
    TableJoint,
    TransducerConfig,
    TransducerResult,
    emission_spans,
    transducer_greedy,
)

__all__ = [
    "BiasConfig",
    "BiasReport",
    "BiasedJoint",
    "ChunkConfig",
    "ChunkResult",
    "CONFIDENCE_FLOOR",
    "CalibrationBin",
    "ChunkState",
    "DecodedToken",
    "TokenConfidence",
    "TrieNode",
    "WordSpan",
    "GreedyResult",
    "Joint",
    "JointCache",
    "PrefixBeamResult",
    "PrefixHypothesis",
    "TableJoint",
    "TransducerConfig",
    "TransducerResult",
    "advance",
    "align_words",
    "apply_bias",
    "bias_greedy",
    "bucket_counts",
    "build_trie",
    "calibration_bins",
    "check_log_posteriors",
    "collect_bonus",
    "ctc_greedy",
    "ctc_prefix_search",
    "decode_all_chunks",
    "decode_chunk",
    "frame_index",
    "emission_spans",
    "expected_calibration_error",
    "frame_times",
    "masked_posteriors",
    "probabilities_of",
    "reliability_curve",
    "scripted_posteriors",
    "span_gaps",
    "symbols_of",
    "to_log_posteriors",
    "token_confidences",
    "total_score",
    "utterance_confidence",
    "transducer_greedy",
]
