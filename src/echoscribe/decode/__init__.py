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

__all__ = [
    "DecodedToken",
    "GreedyResult",
    "PrefixBeamResult",
    "PrefixHypothesis",
    "check_log_posteriors",
    "ctc_greedy",
    "ctc_prefix_search",
    "frame_index",
    "frame_times",
    "masked_posteriors",
    "scripted_posteriors",
    "span_gaps",
    "symbols_of",
    "to_log_posteriors",
    "total_score",
]
