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
"""

__all__: list[str] = []
