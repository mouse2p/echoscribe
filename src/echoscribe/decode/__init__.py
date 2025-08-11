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

__all__: list[str] = []
