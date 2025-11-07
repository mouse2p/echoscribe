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

__all__: list[str] = []
