"""Log-domain arithmetic shared by the CTC, transducer and confidence code.

Decoders accumulate the probabilities of exponentially many paths.  In the
linear domain that underflows to zero within a few dozen frames, so every score
in echoscribe is a natural logarithm and combination happens through
:func:`log_add`.  Two identities matter:

* ``log_add(-inf, x) == x`` -- an impossible path contributes nothing.
* ``log_add(a, b)`` is computed as ``max + log1p(exp(-|a - b|))`` so neither
  ``exp`` overflow nor catastrophic cancellation can occur.

:func:`logsumexp` is the array form of the same identity; CTC prefix search and
confidence estimation both call it.
"""

from __future__ import annotations

import math

NEGATIVE_INFINITY = float("-inf")

__all__ = ["NEGATIVE_INFINITY", "log_add"]


def log_add(left: float, right: float) -> float:
    """Return ``log(exp(left) + exp(right))`` computed entirely in log space."""
    if left == NEGATIVE_INFINITY:
        return right
    if right == NEGATIVE_INFINITY:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))
