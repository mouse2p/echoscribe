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

import numpy as np

from echoscribe.errors import FeatureError, require, require_positive
from echoscribe.types import FEATURE_DTYPE, FloatArray

NEGATIVE_INFINITY = float("-inf")

__all__ = [
    "NEGATIVE_INFINITY",
    "PROBABILITY_FLOOR",
    "log_add",
    "log_softmax",
    "logsumexp",
    "safe_log",
    "softmax",
]


def log_add(left: float, right: float) -> float:
    """Return ``log(exp(left) + exp(right))`` computed entirely in log space."""
    if left == NEGATIVE_INFINITY:
        return right
    if right == NEGATIVE_INFINITY:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def logsumexp(
    values: np.ndarray,
    axis: int | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    """Return ``log(sum(exp(values)))`` reduced over ``axis``.

    Fully masked rows are the sharp edge: subtracting a ``-inf`` row maximum
    yields ``nan``.  Those rows are detected and rewritten to ``-inf`` so the
    result is never ``nan``.
    """
    array = np.asarray(values, dtype=FEATURE_DTYPE)
    maximum = np.max(array, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        # A fully masked row makes `array - maximum` nan; the `masked` rewrite
        # below replaces it, so the intermediate warning is expected, not a bug.
        total = np.sum(np.exp(array - maximum), axis=axis, keepdims=True)
        result = np.log(total) + maximum
    masked = np.all(np.isneginf(array), axis=axis, keepdims=True)
    result = np.where(masked, NEGATIVE_INFINITY, result)
    if keepdims:
        return result
    return result.reshape(()) if axis is None else result.squeeze(axis=axis)


def log_softmax(logits: np.ndarray, axis: int = -1) -> FloatArray:
    """Return log-probabilities normalised along ``axis``.

    Accepts raw logits and already-logarithmic scores alike: only differences
    between entries matter, so both are handled correctly.
    """
    array = np.asarray(logits, dtype=FEATURE_DTYPE)
    normaliser = logsumexp(array, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        result = array - normaliser
    # A fully masked row normalises to -inf, and -inf - -inf is nan.  Such a row
    # carries no probability mass at all, so -inf is the honest answer.
    return np.where(np.isnan(result), NEGATIVE_INFINITY, result)


def softmax(logits: np.ndarray, axis: int = -1) -> FloatArray:
    """Return probabilities normalised along ``axis``."""
    return np.exp(log_softmax(logits, axis=axis))


#: Smallest probability that survives the logarithm.  ``log(PROBABILITY_FLOOR)``
#: is about -690.8, far below any score a decoder compares, and stays finite.
PROBABILITY_FLOOR = 1e-300


def safe_log(values: np.ndarray, floor: float = PROBABILITY_FLOOR) -> FloatArray:
    """Return ``log(values)`` with entries below ``floor`` clamped up to it.

    Filterbank energies and products of small posteriors hit exact zero
    routinely; a raw ``log`` would emit ``-inf`` and poison every later sum.
    """
    require_positive("floor", floor)
    array = np.asarray(values, dtype=FEATURE_DTYPE)
    require(bool(np.all(array >= 0.0)), "safe_log expects non-negative input", FeatureError)
    return np.log(np.maximum(array, floor))
