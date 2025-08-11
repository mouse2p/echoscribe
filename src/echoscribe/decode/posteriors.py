"""Posterior matrices: the input every decoder consumes, and fixtures for it.

A posterior matrix is ``(frames, symbols)`` of natural logarithms, one normalised
distribution per frame.  Two facts about it drive everything downstream: the frame
axis is time and the symbol axis is the output alphabet, and a decoder's job is to
pick a monotonic path through that grid.

:func:`scripted_posteriors` builds a matrix whose argmax path is a chosen token
sequence.  It is a *fixture*, not a model output: the point of it is that a decoder
run against a known answer can be checked exactly, which is impossible with a
trained model whose errors and whose decoder's errors cannot be told apart.  Nothing
in this module claims to resemble the posteriors a real acoustic model produces.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from echoscribe.errors import DecodeError, require, require_positive
from echoscribe.numeric import NEGATIVE_INFINITY, log_softmax, logsumexp
from echoscribe.types import FEATURE_DTYPE, FloatArray

#: Logit given to the planted symbol in a scripted frame.
SCRIPTED_PEAK_LOGIT = 8.0

#: Logit given to every other symbol in a scripted frame.
SCRIPTED_BACKGROUND_LOGIT = -4.0


def frame_times(count: int, frame_shift_ms: float = 10.0) -> FloatArray:
    """Centre-free start time, in seconds, of each of ``count`` frames."""
    require_positive("frame_shift_ms", frame_shift_ms, DecodeError)
    require(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0,
        f"frame count must be a non-negative integer, got {count!r}",
        DecodeError,
    )
    return np.arange(count, dtype=FEATURE_DTYPE) * (frame_shift_ms / 1000.0)


def frame_index(time_seconds: float, frame_shift_ms: float = 10.0) -> int:
    """Index of the frame a timestamp falls in, clamped at zero."""
    require_positive("frame_shift_ms", frame_shift_ms, DecodeError)
    return max(0, int(np.floor(time_seconds * 1000.0 / frame_shift_ms)))


def scripted_posteriors(
    path: Sequence[int],
    num_symbols: int | None = None,
    peak_logit: float = SCRIPTED_PEAK_LOGIT,
    background_logit: float = SCRIPTED_BACKGROUND_LOGIT,
) -> FloatArray:
    """A log-posterior matrix whose per-frame argmax is exactly ``path``.

    Each frame gets ``peak_logit`` on the planted symbol and ``background_logit``
    everywhere else, then the row is renormalised so it is a real log distribution.
    The gap between the two is what makes the answer unambiguous for a greedy
    decoder while still leaving a beam search something to prune.
    """
    require(len(path) > 0, "a scripted posterior needs at least one frame", DecodeError)
    require_positive("peak_logit", peak_logit, DecodeError)
    require(
        peak_logit > background_logit,
        f"peak_logit ({peak_logit}) must exceed background_logit ({background_logit})",
        DecodeError,
    )
    symbols = [int(symbol) for symbol in path]
    width = num_symbols if num_symbols is not None else max(symbols) + 1
    require(
        width > max(symbols),
        f"{width} symbols cannot hold symbol id {max(symbols)}",
        DecodeError,
    )
    matrix = np.full((len(symbols), width), float(background_logit), dtype=FEATURE_DTYPE)
    for frame, symbol in enumerate(symbols):
        require(symbol >= 0, f"symbol ids must not be negative, got {symbol}", DecodeError)
        matrix[frame, symbol] = peak_logit
    return log_softmax(matrix, axis=-1)


#: Tolerance for the "rows sum to one in the linear domain" check.
NORMALIZATION_TOLERANCE = 1e-6


def check_log_posteriors(
    matrix: np.ndarray, name: str = "posteriors", expected_symbols: int | None = None
) -> FloatArray:
    """Validate a log-posterior matrix and return it as ``float64``.

    Three things are checked, because each one silently produces a wrong decoding
    rather than an error:

    * the matrix is 2-D and non-empty along the symbol axis;
    * every entry is finite or ``-inf`` -- a NaN anywhere makes every path score
      NaN, and an argmax over NaNs is arbitrary;
    * each row exponentiates to one, so the scores really are log probabilities and
      a path score can be compared with another path's.
    """
    values = np.asarray(matrix, dtype=FEATURE_DTYPE)
    require(
        values.ndim == 2,
        f"{name} must be a (frames, symbols) matrix, got shape {values.shape}",
        DecodeError,
    )
    require(
        values.shape[1] > 0,
        f"{name} must offer at least one symbol, got shape {values.shape}",
        DecodeError,
    )
    require(
        bool(np.all(np.isfinite(values) | np.isneginf(values))),
        f"{name} contains NaN or +inf entries, which no decoder can rank",
        DecodeError,
    )
    if values.shape[0] == 0:
        return values
    totals = np.exp(logsumexp(values, axis=-1))
    require(
        bool(np.all(np.abs(totals - 1.0) <= NORMALIZATION_TOLERANCE)),
        f"{name} rows are not normalised: totals range "
        f"[{float(totals.min()):.6f}, {float(totals.max()):.6f}]",
        DecodeError,
    )
    if expected_symbols is not None:
        require(
            values.shape[1] == expected_symbols,
            f"{name} has {values.shape[1]} symbols but the model declares {expected_symbols}",
            DecodeError,
        )
    return values


def to_log_posteriors(probabilities: np.ndarray, name: str = "probabilities") -> FloatArray:
    """Renormalise a linear-domain matrix into log posteriors.

    Handy for turning a hand-written probability table into something
    :func:`check_log_posteriors` accepts, and for recovering from a model head that
    emitted logits.
    """
    values = np.asarray(probabilities, dtype=FEATURE_DTYPE)
    require(
        values.ndim == 2,
        f"{name} must be a (frames, symbols) matrix, got shape {values.shape}",
        DecodeError,
    )
    require(
        bool(np.all(values >= 0.0)),
        f"{name} must not hold negative probabilities",
        DecodeError,
    )
    return log_softmax(np.log(np.maximum(values, 1e-300)), axis=-1)


def masked_posteriors(matrix: np.ndarray, allow: Sequence[int]) -> FloatArray:
    """Zero out every symbol not in ``allow`` and renormalise the row.

    This is how a constrained decode is expressed: a vocabulary restriction, a
    grammar, or "no token twice in a row" all become a per-frame mask.  A row whose
    allowed set is empty becomes all ``-inf``, which a decoder then reports as an
    impossible path rather than silently picking something.
    """
    values = check_log_posteriors(matrix).copy()
    keep = np.full(values.shape[1], NEGATIVE_INFINITY, dtype=FEATURE_DTYPE)
    for symbol in allow:
        require(
            0 <= int(symbol) < values.shape[1],
            f"symbol {symbol} is outside the {values.shape[1]}-symbol alphabet",
            DecodeError,
        )
        keep[int(symbol)] = 0.0
    # log_softmax already maps a fully masked row to -inf, so no special case is
    # needed here: an impossible frame stays impossible instead of picking a symbol.
    return log_softmax(values + keep[None, :], axis=-1)
