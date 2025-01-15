"""Array aliases and the dtype contract shared by every echoscribe layer.

Speech code is easy to get wrong through silent dtype drift: an ``int16`` PCM
buffer treated as float, or a ``float32`` feature matrix compared byte-for-byte
against a ``float64`` reference.  echoscribe therefore fixes three dtypes up
front and every public entry point converts to them explicitly.

``FEATURE_DTYPE`` is ``float64`` so that golden feature tests are stable across
BLAS builds; ``SAMPLE_DTYPE`` is ``float32`` because waveforms are stored and
compared at 32-bit precision.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from echoscribe.errors import FeatureError, require

#: Waveform samples in ``[-1.0, 1.0]``.
SampleArray = NDArray[np.float32]
#: Log-mel, spectrum and posterior matrices.
FloatArray = NDArray[np.float64]
#: Log-domain score matrices; every entry is ``<= 0`` or ``-inf``.
LogArray = NDArray[np.float64]
#: Token ids, frame indices and alignment positions.
IntArray = NDArray[np.intp]

SAMPLE_DTYPE = np.dtype(np.float32)
FEATURE_DTYPE = np.dtype(np.float64)
INDEX_DTYPE = np.dtype(np.intp)

__all__ = [
    "FEATURE_DTYPE",
    "INDEX_DTYPE",
    "SAMPLE_DTYPE",
    "FloatArray",
    "IntArray",
    "LogArray",
    "SampleArray",
    "as_features",
    "as_samples",
    "require_finite",
    "require_non_empty",
    "require_rank",
]


def as_samples(values: ArrayLike, name: str = "sample buffer") -> SampleArray:
    """Coerce ``values`` to a contiguous 1-D ``float32`` waveform buffer.

    The result is always a copy, so a caller cannot mutate a cached waveform
    through the array it gets back.
    """
    array = np.array(values, dtype=SAMPLE_DTYPE, order="C")
    require(array.ndim == 1, f"{name} must be 1-D, got shape {array.shape}", FeatureError)
    require_finite(name, array)
    return array


def as_features(values: ArrayLike, name: str = "feature matrix") -> FloatArray:
    """Coerce ``values`` to a 2-D ``float64`` matrix, copying only if needed."""
    array = np.asarray(values, dtype=FEATURE_DTYPE, order="C")
    require(array.ndim == 2, f"{name} must be 2-D, got shape {array.shape}", FeatureError)
    require_finite(name, array)
    return array


def require_finite(name: str, array: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return ``array`` when it holds no NaN and no infinity."""
    bad = int(np.count_nonzero(~np.isfinite(array)))
    require(bad == 0, f"{name} must be finite; found {bad} non-finite entries", FeatureError)
    return array


def require_rank(name: str, array: np.ndarray, rank: int) -> np.ndarray:
    """Return ``array`` when it has exactly ``rank`` dimensions."""
    require(
        array.ndim == rank,
        f"{name} must be {rank}-D, got {array.ndim}-D with shape {array.shape}",
        FeatureError,
    )
    return array


def require_non_empty(name: str, array: np.ndarray) -> np.ndarray:
    """Return ``array`` when it holds at least one element."""
    require(array.size > 0, f"{name} must not be empty", FeatureError)
    return array
