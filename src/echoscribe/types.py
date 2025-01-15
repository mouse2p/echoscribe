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
from numpy.typing import NDArray

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
]
