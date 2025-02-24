"""Symmetric window functions shared by the filter designer and the frame front end.

They live in one module because the same name has to mean the same thing in both
places: a Hamming window on an FIR prototype and a Hamming window on an analysis
frame are the same coefficients, and two implementations is how they drift apart.

``povey`` is the analysis window Kaldi's fbank recipe uses by default: a Hann
window raised to the 0.85 power.  Raising it widens the main lobe slightly and
lifts the shoulders, which reduces the frame-edge discontinuity that otherwise
leaks energy across mel bins.

All of them are the *symmetric* (DFT-even) variants NumPy provides, which is what
an analysis frame wants.  A filter designer needing the periodic variant can ask
for ``length + 1`` samples and drop the last one.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require_in, require_positive

#: Windows accepted for FIR filter design.
FILTER_WINDOWS = ("rectangular", "hamming", "hann", "blackman")

#: Windows accepted when framing a signal for spectral analysis.
FRAME_WINDOWS = FILTER_WINDOWS + ("povey",)

#: Exponent that turns a Hann window into a Povey window.
POVEY_EXPONENT = 0.85


def window_vector(kind: str, length: int) -> np.ndarray:
    """Return the requested symmetric window as ``length`` float64 samples.

    ``length == 1`` yields ``[1.0]`` for every window, matching NumPy's own
    degenerate-case behaviour, so single-tap filters and single-sample frames
    need no special casing by callers.
    """
    require_positive("length", length)
    require_in("window", kind, FRAME_WINDOWS, ConfigError)
    if kind == "rectangular":
        return np.ones(length, dtype=np.float64)
    if kind == "hamming":
        return np.hamming(length).astype(np.float64)
    if kind == "hann":
        return np.hanning(length).astype(np.float64)
    if kind == "blackman":
        return np.blackman(length).astype(np.float64)
    return np.hanning(length).astype(np.float64) ** POVEY_EXPONENT


__all__ = ["FILTER_WINDOWS", "FRAME_WINDOWS", "POVEY_EXPONENT", "window_vector"]
