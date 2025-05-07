"""Two-pole resonators and the phone renderer built on them.

A formant is modelled as a complex-conjugate pole pair.  For a centre frequency
``fc`` and a 3 dB bandwidth ``bw`` the poles sit at radius ``exp(-pi * bw / fs)``
and angle ``2 * pi * fc / fs``, which gives the recursion

    ``y[n] = a0 * x[n] + b1 * y[n-1] + b2 * y[n-2]``

with ``b1 = 2 r cos(theta)``, ``b2 = -r**2`` and ``a0 = 1 - b1 - b2``.  Choosing
``a0`` that way makes the resonator unity-gain at DC, so cascading three of them
does not silently change a phone's loudness.

The recursion is a plain Python loop rather than a blocked or FFT-based filter.
That is deliberate: it has no block-boundary artefacts, it starts from an exactly
zero state, and it is what a streaming synthesizer would run anyway.  It is also
the slowest part of the laboratory, which is why the corpora here are short.
"""

from __future__ import annotations

import math

import numpy as np

from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.types import FEATURE_DTYPE, FloatArray


def resonator_coefficients(
    centre_hz: float, bandwidth_hz: float, sample_rate: int
) -> tuple[float, float, float]:
    """Return ``(a0, b1, b2)`` of the two-pole resonator for one formant."""
    require_positive("centre_hz", centre_hz, ConfigError)
    require_positive("bandwidth_hz", bandwidth_hz, ConfigError)
    require_positive("sample_rate", sample_rate, ConfigError)
    require(
        centre_hz < sample_rate / 2.0,
        f"formant at {centre_hz} Hz is at or above Nyquist ({sample_rate / 2.0} Hz)",
        ConfigError,
    )
    radius = math.exp(-math.pi * bandwidth_hz / sample_rate)
    theta = 2.0 * math.pi * centre_hz / sample_rate
    b1 = 2.0 * radius * math.cos(theta)
    b2 = -(radius * radius)
    return 1.0 - b1 - b2, b1, b2


def apply_resonator(
    signal: np.ndarray, centre_hz: float, bandwidth_hz: float, sample_rate: int
) -> FloatArray:
    """Filter ``signal`` through one resonator starting from a zero state."""
    a0, b1, b2 = resonator_coefficients(centre_hz, bandwidth_hz, sample_rate)
    x = np.asarray(signal, dtype=FEATURE_DTYPE)
    require(x.ndim == 1, f"resonator input must be 1-D, got shape {x.shape}", ConfigError)
    # The recursion is inherently sequential; the only speed available without a
    # compiled filter is to stay in plain Python floats and append to a list.
    output = []
    append = output.append
    y1 = 0.0
    y2 = 0.0
    for value in x.tolist():
        y = a0 * value + b1 * y1 + b2 * y2
        append(y)
        y2 = y1
        y1 = y
    return np.asarray(output, dtype=FEATURE_DTYPE)
