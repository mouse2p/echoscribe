"""Mel scales and the triangular mel filterbank.

Two scales are implemented because they are both in common use and they are *not*
interchangeable: a filterbank built on one and applied to features extracted with
the other shifts every bin.

``htk``
    ``mel = 2595 * log10(1 + f / 700)``.  A single logarithm over the whole range,
    which is what Kaldi's fbank uses.
``slaney``
    Linear below 1 kHz (``mel = f / (200/3)``) and logarithmic above, anchored so
    that 1 kHz maps to mel 15 and 6.4 kHz maps to mel 42.  The linear low end
    gives finer resolution where speech has most of its structure.

Both are invertible, and both are implemented here from their definitions rather
than imported from a larger audio stack.
"""

from __future__ import annotations

import math

import numpy as np

from echoscribe.errors import ConfigError, require, require_in

#: Mel scales understood by :func:`hz_to_mel` and :func:`mel_to_hz`.
MEL_SCALES = ("htk", "slaney")

HTK_REFERENCE_HZ = 700.0
HTK_REFERENCE_MEL = 2595.0

#: Slaney's linear region ends at 1 kHz with a spacing of 200/3 Hz per mel.
SLANEY_MIN_LOG_HZ = 1000.0
SLANEY_HZ_PER_MEL = 200.0 / 3.0
SLANEY_MIN_LOG_MEL = SLANEY_MIN_LOG_HZ / SLANEY_HZ_PER_MEL
SLANEY_LOG_STEP = math.log(6.4) / 27.0


def hz_to_mel(frequency: np.ndarray | float, scale: str = "htk") -> np.ndarray | float:
    """Convert Hz to mel on the requested scale, preserving scalar-ness."""
    require_in("scale", scale, MEL_SCALES, ConfigError)
    values = np.asarray(frequency, dtype=np.float64)
    require(bool(np.all(values >= 0.0)), "frequencies must be non-negative", ConfigError)
    if scale == "htk":
        result = HTK_REFERENCE_MEL * np.log10(1.0 + values / HTK_REFERENCE_HZ)
    else:
        linear = values / SLANEY_HZ_PER_MEL
        logarithmic = (
            SLANEY_MIN_LOG_MEL
            + np.log(np.maximum(values, SLANEY_MIN_LOG_HZ) / SLANEY_MIN_LOG_HZ) / SLANEY_LOG_STEP
        )
        result = np.where(values < SLANEY_MIN_LOG_HZ, linear, logarithmic)
    return float(result) if np.ndim(frequency) == 0 else result


def mel_to_hz(mel: np.ndarray | float, scale: str = "htk") -> np.ndarray | float:
    """Invert :func:`hz_to_mel`."""
    require_in("scale", scale, MEL_SCALES, ConfigError)
    values = np.asarray(mel, dtype=np.float64)
    require(bool(np.all(values >= 0.0)), "mel values must be non-negative", ConfigError)
    if scale == "htk":
        result = HTK_REFERENCE_HZ * (10.0 ** (values / HTK_REFERENCE_MEL) - 1.0)
    else:
        linear = values * SLANEY_HZ_PER_MEL
        logarithmic = SLANEY_MIN_LOG_HZ * np.exp(SLANEY_LOG_STEP * (values - SLANEY_MIN_LOG_MEL))
        result = np.where(values < SLANEY_MIN_LOG_MEL, linear, logarithmic)
    return float(result) if np.ndim(mel) == 0 else result
