"""Noise-profile estimation and the end-to-end waveform denoiser.

The estimator here makes one assumption and says so: the recording begins
with noise before it begins with speech, and the first frames are therefore
a sample of the noise alone.  Recordings from the synthetic laboratory are
mixed that way, and so are most real capture pipelines that keep a moment of
room tone before the first word.  A tracker that followed a moving noise
floor (minimum statistics) would drop the assumption; it would also stop
being a dozen testable lines, and everything downstream only needs *a*
profile, not the best possible one.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require
from echoscribe.types import FEATURE_DTYPE, FloatArray


def estimate_noise_profile(power: FloatArray, frames: int) -> FloatArray:
    """The per-bin mean power over the first ``frames`` frames.

    Averaging rather than taking a minimum biases the profile up when the
    lead-in contains any speech, which costs a little signal and buys
    stability: a minimum over few frames underestimates the noise floor and
    then leaves exactly the residual the gain was supposed to remove.
    """
    matrix = np.asarray(power, dtype=FEATURE_DTYPE)
    require(
        matrix.ndim == 2,
        f"power must be a (frames, bins) matrix, got {matrix.ndim}-D",
        ConfigError,
    )
    require(
        isinstance(frames, int) and not isinstance(frames, bool) and frames >= 1,
        f"frames must be an integer >= 1, got {frames!r}",
        ConfigError,
    )
    require(
        frames <= matrix.shape[0],
        f"the profile asks for {frames} frames but the spectrogram has {matrix.shape[0]}",
        ConfigError,
    )
    return matrix[:frames].mean(axis=0)
