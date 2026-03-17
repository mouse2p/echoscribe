"""Signal-to-noise measurement against a known clean reference.

Every number in this module needs two signals: the degraded one, and the
clean one it was made from.  That is deliberate.  Blind SNR estimators exist,
but a guess presented as a measurement is how a report ends up claiming
something nobody verified.  Where echoscribe reports an SNR, both halves of
the pair exist, and the synthetic laboratory is what guarantees they can.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require
from echoscribe.types import FEATURE_DTYPE, require_rank

SNR_DECIMALS = 6


def snr_db(clean: np.ndarray, degraded: np.ndarray) -> float:
    """``10*log10`` of clean power over residual power, residual = degraded - clean.

    Identical signals leave no residual and measure as positive infinity,
    which is the honest answer rather than a conveniently large number.
    The result is rounded where it is produced, so a summary built from it is
    stable across platforms.
    """
    reference = np.asarray(clean, dtype=FEATURE_DTYPE)
    measured = np.asarray(degraded, dtype=FEATURE_DTYPE)
    require_rank("clean", reference, 1)
    require_rank("degraded", measured, 1)
    require(
        reference.shape == measured.shape,
        f"clean has {reference.size} samples but degraded has {measured.size}; "
        f"a residual between differently sized signals means nothing",
        ConfigError,
    )
    require(reference.size > 0, "cannot measure the SNR of an empty signal", ConfigError)
    signal_power = float(np.mean(reference**2))
    require(
        signal_power > 0,
        "the clean reference is silent, so its SNR is undefined",
        ConfigError,
    )
    residual_power = float(np.mean((measured - reference) ** 2))
    if residual_power == 0.0:
        return float("inf")
    return round(10.0 * float(np.log10(signal_power / residual_power)), SNR_DECIMALS)
