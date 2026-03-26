"""Pointwise spectral gains: what each time-frequency bin keeps.

Both gains here read a signal power spectrogram and a noise power estimate
and return multipliers in ``[floor, 1]``.  The floor is never cosmetic:
gains that switch hard between zero and one in neighbouring bins are what
produce *musical noise*, the isolated tonal blips that make aggressive
subtraction sound worse than the noise it removed.  A floor trades a little
leftover noise for the removal of that artifact.

The noise input may be a full per-frame matrix or a single per-bin profile,
which broadcasts across frames -- the second is what a recording with a
noise-only lead-in produces, and the common case here.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require, require_positive, require_range
from echoscribe.types import FEATURE_DTYPE, FloatArray

GAIN_METHODS = ("subtraction", "wiener")
POWER_EPSILON = 1e-20


def validate_powers(
    signal_power: np.ndarray, noise_power: np.ndarray
) -> tuple[FloatArray, FloatArray]:
    """Check and broadcast a signal/noise power pair to a common shape."""
    signal = np.asarray(signal_power, dtype=FEATURE_DTYPE)
    noise = np.asarray(noise_power, dtype=FEATURE_DTYPE)
    require(
        signal.ndim == 2,
        f"signal_power must be a (frames, bins) matrix, got {signal.ndim}-D",
        ConfigError,
    )
    require(
        np.all(np.isfinite(signal)) and np.all(signal >= 0),
        "signal powers must be finite and non-negative",
        ConfigError,
    )
    if noise.ndim == 1:
        require(
            noise.size == signal.shape[1],
            f"a per-bin noise profile must hold {signal.shape[1]} values, got {noise.size}",
            ConfigError,
        )
        # Broadcast eagerly: the strict shape check below is for genuinely
        # mismatched matrices, not for the profile case it just ruled legal.
        noise = np.broadcast_to(noise[None, :], signal.shape)
    require(
        noise.shape == signal.shape,
        f"noise powers have shape {tuple(noise.shape)} but signal powers have "
        f"{tuple(signal.shape)}",
        ConfigError,
    )
    require(
        np.all(np.isfinite(noise)) and np.all(noise >= 0),
        "noise powers must be finite and non-negative",
        ConfigError,
    )
    return signal, noise


def validate_floor(floor: float) -> float:
    """A gain floor is a fraction of the signal: it lives in ``[0, 1)``."""
    require(
        isinstance(floor, (int, float)) and not isinstance(floor, bool),
        f"floor must be a number, got {floor!r}",
        ConfigError,
    )
    require_range("floor", float(floor), 0.0, 1.0)
    require(float(floor) < 1.0, f"floor must stay below 1.0, got {floor!r}", ConfigError)
    return float(floor)


def spectral_subtraction_gain(
    signal_power: np.ndarray,
    noise_power: np.ndarray,
    oversubtraction: float = 1.0,
    floor: float = 0.05,
) -> FloatArray:
    """``max(1 - alpha * noise / signal, floor)``, computed pointwise.

    ``oversubtraction`` above one removes more of the residual -- and more
    of the signal -- than a literal subtraction would; values below one leave
    noise behind on purpose.  Powers are divided through an epsilon floor, so
    a silent bin with a silent estimate keeps gain one and a silent bin with
    a loud estimate drops to the floor, and neither divides by zero.
    """
    signal, noise = validate_powers(signal_power, noise_power)
    require_positive("oversubtraction", oversubtraction)
    validated_floor = validate_floor(floor)
    ratio = noise / np.maximum(signal, POWER_EPSILON)
    return np.maximum(1.0 - oversubtraction * ratio, validated_floor)


def wiener_gain(
    signal_power: np.ndarray,
    noise_power: np.ndarray,
    floor: float = 0.05,
    smoothing: float = 0.9,
) -> FloatArray:
    """Decision-directed Wiener gain, after Ephraim & Malah (1984).

    The a-priori estimate behind each frame's gain leans on the previous
    frame's own output -- ``smoothing`` of it -- plus ``(1 - smoothing)`` of
    what the current frame suggests on its own.  The recursion is the point:
    it averages the gain over time, which is what suppresses the frame-to-
    frame flutter plain subtraction leaves behind, at the cost of smearing
    transients slightly in time.  The first frame has no history and uses
    its instantaneous estimate alone, so on a single frame this gain and a
    subtraction with ``oversubtraction=1`` agree.
    """
    signal, noise = validate_powers(signal_power, noise_power)
    validated_floor = validate_floor(floor)
    require_range("smoothing", smoothing, 0.0, 1.0)
    posterior = signal / np.maximum(noise, POWER_EPSILON)
    instantaneous = np.maximum(posterior - 1.0, 0.0)
    gains = np.empty_like(signal)
    previous_clean: FloatArray | None = None
    for frame in range(signal.shape[0]):
        if previous_clean is None:
            prior = instantaneous[frame]
        else:
            directed = smoothing * previous_clean / np.maximum(noise[frame], POWER_EPSILON)
            prior = directed + (1.0 - smoothing) * instantaneous[frame]
        gain = prior / (1.0 + prior)
        gains[frame] = np.maximum(gain, validated_floor)
        previous_clean = (gains[frame] ** 2) * signal[frame]
    return gains
