"""Noise robustness: measure it, estimate it, subtract what can be subtracted.

The utilities here are pure NumPy and are tested against the synthetic
laboratory, where both halves of a pair -- the clean signal and the noise it
was mixed with -- are known exactly.  That is what makes the measurements
honest: an improvement in signal-to-noise ratio reported by this package is
computed against a planted reference, never estimated from the degraded
signal alone.

It is also worth saying what spectral methods cannot do.  Subtraction and
Wiener gains attenuate frequency bins where a stationary noise dominates;
they do not restore speech the noise masked, they trade a little distortion
for a lot of noise, and they degrade when the noise is non-stationary or as
loud as the signal in the same bin.  The tests measure the trade rather than
assuming it always wins.

The modules:

:mod:`~echoscribe.noise.stft`
    the transform pair the gains work between;
:mod:`~echoscribe.noise.gains`
    spectral-subtraction and decision-directed Wiener gains;
:mod:`~echoscribe.noise.denoise`
    noise-profile estimation and the end-to-end waveform denoiser;
:mod:`~echoscribe.noise.snr`
    signal-to-noise measurement against a clean reference.
"""

from echoscribe.noise.denoise import (
    DenoiseConfig,
    DenoiseResult,
    denoise,
    estimate_noise_profile,
)
from echoscribe.noise.gains import GAIN_METHODS, spectral_subtraction_gain, wiener_gain
from echoscribe.noise.snr import snr_db, snr_improvement_db
from echoscribe.noise.stft import StftConfig, istft, power_spectrogram, stft

__all__ = [
    "GAIN_METHODS",
    "DenoiseConfig",
    "DenoiseResult",
    "StftConfig",
    "denoise",
    "estimate_noise_profile",
    "istft",
    "power_spectrogram",
    "snr_db",
    "snr_improvement_db",
    "spectral_subtraction_gain",
    "stft",
    "wiener_gain",
]
