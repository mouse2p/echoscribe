"""The short-time Fourier transform the denoisers work inside.

The transform pair is a contract: :func:`stft` cuts a signal into windowed
frames and transforms them, and :func:`istft` puts them back with a
normalized overlap-add that recovers the original samples exactly wherever
the squared window accumulates above zero.  Gains between the two are
pointwise functions of frequency and frame, which is what makes spectral
denoising a one-line idea -- and the transform pair is where the idea either
holds or quietly falls apart, so it gets tested as its own module.

No scaling is applied in either direction: the forward transform multiplies
by the window once, the inverse multiplies by it again and divides by the
accumulated square, so a round-trip is the identity rather than the identity
times some constant a caller has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require, require_in
from echoscribe.types import FEATURE_DTYPE, require_rank
from echoscribe.windows import FILTER_WINDOWS, window_vector


@dataclass(frozen=True)
class StftConfig:
    """Frame geometry of the transform pair, validated once at construction."""

    frame_length: int = 400
    hop_length: int = 160
    fft_size: int = 512
    window: str = "hann"

    def __post_init__(self) -> None:
        for name, value in (
            ("frame_length", self.frame_length),
            ("hop_length", self.hop_length),
            ("fft_size", self.fft_size),
        ):
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{name} must be a positive integer, got {value!r}",
                ConfigError,
            )
        require(
            self.fft_size >= self.frame_length,
            f"fft_size ({self.fft_size}) must cover frame_length ({self.frame_length})",
            ConfigError,
        )
        require(
            self.hop_length <= self.frame_length,
            f"hop_length ({self.hop_length}) must not exceed frame_length ({self.frame_length})",
            ConfigError,
        )
        require_in("window", self.window, FILTER_WINDOWS, ConfigError)

    @property
    def bin_count(self) -> int:
        """Bins a real FFT of ``fft_size`` points produces."""
        return self.fft_size // 2 + 1


def stft(signal: np.ndarray, config: StftConfig | None = None) -> np.ndarray:
    """The complex short-time spectrum of ``signal``, one row per frame.

    Frames are cut with a stride, windowed, and zero-padded to ``fft_size``.
    A signal shorter than one frame is refused rather than padded: a
    spectrum of mostly zeros would be an invention, and every caller here
    works with signals long enough to frame.
    """
    settings = StftConfig() if config is None else config
    samples = np.asarray(signal, dtype=FEATURE_DTYPE)
    require_rank("signal", samples, 1)
    require(
        samples.size >= settings.frame_length,
        f"the signal has {samples.size} samples, fewer than one "
        f"{settings.frame_length}-sample frame",
        ConfigError,
    )
    window = window_vector(settings.window, settings.frame_length).astype(FEATURE_DTYPE)
    offsets = np.arange(settings.frame_length)
    starts = np.arange(0, samples.size - settings.frame_length + 1, settings.hop_length)
    frames = samples[starts[:, None] + offsets[None, :]] * window[None, :]
    return np.fft.rfft(frames, n=settings.fft_size, axis=1)


def istft(
    spectrogram: np.ndarray,
    config: StftConfig | None = None,
    length: int | None = None,
) -> np.ndarray:
    """Rebuild a waveform from a complex spectrogram by normalized overlap-add.

    Each frame is inverse-transformed, windowed again and added into the
    output; the same positions accumulate the squared window, and dividing
    the two recovers the samples exactly wherever that sum is nonzero.
    Positions where it is zero -- possible only at the outermost edges when
    the window itself vanishes there, as ``hann`` does -- are set to zero
    rather than divided by nothing.
    """
    settings = StftConfig() if config is None else config
    matrix = np.asarray(spectrogram)
    require_rank("spectrogram", matrix, 2)
    require(matrix.shape[0] >= 1, "a spectrogram needs at least one frame", ConfigError)
    require(
        matrix.shape[1] == settings.bin_count,
        f"the spectrogram has {matrix.shape[1]} bins, but this config "
        f"produces {settings.bin_count}",
        ConfigError,
    )
    window = window_vector(settings.window, settings.frame_length).astype(FEATURE_DTYPE)
    frames = np.fft.irfft(matrix, n=settings.fft_size, axis=1)[:, : settings.frame_length]
    out_length = settings.frame_length + (frames.shape[0] - 1) * settings.hop_length
    if length is not None:
        require(
            isinstance(length, int) and not isinstance(length, bool) and length >= 0,
            f"length must be a non-negative integer, got {length!r}",
            ConfigError,
        )
        require(
            length <= out_length,
            f"length {length} exceeds the {out_length} samples this spectrogram can rebuild",
            ConfigError,
        )
    accumulated = np.zeros(out_length, dtype=FEATURE_DTYPE)
    weights = np.zeros(out_length, dtype=FEATURE_DTYPE)
    for index in range(frames.shape[0]):
        start = index * settings.hop_length
        stop = start + settings.frame_length
        accumulated[start:stop] += frames[index] * window
        weights[start:stop] += window * window
    rebuilt = np.where(weights > 1e-12, accumulated / np.maximum(weights, 1e-12), 0.0)
    return rebuilt if length is None else rebuilt[:length]


def power_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """The squared magnitude of a complex spectrogram, bin for bin.

    Computed as ``real**2 + imag**2`` rather than ``abs(...) ** 2`` so no
    square root is taken only to be undone: every gain in this package is a
    ratio of powers, and this is the quantity they all read.
    """
    matrix = np.asarray(spectrogram)
    require_rank("spectrogram", matrix, 2)
    return matrix.real.astype(FEATURE_DTYPE) ** 2 + matrix.imag.astype(FEATURE_DTYPE) ** 2
