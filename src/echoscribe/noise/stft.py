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
