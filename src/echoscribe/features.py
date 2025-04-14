"""Log-mel filterbank and MFCC extraction on top of the framing and spectrum stages.

``FeatureConfig`` deliberately does not repeat the sample rate: it is read from the
nested :class:`~echoscribe.framing.FrameConfig`, because a mismatch between the two
would silently rescale every mel bin.  One number, one place.

The output of :func:`extract_log_mel` is natural-log power in mel bins, which is
what a CTC or transducer head consumes.  Values are floored before the logarithm
so a silent frame yields a large finite negative number rather than ``-inf``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require, require_in, require_positive
from echoscribe.framing import FrameConfig, frame_signal
from echoscribe.mel import MEL_SCALES, cached_filterbank
from echoscribe.numeric import safe_log
from echoscribe.spectrum import power_spectrum
from echoscribe.types import FloatArray
from echoscribe.waveform import Audio


@dataclass(frozen=True)
class FeatureConfig:
    """Everything needed to turn a waveform into log-mel frames."""

    num_mel_bins: int = 80
    frame: FrameConfig = FrameConfig()
    low_frequency: float = 20.0
    high_frequency: float | None = None
    mel_scale: str = "slaney"
    normalize_filterbank: bool = False
    log_floor: float = 1e-10

    def __post_init__(self) -> None:
        require(
            isinstance(self.num_mel_bins, int)
            and not isinstance(self.num_mel_bins, bool)
            and self.num_mel_bins >= 1,
            f"num_mel_bins must be a positive integer, got {self.num_mel_bins!r}",
            ConfigError,
        )
        require_in("mel_scale", self.mel_scale, MEL_SCALES, ConfigError)
        require_positive("log_floor", self.log_floor, ConfigError)
        require(
            self.low_frequency >= 0.0,
            f"low_frequency must not be negative, got {self.low_frequency!r}",
            ConfigError,
        )
        if self.high_frequency is not None:
            require(
                self.low_frequency < self.high_frequency <= self.sample_rate / 2.0,
                f"high_frequency must lie in ({self.low_frequency}, {self.sample_rate / 2.0}], "
                f"got {self.high_frequency!r}",
                ConfigError,
            )

    @property
    def sample_rate(self) -> int:
        """Sample rate the features are computed at, taken from the frame config."""
        return self.frame.sample_rate

    @property
    def fft_size(self) -> int:
        """FFT length each frame is transformed at."""
        return self.frame.fft_size

    @property
    def resolved_high_frequency(self) -> float:
        """The high cut, defaulting to Nyquist when unset."""
        return self.sample_rate / 2.0 if self.high_frequency is None else self.high_frequency


def filterbank_matrix(config: FeatureConfig) -> np.ndarray:
    """The cached ``(num_mel_bins, fft_size // 2 + 1)`` matrix for ``config``."""
    return cached_filterbank(
        config.num_mel_bins,
        config.fft_size,
        config.sample_rate,
        config.low_frequency,
        config.high_frequency,
        config.mel_scale,
        config.normalize_filterbank,
    )


def mel_energies(audio: Audio | np.ndarray, config: FeatureConfig) -> FloatArray:
    """Linear mel-band energies per frame, before the logarithm."""
    frames = frame_signal(audio, config.frame)
    power = power_spectrum(frames, config.fft_size)
    return power @ filterbank_matrix(config).T


def extract_log_mel(audio: Audio | np.ndarray, config: FeatureConfig) -> FloatArray:
    """Return the log-mel features of ``audio`` as ``(frames, num_mel_bins)``.

    The logarithm is natural and floored at ``config.log_floor``.  A floor of
    ``1e-10`` maps to about ``-23``, which is far enough below any speech energy
    to be effectively silent without letting ``-inf`` propagate into a model.
    """
    return safe_log(mel_energies(audio, config), config.log_floor)


def dct_matrix(num_output: int, num_input: int) -> np.ndarray:
    """Return the orthonormal DCT-II matrix of shape ``(num_output, num_input)``.

    Row ``k`` is ``sqrt(2 / N) * cos(pi * k * (2n + 1) / (2N))`` with the DC row
    additionally scaled by ``1 / sqrt(2)`` so the whole matrix is orthonormal when
    ``num_output == num_input``.  MFCCs are the first few rows of that transform
    applied to log-mel energies: the cepstrum decorrelates the bands, which is why
    a diagonal-covariance model can use them.
    """
    require_positive("num_output", num_output, ConfigError)
    require_positive("num_input", num_input, ConfigError)
    require(
        num_output <= num_input,
        f"cannot take {num_output} cepstral coefficients from {num_input} mel bins",
        ConfigError,
    )
    rows = np.arange(num_output, dtype=np.float64)[:, None]
    columns = np.arange(num_input, dtype=np.float64)[None, :]
    matrix = np.cos(np.pi * rows * (2.0 * columns + 1.0) / (2.0 * num_input))
    matrix *= np.sqrt(2.0 / num_input)
    matrix[0] /= np.sqrt(2.0)
    return matrix


def extract_mfcc(
    audio: Audio | np.ndarray,
    config: FeatureConfig,
    num_coefficients: int = 13,
) -> FloatArray:
    """Return the first ``num_coefficients`` DCT-II coefficients of the log-mel."""
    log_mel = extract_log_mel(audio, config)
    return log_mel @ dct_matrix(num_coefficients, config.num_mel_bins).T
