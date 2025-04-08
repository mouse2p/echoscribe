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

from echoscribe.errors import ConfigError, require, require_in, require_positive
from echoscribe.framing import FrameConfig
from echoscribe.mel import MEL_SCALES


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
