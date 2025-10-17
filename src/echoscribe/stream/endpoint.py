"""Endpointing: deciding that the speaker has stopped.

Two independent detectors are provided, because they fail in opposite ways and a
real system runs both:

:class:`EnergyEndpointer`
    watches frame energy.  It is cheap, needs no model, and is fooled by any loud
    non-speech -- a door, a keyboard, background music.
:class:`BlankRatioEndpointer`
    watches the decoder's own blank posterior.  It is fooled by a quiet speaker
    rather than by a loud room, and it needs a model to exist at all.

Both are edge-triggered state machines with the same two guards, and the guards are
the part that matters:

``min_speech_frames``
    an endpoint may not fire before the utterance has actually started.  Without
    this, the leading silence of every recording fires an endpoint at frame twenty
    and the recognizer transcribes nothing.
``hangover_frames``
    a run of quiet has to last this long before firing.  Speech is full of short
    energy dips -- between words, inside plosives -- and a detector without hangover
    cuts utterances in half at every one of them.

Both are latching: once fired, an endpointer stays fired until it is reset.  That is
what makes "the endpoint fired" a fact about the utterance rather than a fact about
the frame that happened to be quiet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoscribe.errors import (
    ConfigError,
    StreamError,
    require,
    require_positive,
    require_range,
)


@dataclass(frozen=True)
class EnergyEndpointConfig:
    """Thresholds for the energy-based detector."""

    threshold: float = 0.01
    hangover_frames: int = 20
    min_speech_frames: int = 5

    def __post_init__(self) -> None:
        require_positive("threshold", self.threshold, ConfigError)
        require_positive("hangover_frames", self.hangover_frames, ConfigError)
        require(
            self.min_speech_frames >= 0,
            f"min_speech_frames must not be negative, got {self.min_speech_frames}",
            ConfigError,
        )


class EnergyEndpointer:
    """Fires once a run of quiet frames follows enough speech."""

    def __init__(self, config: EnergyEndpointConfig | None = None) -> None:
        self.config = EnergyEndpointConfig() if config is None else config
        self._quiet_run = 0
        self._speech_frames = 0
        self._started = False
        self._fired = False
        self._fired_at: int | None = None
        self._frames = 0

    @property
    def has_fired(self) -> bool:
        """True once the endpoint has fired and not been reset."""
        return self._fired

    @property
    def fired_at(self) -> int | None:
        """Frame index the endpoint fired on, or ``None``."""
        return self._fired_at

    @property
    def in_speech(self) -> bool:
        """True once at least one frame above the threshold has been seen."""
        return self._started

    @property
    def speech_frames(self) -> int:
        """Frames seen above the threshold so far."""
        return self._speech_frames

    @property
    def quiet_run(self) -> int:
        """Consecutive quiet frames seen so far."""
        return self._quiet_run

    def push_energy(self, energy: float) -> bool:
        """Feed one frame's energy; return True on the frame the endpoint fires."""
        require(energy >= 0.0, f"frame energy must not be negative, got {energy!r}", StreamError)
        self._frames += 1
        if self._fired:
            return False
        if energy >= self.config.threshold:
            self._speech_frames += 1
            self._started = True
            self._quiet_run = 0
            return False
        self._quiet_run += 1
        if not self._started or self._speech_frames < self.config.min_speech_frames:
            return False
        if self._quiet_run < self.config.hangover_frames:
            return False
        self._fired = True
        self._fired_at = self._frames - 1
        return True

    def push_frame(self, samples: np.ndarray) -> bool:
        """Feed one frame of audio, using its RMS as the energy."""
        values = np.asarray(samples, dtype=np.float64)
        require(
            values.ndim == 1,
            f"push_frame needs one frame of samples, got shape {values.shape}",
            StreamError,
        )
        energy = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
        return self.push_energy(energy)

    def reset(self) -> None:
        """Clear the latch and every counter."""
        self._quiet_run = 0
        self._speech_frames = 0
        self._started = False
        self._fired = False
        self._fired_at = None
        self._frames = 0


@dataclass(frozen=True)
class BlankRatioConfig:
    """Thresholds for the decoder-driven detector."""

    blank_id: int = 0
    window_frames: int = 20
    ratio_threshold: float = 0.9
    min_speech_frames: int = 5

    def __post_init__(self) -> None:
        require(
            self.blank_id >= 0,
            f"blank_id must not be negative, got {self.blank_id}",
            ConfigError,
        )
        require_positive("window_frames", self.window_frames, ConfigError)
        require_range("ratio_threshold", self.ratio_threshold, 0.0, 1.0, ConfigError)
        require(
            self.min_speech_frames >= 0,
            f"min_speech_frames must not be negative, got {self.min_speech_frames}",
            ConfigError,
        )


class BlankRatioEndpointer:
    """Fires when the decoder has been emitting blanks for a whole window.

    This is the detector a transducer system actually trusts, because it reads the
    model's own decision rather than the signal's loudness: a speaker who trails off
    into a whisper still produces confident blanks, and a loud room does not.

    The window has to be *full* before the ratio is believed.  Judging on the first
    few frames of an utterance would fire on the leading silence for exactly the
    reason the energy detector guards against.
    """

    def __init__(self, config: BlankRatioConfig | None = None) -> None:
        self.config = BlankRatioConfig() if config is None else config
        self._window: list[float] = []
        self._speech_frames = 0
        self._fired = False
        self._fired_at: int | None = None
        self._frames = 0

    @property
    def has_fired(self) -> bool:
        """True once the endpoint has fired and not been reset."""
        return self._fired

    @property
    def fired_at(self) -> int | None:
        """Frame index the endpoint fired on, or ``None``."""
        return self._fired_at

    @property
    def blank_ratio(self) -> float:
        """Mean blank probability over the frames seen so far in the window."""
        return float(np.mean(self._window)) if self._window else 0.0

    @property
    def speech_frames(self) -> int:
        """Frames whose best symbol was not the blank."""
        return self._speech_frames

    def push_posteriors(self, row: np.ndarray) -> bool:
        """Feed one frame of posterior probabilities; True on the firing frame."""
        values = np.asarray(row, dtype=np.float64)
        require(
            values.ndim == 1 and values.size > self.config.blank_id,
            f"a posterior row must hold at least {self.config.blank_id + 1} symbols, "
            f"got shape {values.shape}",
            StreamError,
        )
        require(
            bool(np.all(values >= -1e-9)),
            "push_posteriors expects probabilities, not log-probabilities",
            StreamError,
        )
        self._frames += 1
        blank = float(values[self.config.blank_id])
        if int(np.argmax(values)) != self.config.blank_id:
            self._speech_frames += 1
        self._window.append(blank)
        if len(self._window) > self.config.window_frames:
            self._window.pop(0)
        if self._fired:
            return False
        if len(self._window) < self.config.window_frames:
            return False
        if self._speech_frames < self.config.min_speech_frames:
            return False
        if self.blank_ratio < self.config.ratio_threshold:
            return False
        self._fired = True
        self._fired_at = self._frames - 1
        return True

    def reset(self) -> None:
        """Clear the latch, the window and the counters."""
        self._window = []
        self._speech_frames = 0
        self._fired = False
        self._fired_at = None
        self._frames = 0
