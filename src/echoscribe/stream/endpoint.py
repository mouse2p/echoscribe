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

from echoscribe.errors import ConfigError, StreamError, require, require_positive


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
