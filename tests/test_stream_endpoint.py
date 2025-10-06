"""Endpointing is checked on the two failures it exists to avoid."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError, StreamError
from echoscribe.stream.endpoint import EnergyEndpointConfig, EnergyEndpointer

LOUD = 0.4
QUIET = 0.0


def _run(energies, config=None):
    """Push a whole energy trace and return the frame the endpoint fired on."""
    endpointer = EnergyEndpointer(config)
    for index, energy in enumerate(energies):
        if endpointer.push_energy(energy):
            return index
    return None


def test_leading_silence_never_fires_an_endpoint() -> None:
    assert _run([QUIET] * 200) is None


def test_speech_alone_never_fires_an_endpoint() -> None:
    assert _run([LOUD] * 100) is None


def test_a_short_pause_inside_speech_does_not_fire() -> None:
    trace = [LOUD] * 30 + [QUIET] * 10 + [LOUD] * 30
    assert _run(trace) is None


def test_a_long_trailing_silence_fires_after_the_hangover() -> None:
    trace = [LOUD] * 30 + [QUIET] * 40
    fired = _run(trace, EnergyEndpointConfig(hangover_frames=20, min_speech_frames=5))
    assert fired == 30 + 19


def test_the_hangover_length_controls_when_it_fires() -> None:
    trace = [LOUD] * 10 + [QUIET] * 60
    for hangover in (5, 12, 30):
        config = EnergyEndpointConfig(hangover_frames=hangover)
        assert _run(trace, config) == 10 + hangover - 1


def test_min_speech_frames_delays_the_first_possible_endpoint() -> None:
    trace = [LOUD] * 3 + [QUIET] * 60
    assert _run(trace, EnergyEndpointConfig(min_speech_frames=5)) is None
    assert _run([LOUD] * 5 + [QUIET] * 60, EnergyEndpointConfig(min_speech_frames=5)) == 5 + 19


def test_the_endpointer_latches_after_firing() -> None:
    endpointer = EnergyEndpointer(EnergyEndpointConfig(hangover_frames=3, min_speech_frames=1))
    for _ in range(10):
        endpointer.push_energy(LOUD)
    fired = [endpointer.push_energy(QUIET) for _ in range(10)]
    assert sum(fired) == 1
    assert endpointer.has_fired
    assert endpointer.fired_at == 12


def test_a_reset_clears_the_latch() -> None:
    endpointer = EnergyEndpointer(EnergyEndpointConfig(hangover_frames=2, min_speech_frames=1))
    endpointer.push_energy(LOUD)
    endpointer.push_energy(QUIET)
    endpointer.push_energy(QUIET)
    assert endpointer.has_fired
    endpointer.reset()
    assert not endpointer.has_fired
    assert endpointer.fired_at is None
    assert endpointer.speech_frames == 0
    assert not endpointer.in_speech


def test_push_frame_uses_the_rms_of_the_samples() -> None:
    config = EnergyEndpointConfig(threshold=0.1, hangover_frames=1, min_speech_frames=0)
    endpointer = EnergyEndpointer(config)
    assert not endpointer.push_frame(np.full(160, 0.5))
    assert endpointer.in_speech
    assert endpointer.push_frame(np.zeros(160))


def test_push_frame_of_an_empty_frame_counts_as_silence() -> None:
    config = EnergyEndpointConfig(threshold=0.1, hangover_frames=1, min_speech_frames=0)
    endpointer = EnergyEndpointer(config)
    endpointer.push_frame(np.full(160, 0.5))
    assert endpointer.push_frame(np.zeros(0))


def test_push_frame_rejects_a_matrix() -> None:
    with pytest.raises(StreamError, match="one frame of samples"):
        EnergyEndpointer().push_frame(np.zeros((4, 2)))


def test_a_negative_energy_is_refused() -> None:
    with pytest.raises(StreamError, match="must not be negative"):
        EnergyEndpointer().push_energy(-0.1)


def test_the_threshold_is_inclusive() -> None:
    config = EnergyEndpointConfig(threshold=0.2, hangover_frames=1, min_speech_frames=0)
    endpointer = EnergyEndpointer(config)
    assert not endpointer.push_energy(0.2)
    assert endpointer.in_speech


@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": 0.0}, {"hangover_frames": 0}, {"min_speech_frames": -1}],
)
def test_invalid_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        EnergyEndpointConfig(**kwargs)
