"""The room model is pinned through its decay law, which is the only claim it makes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.room import RT60_DECAY_CONSTANT, RoomConfig, decay_envelope

RATE = 16000


def test_the_decay_constant_is_ln_of_a_thousand() -> None:
    assert RT60_DECAY_CONSTANT == pytest.approx(math.log(1000.0))


def test_the_envelope_starts_at_one() -> None:
    assert decay_envelope(200.0, RATE, 8)[0] == pytest.approx(1.0)


def test_the_envelope_reaches_minus_sixty_db_at_rt60() -> None:
    envelope = decay_envelope(200.0, RATE, RATE)
    assert envelope[int(0.2 * RATE)] == pytest.approx(1e-3, rel=1e-6)


def test_a_longer_rt60_decays_more_slowly() -> None:
    short = decay_envelope(100.0, RATE, RATE)
    long = decay_envelope(600.0, RATE, RATE)
    assert long[RATE // 2] > short[RATE // 2]


def test_the_envelope_is_strictly_decreasing() -> None:
    envelope = decay_envelope(300.0, RATE, 4096)
    assert np.all(np.diff(envelope) < 0.0)


def test_the_envelope_of_no_samples_is_empty() -> None:
    assert decay_envelope(200.0, RATE, 0).shape == (0,)


def test_the_envelope_rejects_bad_arguments() -> None:
    with pytest.raises(ConfigError, match="rt60_ms must be positive"):
        decay_envelope(0.0, RATE, 16)
    with pytest.raises(ConfigError, match="num_samples must be a non-negative integer"):
        decay_envelope(200.0, RATE, -1)


def test_config_defaults_describe_a_small_room() -> None:
    config = RoomConfig()
    assert config.rt60_ms == 200.0
    assert config.direct_gain == 1.0
    assert config.tail_gain == 0.3
    assert config.early_reflections == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rt60_ms": 0.0},
        {"direct_gain": 0.0},
        {"tail_gain": -1.0},
        {"early_reflections": -1},
        {"seed": 7},
    ],
)
def test_invalid_room_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        RoomConfig(**kwargs)
