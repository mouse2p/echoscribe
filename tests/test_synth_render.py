"""Render configuration and acoustic conditions are validated up front."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.render import AcousticCondition, RenderConfig


def test_the_default_condition_is_clean() -> None:
    assert AcousticCondition().label == "clean"
    assert RenderConfig().condition.label == "clean"


def test_a_condition_label_names_every_active_stage() -> None:
    condition = AcousticCondition(noise_kind="pink", noise_snr_db=10.0, rt60_ms=300.0)
    assert condition.label == "rt60-300ms+pink-10dB"


def test_perturbation_appears_in_the_label() -> None:
    assert AcousticCondition(speed_factor=1.1).label == "speed-1.1"
    assert AcousticCondition(volume_db=-6.0).label == "volume--6dB"


def test_a_condition_round_trips_through_a_dict() -> None:
    condition = AcousticCondition(noise_kind="modulated", noise_snr_db=5.0, rt60_ms=250.0)
    assert AcousticCondition.from_dict(condition.to_dict()) == condition


def test_a_serialized_condition_rounds_at_the_production_point() -> None:
    payload = AcousticCondition(noise_snr_db=1.0 / 3.0).to_dict()
    assert payload["noise_snr_db"] == 0.333333


def test_a_condition_from_a_dict_rejects_a_missing_field() -> None:
    payload = AcousticCondition().to_dict()
    del payload["speed_factor"]
    with pytest.raises(ConfigError, match="missing 'speed_factor'"):
        AcousticCondition.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_kind": "babble"},
        {"rt60_ms": 0.0},
        {"rt60_ms": -10.0},
        {"speed_factor": 0.0},
    ],
)
def test_invalid_conditions_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        AcousticCondition(**kwargs)


def test_render_defaults_describe_a_16k_render() -> None:
    config = RenderConfig()
    assert config.sample_rate == 16000
    assert config.base_f0 == 120.0
    assert config.leading_silence_ms == 250.0
    assert config.boundary_silence_ms == 80.0
    assert config.peak_target == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": 16000.0},
        {"base_f0": 0.0},
        {"leading_silence_ms": -1.0},
        {"boundary_silence_ms": -0.5},
        {"peak_target": 0.0},
        {"peak_target": 1.5},
        {"condition": "clean"},
    ],
)
def test_invalid_render_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        RenderConfig(**kwargs)


def test_both_configs_are_frozen() -> None:
    with pytest.raises(AttributeError):
        RenderConfig().sample_rate = 8000  # type: ignore[misc]
    with pytest.raises(AttributeError):
        AcousticCondition().volume_db = 3.0  # type: ignore[misc]


def test_a_condition_is_hashable_so_it_can_key_a_cache() -> None:
    first = AcousticCondition(noise_kind="white", noise_snr_db=15.0)
    second = AcousticCondition(noise_kind="white", noise_snr_db=15.0)
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_the_dict_form_has_a_stable_key_order() -> None:
    payload = AcousticCondition(noise_kind="white").to_dict()
    assert list(payload) == [
        "noise_kind",
        "noise_snr_db",
        "rt60_ms",
        "speed_factor",
        "volume_db",
    ]


def test_a_render_config_carries_its_condition_through() -> None:
    condition = AcousticCondition(rt60_ms=400.0)
    assert RenderConfig(condition=condition).condition is condition


def test_the_default_condition_is_shared_but_immutable() -> None:
    assert RenderConfig().condition is RenderConfig().condition
    assert RenderConfig().condition.to_dict() == AcousticCondition().to_dict()


def test_a_condition_label_is_stable_for_the_same_settings() -> None:
    left = AcousticCondition(noise_kind="white", noise_snr_db=20.0, rt60_ms=150.0)
    right = AcousticCondition(noise_kind="white", noise_snr_db=20.0, rt60_ms=150.0)
    assert left.label == right.label == "rt60-150ms+white-20dB"
