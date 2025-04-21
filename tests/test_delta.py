"""Deltas are the fitted slope, so a ramp and a parabola pin the behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.delta import (
    DeltaConfig,
    delta_features,
    derivative_orders,
    stack_features,
)
from echoscribe.errors import ConfigError


def test_the_delta_of_a_linear_ramp_is_its_slope() -> None:
    features = np.arange(20, dtype=np.float64)[:, None] * 0.5
    deltas = delta_features(features, DeltaConfig(window=2))
    assert np.allclose(deltas[2:-2], 0.5, atol=1e-12)


def test_the_delta_of_a_constant_is_zero() -> None:
    features = np.full((15, 4), 3.25)
    assert np.allclose(delta_features(features, DeltaConfig(window=3)), 0.0, atol=1e-12)


def test_the_delta_preserves_the_frame_count_and_width() -> None:
    features = np.random.default_rng(1).standard_normal((37, 11))
    assert delta_features(features).shape == (37, 11)


def test_interior_deltas_of_a_unit_ramp_are_exactly_one() -> None:
    features = np.arange(10, dtype=np.float64)[:, None]
    deltas = delta_features(features, DeltaConfig(window=2))
    assert np.allclose(deltas[2:-2], 1.0, atol=1e-12)


def test_replicated_edges_attenuate_the_fitted_slope() -> None:
    # Padding replicates the boundary sample instead of extrapolating, so the
    # regression sees a flattened trajectory and reports a smaller slope.  Zero
    # padding would invent a step and overshoot instead.
    features = np.arange(10, dtype=np.float64)[:, None]
    deltas = delta_features(features, DeltaConfig(window=2))
    assert deltas[0, 0] == pytest.approx(0.5, abs=1e-12)
    assert deltas[-1, 0] == pytest.approx(0.5, abs=1e-12)


def test_a_wider_window_smooths_a_noisy_ramp() -> None:
    rng = np.random.default_rng(2)
    clean = np.arange(200, dtype=np.float64)[:, None] * 0.1
    noisy = clean + rng.standard_normal((200, 1)) * 0.5
    narrow = delta_features(noisy, DeltaConfig(window=1))
    wide = delta_features(noisy, DeltaConfig(window=4))
    assert wide[5:-5].std() < narrow[5:-5].std()
    assert np.allclose(wide.mean(), 0.1, atol=0.05)


def test_the_delta_of_no_frames_is_empty() -> None:
    assert delta_features(np.zeros((0, 5))).shape == (0, 5)


def test_the_denominator_is_twice_the_sum_of_squares() -> None:
    assert DeltaConfig(window=1).denominator == 2.0
    assert DeltaConfig(window=2).denominator == 10.0
    assert DeltaConfig(window=3).denominator == 28.0


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"window": -2}, {"order": 0}, {"window": 1.5}])
def test_invalid_delta_configs_are_rejected(kwargs) -> None:
    with pytest.raises(ConfigError):
        DeltaConfig(**kwargs)


def test_delta_rejects_a_flat_vector() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        delta_features(np.zeros(10))


def test_delta_is_deterministic_and_leaves_its_input_alone() -> None:
    features = np.random.default_rng(3).standard_normal((30, 6))
    copy = features.copy()
    first = delta_features(features, DeltaConfig(window=2))
    assert np.array_equal(delta_features(features, DeltaConfig(window=2)), first)
    assert np.array_equal(features, copy)


def test_derivative_orders_returns_statics_first() -> None:
    features = np.random.default_rng(4).standard_normal((24, 5))
    orders = derivative_orders(features, DeltaConfig(window=2, order=2))
    assert len(orders) == 3
    assert np.array_equal(orders[0], features)
    assert np.array_equal(orders[1], delta_features(features, DeltaConfig(window=2)))


def test_a_first_order_config_yields_two_orders() -> None:
    assert len(derivative_orders(np.zeros((5, 2)), DeltaConfig(order=1))) == 2


def test_stacking_multiplies_the_width() -> None:
    features = np.random.default_rng(5).standard_normal((30, 13))
    for order in (1, 2, 3):
        config = DeltaConfig(window=2, order=order)
        assert stack_features(features, config).shape == (30, 13 * config.stacked_width_factor)


def test_stacking_keeps_the_frame_count() -> None:
    features = np.random.default_rng(6).standard_normal((17, 8))
    assert stack_features(features).shape[0] == 17


def test_stacked_statics_are_the_leading_columns() -> None:
    features = np.random.default_rng(7).standard_normal((12, 4))
    stacked = stack_features(features, DeltaConfig(order=2))
    assert np.array_equal(stacked[:, :4], features)


def test_a_parabola_has_a_linear_delta_and_a_constant_second_delta() -> None:
    t = np.arange(-20, 21, dtype=np.float64)
    features = (3.0 * t**2)[:, None]
    config = DeltaConfig(window=2, order=2)
    stacked = stack_features(features, config)
    interior = slice(config.window * 2, len(t) - config.window * 2)
    assert np.allclose(stacked[interior, 1], 6.0 * t[interior], atol=1e-9)
    assert np.allclose(stacked[interior, 2], 6.0, atol=1e-9)
