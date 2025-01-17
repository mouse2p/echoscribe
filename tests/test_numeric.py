"""Log-domain helpers are checked against direct arithmetic where that is safe."""

from __future__ import annotations

import math

import numpy as np
import pytest

from echoscribe.errors import FeatureError
from echoscribe.numeric import (
    NEGATIVE_INFINITY,
    PROBABILITY_FLOOR,
    log_add,
    log_softmax,
    logsumexp,
    safe_log,
    softmax,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [(-1.0, -2.0), (-0.5, -0.5), (-20.0, -0.1), (-1e-9, -30.0), (0.0, -7.0)],
)
def test_log_add_matches_direct_arithmetic(left: float, right: float) -> None:
    expected = math.log(math.exp(left) + math.exp(right))
    assert math.isclose(log_add(left, right), expected, rel_tol=1e-12, abs_tol=1e-15)


def test_log_add_is_commutative() -> None:
    assert log_add(-3.0, -11.0) == log_add(-11.0, -3.0)


def test_log_add_treats_minus_infinity_as_the_identity() -> None:
    assert log_add(NEGATIVE_INFINITY, -2.5) == -2.5
    assert log_add(-2.5, NEGATIVE_INFINITY) == -2.5
    assert log_add(NEGATIVE_INFINITY, NEGATIVE_INFINITY) == NEGATIVE_INFINITY


def test_log_add_does_not_overflow_on_very_negative_scores() -> None:
    # exp(-900) underflows to 0 in float64, so a naive log(exp(a) + exp(b))
    # would return exactly -900 by luck or nan by cancellation.
    assert math.isclose(log_add(-900.0, -1000.0), -900.0, abs_tol=1e-9)


def test_log_add_keeps_the_dominant_term_when_the_gap_is_huge() -> None:
    assert log_add(-1.0, -700.0) == pytest.approx(-1.0, abs=1e-12)


def test_logsumexp_matches_the_naive_formula_for_moderate_values() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(-5.0, 0.0, size=(6, 4))
    expected = np.log(np.sum(np.exp(values), axis=1))
    assert np.allclose(logsumexp(values, axis=1), expected, rtol=1e-12)


def test_logsumexp_survives_a_large_common_offset() -> None:
    values = np.array([-800.0, -810.0, -820.0])
    expected = -800.0 + math.log1p(math.exp(-10.0) + math.exp(-20.0))
    assert math.isclose(float(logsumexp(values)), expected, abs_tol=1e-9)


def test_logsumexp_of_a_single_value_is_that_value() -> None:
    assert float(logsumexp(np.array([-3.25]))) == -3.25


def test_logsumexp_of_all_minus_infinity_is_minus_infinity() -> None:
    result = logsumexp(np.full((2, 3), NEGATIVE_INFINITY), axis=1)
    assert np.all(result == NEGATIVE_INFINITY)
    assert not np.any(np.isnan(result))


def test_logsumexp_keepdims_preserves_the_axis() -> None:
    values = np.log(np.array([[0.5, 0.5], [0.25, 0.75]]))
    result = logsumexp(values, axis=1, keepdims=True)
    assert result.shape == (2, 1)
    assert np.allclose(result, 0.0, atol=1e-12)


def test_logsumexp_without_an_axis_reduces_everything() -> None:
    values = np.log(np.full((3, 3), 1.0 / 9.0))
    assert float(logsumexp(values)) == pytest.approx(0.0, abs=1e-12)


def test_logsumexp_is_bounded_by_max_and_max_plus_log_n() -> None:
    rng = np.random.default_rng(3)
    values = rng.uniform(-12.0, -1.0, size=(20, 7))
    result = logsumexp(values, axis=1)
    maximum = values.max(axis=1)
    assert np.all(result >= maximum - 1e-12)
    assert np.all(result <= maximum + math.log(7) + 1e-12)


def test_log_softmax_rows_normalise_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    result = np.exp(log_softmax(logits, axis=-1))
    assert np.allclose(result.sum(axis=-1), 1.0, atol=1e-12)


def test_softmax_is_log_softmax_exponentiated() -> None:
    logits = np.array([[-4.0, 0.5, 2.0]])
    assert np.allclose(softmax(logits), np.exp(log_softmax(logits)))


def test_softmax_of_equal_logits_is_uniform() -> None:
    assert np.allclose(softmax(np.zeros((2, 5))), 0.2)


def test_softmax_preserves_the_order_of_the_logits() -> None:
    logits = np.array([[3.0, -1.0, 0.5, 2.0]])
    result = softmax(logits)[0]
    assert list(np.argsort(-result)) == list(np.argsort(-logits[0]))


def test_log_softmax_handles_a_fully_masked_row() -> None:
    logits = np.array([[NEGATIVE_INFINITY, NEGATIVE_INFINITY], [0.0, -1.0]])
    result = log_softmax(logits, axis=-1)
    assert np.all(np.isneginf(result[0]))
    assert not np.any(np.isnan(result))


def test_softmax_normalises_along_the_requested_axis() -> None:
    logits = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert np.allclose(softmax(logits, axis=0).sum(axis=0), 1.0)
    assert np.allclose(softmax(logits, axis=1).sum(axis=1), 1.0)


def test_softmax_output_is_float64_even_for_float32_input() -> None:
    assert softmax(np.zeros((2, 2), dtype=np.float32)).dtype == np.float64
    assert log_softmax(np.zeros((2, 2), dtype=np.float32)).dtype == np.float64


def test_masked_rows_do_not_emit_numpy_warnings() -> None:
    import warnings

    values = np.full((2, 3), NEGATIVE_INFINITY)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(logsumexp(values, axis=1) == NEGATIVE_INFINITY)
        assert np.all(log_softmax(values, axis=1) == NEGATIVE_INFINITY)


def test_safe_log_of_one_is_zero() -> None:
    assert float(safe_log(np.array([1.0]))[0]) == 0.0


def test_safe_log_clamps_exact_zeros_to_the_floor() -> None:
    result = safe_log(np.array([0.0]))
    assert np.all(np.isfinite(result))
    assert float(result[0]) == pytest.approx(math.log(PROBABILITY_FLOOR), rel=1e-12)


def test_safe_log_passes_values_above_the_floor_through() -> None:
    values = np.array([0.25, 0.5, 2.0])
    assert np.allclose(safe_log(values), np.log(values))


def test_safe_log_is_monotonic() -> None:
    result = safe_log(np.linspace(0.0, 1.0, 32))
    assert np.all(np.diff(result) >= 0.0)


def test_safe_log_rejects_negative_input_and_a_bad_floor() -> None:
    with pytest.raises(FeatureError, match="non-negative"):
        safe_log(np.array([-1.0]))
    with pytest.raises(ValueError, match="floor must be positive"):
        safe_log(np.array([1.0]), floor=0.0)


def test_safe_log_then_softmax_recovers_log_softmax() -> None:
    logits = np.array([[2.0, -1.0, 0.0], [-5.0, 5.0, 0.0]])
    assert np.allclose(safe_log(softmax(logits)), log_softmax(logits), atol=1e-12)
