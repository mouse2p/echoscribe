"""Log-domain helpers are checked against direct arithmetic where that is safe."""

from __future__ import annotations

import math

import pytest

from echoscribe.numeric import NEGATIVE_INFINITY, log_add


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
