"""Normalisation statistics are checked against direct numpy reductions."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.cmvn import CmvnStatistics
from echoscribe.errors import ConfigError, EvalError


def test_fresh_statistics_have_a_zero_mean_and_variance() -> None:
    stats = CmvnStatistics(dimension=5)
    assert stats.count == 0
    assert np.all(stats.mean() == 0.0)
    assert np.all(stats.variance() == 0.0)


def test_statistics_reject_a_non_positive_dimension() -> None:
    with pytest.raises(ConfigError, match="dimension must be positive"):
        CmvnStatistics(dimension=0)


def test_update_tracks_the_count_and_the_sums() -> None:
    rng = np.random.default_rng(1)
    features = rng.standard_normal((30, 7))
    stats = CmvnStatistics(dimension=7)
    stats.update(features[:12])
    stats.update(features[12:])
    assert stats.count == 30
    assert np.allclose(stats.mean(), features.mean(axis=0))
    assert np.allclose(stats.variance(), features.var(axis=0), atol=1e-12)


def test_a_constant_stream_has_zero_variance() -> None:
    stats = CmvnStatistics(dimension=3)
    stats.update(np.full((10, 3), 2.5))
    assert np.allclose(stats.mean(), 2.5)
    assert np.all(stats.variance() == 0.0)


def test_update_rejects_a_width_mismatch() -> None:
    stats = CmvnStatistics(dimension=4)
    with pytest.raises(EvalError, match="features have 5 columns"):
        stats.update(np.zeros((2, 5)))


def test_statistics_reject_a_mismatched_accumulator() -> None:
    with pytest.raises(ConfigError, match="total must have shape"):
        CmvnStatistics(dimension=4, total=np.zeros(3))


def test_a_single_frame_has_zero_variance() -> None:
    stats = CmvnStatistics(dimension=2)
    stats.update(np.array([[1.0, -2.0]]))
    assert np.allclose(stats.mean(), [1.0, -2.0])
    assert np.all(stats.variance() == 0.0)


def test_variance_never_goes_negative_from_cancellation() -> None:
    # A large mean with a tiny spread is where E[x^2] - E[x]^2 loses precision.
    stats = CmvnStatistics(dimension=1)
    stats.update(np.full((1000, 1), 1e6) + np.linspace(-1e-6, 1e-6, 1000)[:, None])
    assert stats.variance()[0] >= 0.0
