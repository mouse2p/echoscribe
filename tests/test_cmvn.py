"""Normalisation statistics are checked against direct numpy reductions."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.cmvn import CmvnStatistics, apply_cmvn
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


def _stats_of(features: np.ndarray) -> CmvnStatistics:
    stats = CmvnStatistics(dimension=features.shape[1])
    stats.update(features)
    return stats


def test_applying_statistics_of_the_same_frames_standardises_them() -> None:
    rng = np.random.default_rng(5)
    features = rng.standard_normal((200, 13)) * 3.0 + 7.0
    normalised = apply_cmvn(features, _stats_of(features))
    assert np.allclose(normalised.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(normalised.std(axis=0), 1.0, atol=1e-9)


def test_mean_only_normalisation_keeps_the_scale() -> None:
    rng = np.random.default_rng(6)
    features = rng.standard_normal((100, 4)) * 5.0 - 2.0
    normalised = apply_cmvn(features, _stats_of(features), normalize_variance=False)
    assert np.allclose(normalised.mean(axis=0), 0.0, atol=1e-12)
    # Mean-only CMVN must leave the empirical spread exactly where it was.
    assert np.allclose(normalised.std(axis=0), features.std(axis=0), atol=1e-9)


def test_empty_statistics_leave_the_features_untouched() -> None:
    features = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(
        apply_cmvn(features, CmvnStatistics(dimension=4), normalize_variance=False), features
    )


def test_a_dead_band_is_not_amplified() -> None:
    features = np.zeros((20, 2))
    features[:, 1] = np.linspace(-1.0, 1.0, 20)
    normalised = apply_cmvn(features, _stats_of(features))
    assert np.all(normalised[:, 0] == 0.0)
    assert np.all(np.isfinite(normalised))


def test_apply_cmvn_rejects_a_width_mismatch() -> None:
    with pytest.raises(EvalError, match="statistics track 5"):
        apply_cmvn(np.zeros((3, 4)), CmvnStatistics(dimension=5))


def test_merging_matches_one_pass_over_the_concatenation() -> None:
    rng = np.random.default_rng(9)
    left = rng.standard_normal((40, 6))
    right = rng.standard_normal((25, 6)) * 2.0 + 1.0
    merged = _stats_of(left).merge(_stats_of(right))
    reference = _stats_of(np.vstack([left, right]))
    assert merged.count == reference.count == 65
    assert np.allclose(merged.mean(), reference.mean())
    assert np.allclose(merged.variance(), reference.variance(), atol=1e-12)


def test_merging_rejects_a_dimension_mismatch() -> None:
    with pytest.raises(ConfigError, match="cannot merge statistics of 3 and 4"):
        CmvnStatistics(dimension=3).merge(CmvnStatistics(dimension=4))
