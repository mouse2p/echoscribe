"""Normalisation statistics are checked against direct numpy reductions."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.cmvn import CmvnStatistics, apply_cmvn, sliding_window_cmvn
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


def test_a_window_of_one_leaves_every_frame_at_zero() -> None:
    rng = np.random.default_rng(2)
    features = rng.standard_normal((25, 5)) * 4.0 + 9.0
    normalised = sliding_window_cmvn(features, 1)
    # A one-frame window has zero variance, so the numerator is the cumulative
    # sum's rounding residue and the denominator is the variance floor.  The
    # result is numerically zero but not bit zero.
    assert np.max(np.abs(normalised)) < 1e-4


def test_a_full_length_window_matches_global_cmvn_at_the_last_frame() -> None:
    rng = np.random.default_rng(3)
    features = rng.standard_normal((40, 6)) * 2.0 - 3.0
    sliding = sliding_window_cmvn(features, 40)
    global_ = apply_cmvn(features, _stats_of(features))
    # Only the final frame has a window that spans the whole utterance; earlier
    # frames see a shorter causal window by construction.
    assert np.allclose(sliding[-1], global_[-1], atol=1e-9)


def test_the_window_is_causal_so_a_future_change_does_not_move_the_past() -> None:
    rng = np.random.default_rng(4)
    features = rng.standard_normal((60, 4))
    before = sliding_window_cmvn(features, 10)
    altered = features.copy()
    altered[55:] *= 100.0
    after = sliding_window_cmvn(altered, 10)
    # Frames up to 45 have a window that ends before the alteration starts.
    assert np.allclose(before[:46], after[:46], atol=1e-12)
    assert not np.allclose(before[56:], after[56:])


def test_each_frame_is_standardised_over_its_own_window() -> None:
    rng = np.random.default_rng(8)
    features = rng.standard_normal((120, 5)) * 3.0 + 2.0
    normalised = sliding_window_cmvn(features, 30)
    for index in range(30, 120):
        window = features[index - 29 : index + 1]
        expected = (features[index] - window.mean(axis=0)) / window.std(axis=0)
        assert np.allclose(normalised[index], expected, atol=1e-8)


def test_mean_only_sliding_normalisation_subtracts_the_local_mean() -> None:
    rng = np.random.default_rng(10)
    features = rng.standard_normal((80, 3)) * 4.0
    normalised = sliding_window_cmvn(features, 20, normalize_variance=False)
    for index in range(20, 80):
        window_mean = features[index - 19 : index + 1].mean(axis=0)
        assert np.allclose(normalised[index], features[index] - window_mean, atol=1e-8)


def test_sliding_normalisation_leaves_the_local_spread_at_one() -> None:
    rng = np.random.default_rng(12)
    features = rng.standard_normal((90, 4)) * 6.0 - 1.0
    normalised = sliding_window_cmvn(features, 25)
    for index in range(25, 90):
        window = normalised[index - 24 : index + 1]
        # Each frame is scaled by its own window, so the window is not exactly
        # unit variance; it is close, and never the 6.0 it started at.
        assert 0.5 < float(window.std()) < 2.0


def test_sliding_normalisation_of_no_frames_is_empty() -> None:
    assert sliding_window_cmvn(np.zeros((0, 4)), 10).shape == (0, 4)


def test_sliding_normalisation_rejects_a_bad_window() -> None:
    with pytest.raises(ConfigError, match="window_frames must be a positive integer"):
        sliding_window_cmvn(np.zeros((4, 2)), 0)


def test_a_constant_stream_normalises_to_zero() -> None:
    features = np.full((50, 3), -7.5)
    assert np.allclose(sliding_window_cmvn(features, 10), 0.0, atol=1e-9)
