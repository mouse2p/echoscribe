"""The posterior fixtures are what every decoder test is built on, so they are pinned."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.posteriors import (
    SCRIPTED_BACKGROUND_LOGIT,
    SCRIPTED_PEAK_LOGIT,
    check_log_posteriors,
    frame_index,
    frame_times,
    masked_posteriors,
    scripted_posteriors,
    to_log_posteriors,
)
from echoscribe.errors import DecodeError
from echoscribe.numeric import logsumexp


def test_the_scripted_argmax_is_the_planted_path() -> None:
    path = [0, 3, 3, 1, 0]
    matrix = scripted_posteriors(path)
    assert matrix.shape == (5, 4)
    assert [int(value) for value in matrix.argmax(axis=-1)] == path


def test_rows_are_normalised_log_distributions() -> None:
    matrix = scripted_posteriors([1, 2, 1], num_symbols=5)
    totals = np.exp(matrix).sum(axis=-1)
    assert np.allclose(totals, 1.0, atol=1e-12)
    assert np.all(matrix <= 0.0)


def test_the_planted_symbol_dominates_its_frame() -> None:
    matrix = scripted_posteriors([2], num_symbols=6)
    planted = matrix[0, 2]
    others = np.delete(matrix[0], 2)
    assert planted > np.max(others)
    assert planted - np.max(others) > SCRIPTED_PEAK_LOGIT + SCRIPTED_BACKGROUND_LOGIT


def test_an_explicit_width_pads_the_alphabet() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=10)
    assert matrix.shape == (2, 10)


def test_the_default_width_is_one_past_the_largest_symbol() -> None:
    assert scripted_posteriors([0, 4, 2]).shape == (3, 5)


def test_a_repeated_symbol_is_fine() -> None:
    matrix = scripted_posteriors([1, 1, 1], num_symbols=3)
    assert [int(value) for value in matrix.argmax(axis=-1)] == [1, 1, 1]


def test_an_empty_path_is_refused() -> None:
    with pytest.raises(DecodeError, match="at least one frame"):
        scripted_posteriors([])


def test_a_path_that_does_not_fit_the_alphabet_is_refused() -> None:
    with pytest.raises(DecodeError, match="cannot hold symbol id"):
        scripted_posteriors([0, 7], num_symbols=4)


def test_a_negative_symbol_is_refused() -> None:
    with pytest.raises(DecodeError, match="must not be negative"):
        scripted_posteriors([0, -1])


def test_a_background_above_the_peak_is_refused() -> None:
    with pytest.raises(DecodeError, match="must exceed background_logit"):
        scripted_posteriors([0], background_logit=SCRIPTED_PEAK_LOGIT)


def test_frame_times_are_a_regular_grid() -> None:
    times = frame_times(5, 10.0)
    assert times.tolist() == [0.0, 0.01, 0.02, 0.03, 0.04]


def test_frame_times_scale_with_the_shift() -> None:
    assert frame_times(3, 20.0).tolist() == [0.0, 0.02, 0.04]


def test_frame_times_of_no_frames_is_empty() -> None:
    assert frame_times(0).shape == (0,)


def test_frame_index_inverts_frame_times() -> None:
    for index, time in enumerate(frame_times(12, 10.0)):
        assert frame_index(float(time) + 1e-9, 10.0) == index


def test_frame_index_clamps_a_negative_time() -> None:
    assert frame_index(-1.0) == 0


def test_frame_helpers_reject_a_bad_shift() -> None:
    with pytest.raises(DecodeError, match="frame_shift_ms must be positive"):
        frame_times(4, 0.0)
    with pytest.raises(DecodeError, match="frame_shift_ms must be positive"):
        frame_index(0.1, -1.0)


def test_a_bad_frame_count_is_refused() -> None:
    with pytest.raises(DecodeError, match="non-negative integer"):
        frame_times(-1)


def test_a_scripted_matrix_passes_validation() -> None:
    matrix = scripted_posteriors([0, 1, 2], num_symbols=4)
    assert np.array_equal(check_log_posteriors(matrix), matrix)


def test_a_flat_vector_is_refused() -> None:
    with pytest.raises(DecodeError, match=r"must be a \(frames, symbols\) matrix"):
        check_log_posteriors(np.zeros(5))


def test_an_empty_symbol_axis_is_refused() -> None:
    with pytest.raises(DecodeError, match="at least one symbol"):
        check_log_posteriors(np.zeros((3, 0)))


def test_a_nan_is_refused() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=3)
    matrix[1, 2] = np.nan
    with pytest.raises(DecodeError, match="NaN"):
        check_log_posteriors(matrix)


def test_an_unnormalised_row_is_refused() -> None:
    matrix = np.log(np.full((2, 3), 0.5))
    with pytest.raises(DecodeError, match="not normalised"):
        check_log_posteriors(matrix)


def test_a_row_of_minus_infinity_is_refused() -> None:
    matrix = scripted_posteriors([0], num_symbols=3)
    matrix[0, :] = float("-inf")
    with pytest.raises(DecodeError, match="not normalised"):
        check_log_posteriors(matrix)


def test_zero_frames_passes_without_a_normalisation_check() -> None:
    assert check_log_posteriors(np.zeros((0, 4))).shape == (0, 4)


def test_the_expected_symbol_count_is_enforced() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=3)
    assert check_log_posteriors(matrix, expected_symbols=3).shape == (2, 3)
    with pytest.raises(DecodeError, match="the model declares 5"):
        check_log_posteriors(matrix, expected_symbols=5)


def test_a_probability_table_converts_to_valid_log_posteriors() -> None:
    probabilities = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
    matrix = to_log_posteriors(probabilities)
    assert np.array_equal(check_log_posteriors(matrix), matrix)
    assert np.allclose(np.exp(matrix), probabilities, atol=1e-12)


def test_converting_a_negative_probability_is_refused() -> None:
    with pytest.raises(DecodeError, match="negative probabilities"):
        to_log_posteriors(np.array([[1.5, -0.5]]))


def test_a_mask_keeps_only_the_allowed_symbols() -> None:
    matrix = scripted_posteriors([0, 1, 2], num_symbols=4)
    masked = masked_posteriors(matrix, [1, 3])
    for frame in range(3):
        assert int(masked[frame].argmax()) in (1, 3)
    assert np.allclose(np.exp(logsumexp(masked, axis=-1)), 1.0, atol=1e-9)


def test_a_mask_that_excludes_everything_leaves_minus_infinity() -> None:
    matrix = scripted_posteriors([0], num_symbols=3)
    masked = masked_posteriors(matrix, [])
    assert np.all(np.isneginf(masked))


def test_a_mask_rejects_an_out_of_range_symbol() -> None:
    matrix = scripted_posteriors([0], num_symbols=3)
    with pytest.raises(DecodeError, match="outside the 3-symbol alphabet"):
        masked_posteriors(matrix, [7])
