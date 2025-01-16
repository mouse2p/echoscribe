"""The dtype contract is what makes byte-identical golden tests meaningful."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe import types
from echoscribe.errors import FeatureError


def test_dtypes_are_the_documented_numpy_kinds() -> None:
    assert types.SAMPLE_DTYPE == np.dtype(np.float32)
    assert types.FEATURE_DTYPE == np.dtype(np.float64)
    assert types.INDEX_DTYPE == np.dtype(np.intp)


def test_sample_dtype_is_narrower_than_feature_dtype() -> None:
    assert types.SAMPLE_DTYPE.itemsize < types.FEATURE_DTYPE.itemsize


def test_index_dtype_can_address_every_frame_of_a_day_of_audio() -> None:
    # 24 h at a 10 ms shift is 8.64M frames; intp is what numpy itself uses for
    # indexing, so it can never be too small for a waveform we can hold in RAM.
    assert types.INDEX_DTYPE.itemsize >= 4
    assert np.iinfo(types.INDEX_DTYPE).max > 24 * 3600 * 100


def test_every_exported_name_exists_and_is_listed_once() -> None:
    assert len(types.__all__) == len(set(types.__all__))
    for name in types.__all__:
        assert hasattr(types, name), name


def test_aliases_and_dtypes_are_all_exported() -> None:
    for name in (
        "SampleArray",
        "FloatArray",
        "LogArray",
        "IntArray",
        "SAMPLE_DTYPE",
        "FEATURE_DTYPE",
        "INDEX_DTYPE",
    ):
        assert name in types.__all__


def test_as_samples_returns_a_contiguous_float32_copy() -> None:
    source = np.array([0, 1, -1], dtype=np.int16)
    result = types.as_samples(source)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]
    result[0] = 99.0
    assert source[0] == 0


def test_as_samples_rejects_a_matrix() -> None:
    with pytest.raises(FeatureError, match="must be 1-D"):
        types.as_samples(np.zeros((2, 3), dtype=np.float32), name="batch")


def test_as_samples_rejects_nan_and_infinity() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(FeatureError, match="must be finite"):
            types.as_samples([0.0, bad, 0.1], name="clip")


def test_as_features_upcasts_to_float64() -> None:
    assert types.as_features(np.zeros((2, 3), dtype=np.float32)).dtype == np.float64


def test_as_features_rejects_a_vector() -> None:
    with pytest.raises(FeatureError, match="must be 2-D"):
        types.as_features(np.zeros(5))


def test_require_rank_reports_both_ranks() -> None:
    array = np.zeros((2, 3, 4))
    assert types.require_rank("cubes", array, 3) is array
    with pytest.raises(FeatureError, match="must be 2-D, got 3-D"):
        types.require_rank("cubes", array, 2)


def test_require_non_empty_accepts_zero_valued_but_rejects_zero_sized() -> None:
    assert types.require_non_empty("x", np.zeros(1)).size == 1
    with pytest.raises(FeatureError, match="must not be empty"):
        types.require_non_empty("x", np.zeros(0))


def test_as_samples_preserves_values_that_float32_can_hold() -> None:
    values = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
    result = types.as_samples(values)
    assert np.array_equal(result, values.astype(np.float32))


def test_as_samples_accepts_a_python_list() -> None:
    assert list(types.as_samples([0.5, -0.25])) == [0.5, -0.25]


def test_as_samples_accepts_an_empty_buffer() -> None:
    # Zero samples is legal: a trailing chunk may be empty.  Only non-finite
    # data is rejected.
    assert types.as_samples([]).shape == (0,)


def test_require_finite_returns_the_same_object() -> None:
    array = np.ones((2, 2))
    assert types.require_finite("x", array) is array


def test_require_finite_counts_the_bad_entries() -> None:
    array = np.array([1.0, float("nan"), 2.0, float("inf")])
    with pytest.raises(FeatureError, match="found 2 non-finite entries"):
        types.require_finite("energy", array)


def test_as_features_does_not_copy_an_already_correct_matrix() -> None:
    source = np.zeros((3, 2), dtype=np.float64, order="C")
    assert np.shares_memory(types.as_features(source), source)
