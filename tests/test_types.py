"""The dtype contract is what makes byte-identical golden tests meaningful."""

from __future__ import annotations

import numpy as np

from echoscribe import types


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
