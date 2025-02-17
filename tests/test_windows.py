"""Window coefficients are pinned at the edges and at the centre."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.windows import FILTER_WINDOWS, window_vector


def test_rectangular_window_is_all_ones() -> None:
    assert window_vector("rectangular", 7).tolist() == [1.0] * 7


def test_hamming_keeps_a_small_non_zero_endpoint() -> None:
    window = window_vector("hamming", 9)
    assert window[0] == pytest.approx(0.08, abs=1e-12)
    assert window[-1] == pytest.approx(0.08, abs=1e-12)
    assert window[4] == pytest.approx(1.0)


def test_hann_endpoints_are_exactly_zero() -> None:
    window = window_vector("hann", 9)
    assert window[0] == 0.0
    assert window[-1] == 0.0
    assert window[4] == pytest.approx(1.0)


def test_blackman_endpoints_are_zero_to_float_precision() -> None:
    window = window_vector("blackman", 9)
    assert abs(window[0]) < 1e-15
    assert window[4] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", list(FILTER_WINDOWS))
def test_every_window_is_symmetric_and_peaks_at_the_centre(kind: str) -> None:
    window = window_vector(kind, 17)
    assert np.allclose(window, window[::-1], atol=1e-15)
    assert window[8] == pytest.approx(float(window.max()))
    if kind != "rectangular":
        # A rectangular window is flat, so only tapered ones have a unique peak.
        assert int(np.argmax(window)) == 8
    assert window.dtype == np.float64


@pytest.mark.parametrize("kind", list(FILTER_WINDOWS))
def test_length_one_is_all_ones(kind: str) -> None:
    assert window_vector(kind, 1).tolist() == [1.0]


def test_window_vector_rejects_an_unknown_kind() -> None:
    with pytest.raises(ConfigError, match="window must be one of"):
        window_vector("kaiser", 8)


def test_window_vector_rejects_a_non_positive_length() -> None:
    with pytest.raises(ConfigError, match="length must be positive"):
        window_vector("hamming", 0)


def test_filter_windows_lists_exactly_four_designs() -> None:
    assert FILTER_WINDOWS == ("rectangular", "hamming", "hann", "blackman")
