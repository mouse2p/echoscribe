"""The two mel scales are pinned at frequencies where their value is exact."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.mel import (
    HTK_REFERENCE_HZ,
    HTK_REFERENCE_MEL,
    MEL_SCALES,
    SLANEY_MIN_LOG_HZ,
    SLANEY_MIN_LOG_MEL,
    hz_to_mel,
    mel_centre_frequencies,
    mel_edges,
    mel_filterbank,
    mel_to_hz,
)


def test_both_scales_map_zero_to_zero() -> None:
    for scale in MEL_SCALES:
        assert hz_to_mel(0.0, scale) == pytest.approx(0.0, abs=1e-12)
        assert mel_to_hz(0.0, scale) == pytest.approx(0.0, abs=1e-12)


def test_htk_maps_seven_times_the_reference_to_exactly_one_decade() -> None:
    # 1 + 6300/700 = 10, so the mel value is the reference constant itself.
    assert hz_to_mel(9.0 * HTK_REFERENCE_HZ, "htk") == pytest.approx(HTK_REFERENCE_MEL, rel=1e-12)


def test_slaney_is_linear_below_one_kilohertz() -> None:
    assert hz_to_mel(SLANEY_MIN_LOG_HZ, "slaney") == pytest.approx(SLANEY_MIN_LOG_MEL)
    assert hz_to_mel(500.0, "slaney") == pytest.approx(7.5)
    assert hz_to_mel(250.0, "slaney") == pytest.approx(3.75)


def test_slaney_log_anchor_is_6400_hz_at_mel_42() -> None:
    assert hz_to_mel(6400.0, "slaney") == pytest.approx(42.0, rel=1e-12)


def test_slaney_is_continuous_across_its_break_point() -> None:
    below = hz_to_mel(SLANEY_MIN_LOG_HZ - 1e-6, "slaney")
    above = hz_to_mel(SLANEY_MIN_LOG_HZ + 1e-6, "slaney")
    assert below == pytest.approx(above, abs=1e-6)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_conversions_are_monotonic(scale: str) -> None:
    frequencies = np.linspace(0.0, 8000.0, 512)
    mels = hz_to_mel(frequencies, scale)
    assert np.all(np.diff(mels) > 0.0)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_round_trip_recovers_the_frequency(scale: str) -> None:
    frequencies = np.array([0.0, 20.0, 300.0, 1000.0, 4000.0, 7999.0])
    assert np.allclose(mel_to_hz(hz_to_mel(frequencies, scale), scale), frequencies, atol=1e-9)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_a_scalar_stays_a_scalar_and_an_array_stays_an_array(scale: str) -> None:
    assert isinstance(hz_to_mel(1000.0, scale), float)
    assert isinstance(hz_to_mel(np.array([1000.0]), scale), np.ndarray)
    assert isinstance(mel_to_hz(15.0, scale), float)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_negative_inputs_are_rejected(scale: str) -> None:
    with pytest.raises(ConfigError, match="non-negative"):
        hz_to_mel(-1.0, scale)
    with pytest.raises(ConfigError, match="non-negative"):
        mel_to_hz(-1.0, scale)


def test_an_unknown_scale_is_rejected() -> None:
    with pytest.raises(ConfigError, match="scale must be one of"):
        hz_to_mel(1000.0, "erb")
    with pytest.raises(ConfigError, match="scale must be one of"):
        mel_to_hz(15.0, "erb")


def test_the_two_scales_disagree_above_the_slaney_break_point() -> None:
    htk = hz_to_mel(4000.0, "htk")
    slaney = hz_to_mel(4000.0, "slaney")
    assert htk != pytest.approx(slaney)
    assert slaney < htk


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_edges_start_at_the_low_cut_and_end_at_the_high_cut(scale: str) -> None:
    edges = mel_edges(23, 16000, 100.0, 7500.0, scale)
    assert edges.shape == (25,)
    assert edges[0] == pytest.approx(100.0)
    assert edges[-1] == pytest.approx(7500.0)


@pytest.mark.parametrize("scale", list(MEL_SCALES))
def test_edges_are_strictly_increasing(scale: str) -> None:
    edges = mel_edges(40, 16000, 20.0, None, scale)
    assert np.all(np.diff(edges) > 0.0)


def test_the_default_high_edge_is_nyquist() -> None:
    assert mel_edges(10, 16000, 0.0)[-1] == pytest.approx(8000.0)
    assert mel_edges(10, 8000, 0.0)[-1] == pytest.approx(4000.0)


def test_centre_frequencies_are_the_interior_edges() -> None:
    edges = mel_edges(8, 16000, 50.0, 7000.0)
    assert np.array_equal(mel_centre_frequencies(8, 16000, 50.0, 7000.0), edges[1:-1])


def test_centres_sit_between_their_own_edges() -> None:
    edges = mel_edges(20, 16000, 20.0, None, "slaney")
    centres = mel_centre_frequencies(20, 16000, 20.0, None, "slaney")
    assert np.all(centres > edges[:-2])
    assert np.all(centres < edges[2:])


def test_mel_spacing_compresses_the_high_end() -> None:
    edges = mel_edges(20, 16000, 0.0, None, "htk")
    widths = np.diff(edges)
    assert widths[-1] > 4 * widths[0]


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1.0, 4000.0), (4000.0, 4000.0), (5000.0, 4000.0), (0.0, 9000.0)],
)
def test_an_impossible_band_is_rejected(low: float, high: float) -> None:
    with pytest.raises(ConfigError, match="band must satisfy"):
        mel_edges(10, 16000, low, high)


@pytest.mark.parametrize("bins", [0, -3, 2.5])
def test_a_bad_bin_count_is_rejected(bins) -> None:
    with pytest.raises(ConfigError, match="num_bins must be a positive integer"):
        mel_edges(bins, 16000)


def test_filterbank_shape_matches_the_rfft_width() -> None:
    bank = mel_filterbank(23, 512, 16000)
    assert bank.shape == (23, 257)


def test_filterbank_weights_stay_inside_zero_and_one() -> None:
    bank = mel_filterbank(40, 512, 16000, 20.0, None, "slaney")
    assert bank.min() >= 0.0
    assert bank.max() <= 1.0


def test_each_filter_has_one_contiguous_non_zero_run() -> None:
    bank = mel_filterbank(23, 512, 16000)
    for row in bank:
        support = np.flatnonzero(row > 0.0)
        if support.size:
            assert support[-1] - support[0] + 1 == support.size


def test_filter_centres_move_up_the_frequency_axis() -> None:
    bank = mel_filterbank(23, 512, 16000, 20.0)
    peaks = [int(np.argmax(row)) for row in bank]
    assert peaks == sorted(peaks)
    assert len(set(peaks)) == len(peaks)


def test_filters_below_the_low_cut_are_empty() -> None:
    bank = mel_filterbank(10, 512, 16000, 2000.0)
    assert np.all(bank[:, :64] == 0.0)


def test_filters_above_the_high_cut_are_empty() -> None:
    bank = mel_filterbank(10, 512, 16000, 0.0, 3000.0)
    frequencies = np.fft.rfftfreq(512, 1 / 16000)
    assert np.all(bank[:, frequencies > 3000.0] == 0.0)


def test_a_single_bin_bank_spans_the_whole_band() -> None:
    bank = mel_filterbank(1, 8, 8000, 0.0, 4000.0, "htk")
    assert bank.shape == (1, 5)
    centre = mel_to_hz(hz_to_mel(4000.0, "htk") / 2.0, "htk")
    frequencies = np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
    expected = np.where(
        frequencies <= centre,
        frequencies / centre,
        (4000.0 - frequencies) / (4000.0 - centre),
    )
    assert np.allclose(bank[0], np.clip(expected, 0.0, 1.0), atol=1e-9)
    assert bank[0, 0] == 0.0
    assert bank[0, -1] == pytest.approx(0.0, abs=1e-12)


def test_the_filterbank_is_deterministic() -> None:
    first = mel_filterbank(40, 512, 16000, 20.0, None, "slaney")
    assert np.array_equal(first, mel_filterbank(40, 512, 16000, 20.0, None, "slaney"))
