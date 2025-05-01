"""Tone contours are pinned at their control points, which is all there is to them."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.synth.tone import (
    NEUTRAL_TONE,
    TONE_INDICES,
    TONES,
    ToneContour,
    tone_by_index,
)


def _level() -> ToneContour:
    return ToneContour(1, "high-level", ((0.0, 1.1), (1.0, 1.1)))


def test_a_contour_keeps_its_identity() -> None:
    contour = ToneContour(3, "dipping", ((0.0, 1.0), (0.4, 0.75), (1.0, 1.15)))
    assert contour.index == 3
    assert contour.name == "dipping"
    assert len(contour.control_points) == 3


def test_at_returns_the_control_values_at_the_control_times() -> None:
    contour = ToneContour(2, "rising", ((0.0, 0.85), (0.5, 1.05), (1.0, 1.25)))
    assert contour.at(0.0) == pytest.approx(0.85)
    assert contour.at(0.5) == pytest.approx(1.05)
    assert contour.at(1.0) == pytest.approx(1.25)


def test_at_interpolates_linearly_between_control_points() -> None:
    contour = ToneContour(2, "rising", ((0.0, 0.8), (1.0, 1.2)))
    assert contour.at(0.25) == pytest.approx(0.9)
    assert contour.at(0.5) == pytest.approx(1.0)
    assert contour.at(0.75) == pytest.approx(1.1)


def test_at_clamps_outside_the_syllable() -> None:
    contour = _level()
    assert contour.at(-1.0) == contour.at(0.0)
    assert contour.at(2.0) == contour.at(1.0)


def test_a_level_tone_is_flat() -> None:
    trajectory = _level().trajectory(64)
    assert np.allclose(trajectory, 1.1)


def test_trajectory_length_and_endpoints() -> None:
    contour = ToneContour(4, "falling", ((0.0, 1.3), (1.0, 0.8)))
    trajectory = contour.trajectory(11)
    assert trajectory.shape == (11,)
    assert trajectory[0] == pytest.approx(1.3)
    assert trajectory[-1] == pytest.approx(0.8)


def test_a_single_sample_trajectory_is_the_start_of_the_contour() -> None:
    contour = ToneContour(4, "falling", ((0.0, 1.3), (1.0, 0.8)))
    assert contour.trajectory(1).tolist() == [pytest.approx(1.3)]


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 1.0),),
        ((0.2, 1.0), (1.0, 1.1)),
        ((0.0, 1.0), (0.9, 1.1)),
        ((0.0, 1.0), (0.5, 1.1), (0.5, 1.2), (1.0, 1.0)),
        ((0.0, 0.2), (1.0, 1.0)),
        ((0.0, 1.0), (1.0, 2.5)),
    ],
)
def test_malformed_contours_are_rejected(points) -> None:
    with pytest.raises(ConfigError):
        ToneContour(9, "bad", points)


def test_trajectory_rejects_a_non_positive_length() -> None:
    with pytest.raises(ConfigError, match="at least one sample"):
        _level().trajectory(0)


def test_a_contour_is_immutable() -> None:
    contour = _level()
    with pytest.raises(AttributeError):
        contour.name = "other"  # type: ignore[misc]


def test_the_laboratory_defines_four_lexical_tones() -> None:
    assert TONE_INDICES == (1, 2, 3, 4)
    assert [tone.name for tone in TONES] == ["high-level", "rising", "dipping", "falling"]


def test_tone_one_is_monotonically_flat_to_falling() -> None:
    trajectory = TONES[0].trajectory(64)
    assert np.all(np.diff(trajectory) <= 1e-12)


def test_tone_two_rises_throughout() -> None:
    trajectory = TONES[1].trajectory(64)
    assert np.all(np.diff(trajectory) > 0.0)


def test_tone_three_dips_below_both_of_its_endpoints() -> None:
    trajectory = TONES[2].trajectory(256)
    assert trajectory.min() < trajectory[0]
    assert trajectory.min() < trajectory[-1]
    assert int(np.argmin(trajectory)) not in (0, 255)


def test_tone_four_falls_throughout() -> None:
    trajectory = TONES[3].trajectory(64)
    assert np.all(np.diff(trajectory) < 0.0)


def test_the_four_tones_are_distinguishable_at_the_syllable_end() -> None:
    endings = {round(float(tone.trajectory(64)[-1]), 6) for tone in TONES}
    assert len(endings) == 4


def test_every_registered_tone_stays_inside_the_voice_range() -> None:
    for tone in (*TONES, NEUTRAL_TONE):
        trajectory = tone.trajectory(128)
        assert trajectory.min() > 0.7
        assert trajectory.max() < 1.4


def test_lookup_returns_the_registered_object() -> None:
    for index in (1, 2, 3, 4):
        assert tone_by_index(index) is TONES[index - 1]
    assert tone_by_index(0) is NEUTRAL_TONE


def test_lookup_rejects_an_unknown_index() -> None:
    with pytest.raises(ConfigError, match="unknown tone index 7"):
        tone_by_index(7)
