"""Rate conversion is checked on tones, where an error shows up as a pitch shift."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.resample import rational_factors


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (16000, 8000, (1, 2)),
        (8000, 16000, (2, 1)),
        (44100, 16000, (160, 441)),
        (16000, 16000, (1, 1)),
        (48000, 16000, (1, 3)),
        (22050, 44100, (2, 1)),
    ],
)
def test_rational_factors_reduces_to_a_coprime_pair(source, target, expected) -> None:
    assert rational_factors(source, target) == expected


def test_rational_factors_reproduces_the_ratio() -> None:
    for source, target in [(44100, 16000), (16000, 22050), (8000, 48000)]:
        up, down = rational_factors(source, target)
        assert up / down == pytest.approx(target / source, rel=1e-12)


@pytest.mark.parametrize("bad", [0, -16000, 16000.0, "16000", None, True])
def test_rational_factors_rejects_a_bad_rate(bad) -> None:
    with pytest.raises(ConfigError):
        rational_factors(16000, bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        rational_factors(bad, 16000)  # type: ignore[arg-type]
