"""The decoded-token record is the contract every decoder has to satisfy."""

from __future__ import annotations

import pytest

from echoscribe.decode.tokens import DecodedToken, span_gaps, symbols_of, total_score
from echoscribe.errors import DecodeError


def test_a_token_keeps_its_span_and_score() -> None:
    token = DecodedToken(3, 10, 14, -2.5)
    assert token.symbol == 3
    assert token.num_frames == 4
    assert token.score == -2.5


def test_a_zero_length_span_is_allowed_but_not_a_reversed_one() -> None:
    assert DecodedToken(1, 5, 5).num_frames == 0
    with pytest.raises(DecodeError, match="not a valid half-open range"):
        DecodedToken(1, 5, 4)


def test_a_negative_start_is_refused() -> None:
    with pytest.raises(DecodeError, match="not a valid half-open range"):
        DecodedToken(1, -1, 3)


def test_a_negative_symbol_is_refused() -> None:
    with pytest.raises(DecodeError, match="symbol must not be negative"):
        DecodedToken(-1, 0, 1)


def test_a_positive_score_is_refused() -> None:
    with pytest.raises(DecodeError, match="cannot be positive"):
        DecodedToken(1, 0, 1, 0.5)


def test_a_score_a_hair_above_zero_from_rounding_is_allowed() -> None:
    assert DecodedToken(1, 0, 1, 1e-19).score == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(DecodeError, match="cannot be positive"):
        DecodedToken(1, 0, 1, 1e-6)


def test_minus_infinity_is_an_allowed_score() -> None:
    assert DecodedToken(1, 0, 1, float("-inf")).score == float("-inf")


def test_times_follow_the_frame_shift() -> None:
    assert DecodedToken(1, 10, 14).times(10.0) == (0.1, pytest.approx(0.14))
    assert DecodedToken(1, 10, 14).times(20.0) == (0.2, pytest.approx(0.28))


def test_symbols_of_projects_the_sequence() -> None:
    tokens = (DecodedToken(4, 0, 1), DecodedToken(7, 1, 3))
    assert symbols_of(tokens) == (4, 7)
    assert symbols_of(()) == ()


def test_total_score_adds_the_log_probabilities() -> None:
    tokens = (DecodedToken(1, 0, 1, -1.5), DecodedToken(2, 1, 2, -2.5))
    assert total_score(tokens) == pytest.approx(-4.0)


def test_span_gaps_measures_the_blank_runs() -> None:
    tokens = (DecodedToken(1, 0, 3), DecodedToken(1, 6, 8), DecodedToken(2, 8, 9))
    assert span_gaps(tokens) == (3, 0)


def test_span_gaps_of_a_single_token_is_empty() -> None:
    assert span_gaps((DecodedToken(1, 0, 3),)) == ()


def test_a_token_is_immutable() -> None:
    token = DecodedToken(1, 0, 2)
    with pytest.raises(AttributeError):
        token.end = 5  # type: ignore[misc]


def test_the_decode_subpackage_re_exports_its_records() -> None:
    from echoscribe import decode

    assert decode.DecodedToken is DecodedToken
    assert decode.symbols_of is symbols_of
    for name in ("DecodedToken", "ctc_greedy", "ctc_prefix_search", "scripted_posteriors"):
        assert name in decode.__all__
