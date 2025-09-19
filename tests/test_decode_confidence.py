"""Confidence is checked against posteriors whose answer can be written down."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.confidence import (
    CONFIDENCE_FLOOR,
    TokenConfidence,
    probabilities_of,
    token_confidences,
)
from echoscribe.decode.posteriors import scripted_posteriors, to_log_posteriors
from echoscribe.decode.tokens import DecodedToken
from echoscribe.errors import DecodeError


def test_a_dominant_token_is_fully_confident() -> None:
    matrix = scripted_posteriors([1, 1, 1], num_symbols=3, peak_logit=20.0)
    confidence = token_confidences(matrix, [DecodedToken(1, 0, 3)])[0]
    assert confidence.probability == pytest.approx(1.0, abs=1e-6)
    assert confidence.rank == 0
    assert confidence.margin == pytest.approx(0.0, abs=1e-6)


def test_a_uniform_posterior_gives_the_inverse_alphabet_confidence() -> None:
    matrix = to_log_posteriors(np.full((4, 4), 0.25))
    confidence = token_confidences(matrix, [DecodedToken(2, 0, 4)])[0]
    assert confidence.probability == pytest.approx(0.25, rel=1e-9)
    assert confidence.margin == pytest.approx(0.0, abs=1e-12)


def test_the_rank_counts_better_symbols() -> None:
    matrix = to_log_posteriors(np.array([[0.1, 0.2, 0.3, 0.4]]))
    ranks = [token_confidences(matrix, [DecodedToken(symbol, 0, 1)])[0].rank for symbol in range(4)]
    assert ranks == [3, 2, 1, 0]


def test_the_margin_is_the_gap_to_the_best_symbol() -> None:
    matrix = to_log_posteriors(np.array([[0.1, 0.2, 0.3, 0.4]]))
    confidence = token_confidences(matrix, [DecodedToken(1, 0, 1)])[0]
    assert confidence.margin == pytest.approx(0.2, rel=1e-9)


def test_the_strongest_frame_is_reported() -> None:
    matrix = to_log_posteriors(np.array([[0.4, 0.6], [0.9, 0.1], [0.5, 0.5]]))
    confidence = token_confidences(matrix, [DecodedToken(0, 0, 3)])[0]
    assert confidence.frame == 1


def test_a_geometric_mean_penalises_one_bad_frame() -> None:
    matrix = to_log_posteriors(np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]]))
    confidence = token_confidences(matrix, [DecodedToken(0, 0, 3)])[0]
    arithmetic = (0.99 + 0.01 + 0.99) / 3
    assert confidence.probability < arithmetic
    assert confidence.probability == pytest.approx((0.99 * 0.01 * 0.99) ** (1 / 3), rel=1e-9)


def test_a_zero_width_span_uses_its_start_frame() -> None:
    matrix = to_log_posteriors(np.array([[0.2, 0.8], [0.9, 0.1]]))
    confidence = token_confidences(matrix, [DecodedToken(1, 0, 0)])[0]
    assert confidence.probability == pytest.approx(0.8, rel=1e-9)
    assert confidence.frame == 0


def test_no_tokens_gives_no_confidences() -> None:
    assert token_confidences(scripted_posteriors([0, 1], num_symbols=2), []) == ()


def test_no_frames_gives_no_confidences() -> None:
    assert token_confidences(np.zeros((0, 3)), [DecodedToken(1, 0, 0)]) == ()


def test_a_token_outside_the_matrix_is_refused() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=3)
    with pytest.raises(DecodeError, match="outside the 2-frame posterior matrix"):
        token_confidences(matrix, [DecodedToken(1, 5, 6)])


def test_a_symbol_outside_the_alphabet_is_refused() -> None:
    matrix = scripted_posteriors([0, 1], num_symbols=3)
    with pytest.raises(DecodeError, match="outside the 3-symbol alphabet"):
        token_confidences(matrix, [DecodedToken(9, 0, 1)])


def test_probabilities_of_projects_the_confidence_list() -> None:
    matrix = scripted_posteriors([1, 2], num_symbols=3)
    confidences = token_confidences(matrix, [DecodedToken(1, 0, 1), DecodedToken(2, 1, 2)])
    assert probabilities_of(confidences) == tuple(item.probability for item in confidences)


def test_the_confidence_floor_is_below_any_real_posterior() -> None:
    assert CONFIDENCE_FLOOR < 1e-100


def test_a_confidence_rejects_an_out_of_range_probability() -> None:
    with pytest.raises(DecodeError, match="must lie in"):
        TokenConfidence(1, 1.5, 0, 0.0, 0)


def test_a_confidence_rejects_a_negative_rank_or_margin() -> None:
    with pytest.raises(DecodeError, match="rank must not be negative"):
        TokenConfidence(1, 0.5, -1, 0.0, 0)
    with pytest.raises(DecodeError, match="margin must not be negative"):
        TokenConfidence(1, 0.5, 0, -0.5, 0)


def test_every_token_of_a_decode_gets_a_confidence() -> None:
    matrix = scripted_posteriors([0, 1, 1, 0, 2], num_symbols=3)
    from echoscribe.decode.ctc_greedy import ctc_greedy

    result = ctc_greedy(matrix)
    confidences = token_confidences(matrix, result.tokens)
    assert len(confidences) == len(result.tokens)
    assert [item.symbol for item in confidences] == list(result.symbols)


def test_confidence_never_exceeds_one() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.01, 1.0, size=(20, 5))
    matrix = to_log_posteriors(probabilities / probabilities.sum(axis=1, keepdims=True))
    tokens = [DecodedToken(int(row.argmax()), index, index + 1) for index, row in enumerate(matrix)]
    assert all(item.probability <= 1.0 for item in token_confidences(matrix, tokens))


def test_a_confident_decode_beats_an_uncertain_one() -> None:
    sharp = scripted_posteriors([1, 2, 1, 2], num_symbols=4, peak_logit=12.0)
    flat = to_log_posteriors(np.full((4, 4), 0.25))
    tokens = [DecodedToken(symbol, index, index + 1) for index, symbol in enumerate([1, 2, 1, 2])]
    confident = token_confidences(sharp, tokens)
    uncertain = token_confidences(flat, tokens)
    assert min(item.probability for item in confident) > max(item.probability for item in uncertain)


def test_the_margin_measures_the_gap_to_the_best_symbol() -> None:
    tied = to_log_posteriors(np.array([[0.5, 0.5]]))
    clear = to_log_posteriors(np.array([[0.9, 0.1]]))
    # The winner's margin is zero by definition; the runner-up's is the gap.
    assert token_confidences(tied, [DecodedToken(0, 0, 1)])[0].margin == pytest.approx(0.0)
    assert token_confidences(clear, [DecodedToken(0, 0, 1)])[0].margin == pytest.approx(0.0)
    assert token_confidences(clear, [DecodedToken(1, 0, 1)])[0].margin == pytest.approx(0.8)
