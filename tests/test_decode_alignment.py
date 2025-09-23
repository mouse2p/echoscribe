"""Alignment is a prefix sum, so its arithmetic is pinned exactly."""

from __future__ import annotations

import pytest

from echoscribe.decode.alignment import WordSpan, align_words
from echoscribe.decode.tokens import DecodedToken
from echoscribe.errors import DecodeError


def _tokens(starts, ends, symbols=None):
    if symbols is None:
        symbols = range(len(starts))
    return tuple(
        DecodedToken(symbol, start, end)
        for symbol, start, end in zip(symbols, starts, ends, strict=False)
    )


def test_words_take_the_frames_of_their_own_tokens() -> None:
    tokens = _tokens([0, 2, 5, 7], [2, 5, 7, 9])
    spans = align_words(["hello", "你"], [3, 1], tokens)
    assert [span.word for span in spans] == ["hello", "你"]
    assert (spans[0].start, spans[0].end) == (0, 7)
    assert (spans[1].start, spans[1].end) == (7, 9)


def test_every_token_lands_in_exactly_one_word() -> None:
    tokens = _tokens([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    spans = align_words(["ab", "c", "de"], [2, 1, 2], tokens)
    assigned = [token for span in spans for token in span.tokens]
    assert len(assigned) == len(tokens)
    assert assigned == list(tokens)


def test_word_spans_are_ordered_and_do_not_run_backwards() -> None:
    tokens = _tokens([0, 3, 3, 6], [3, 3, 6, 9])
    spans = align_words(["a", "b", "c"], [1, 2, 1], tokens)
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert later.start >= earlier.start


def test_a_word_with_no_units_gets_a_zero_width_span() -> None:
    tokens = _tokens([0, 4], [4, 8])
    spans = align_words(["a", "missing", "b"], [1, 0, 1], tokens)
    assert spans[1].num_tokens == 0
    assert spans[1].num_frames == 0
    assert spans[1].start == spans[0].end


def test_a_leading_empty_word_starts_at_zero() -> None:
    tokens = _tokens([3], [7])
    spans = align_words(["", "a"], [0, 1], tokens)
    assert spans[0].start == 0
    assert spans[0].end == 0


def test_times_follow_the_frame_shift() -> None:
    tokens = _tokens([10], [14])
    assert align_words(["a"], [1], tokens)[0].times(10.0) == (0.1, pytest.approx(0.14))


def test_confidences_combine_geometrically() -> None:
    tokens = _tokens([0, 1], [1, 2])
    spans = align_words(["ab"], [2], tokens, confidences=(0.25, 1.0))
    assert spans[0].confidence == pytest.approx(0.5)


def test_a_word_without_units_has_no_confidence() -> None:
    tokens = _tokens([0], [2])
    spans = align_words(["missing", "a"], [0, 1], tokens, confidences=(0.4,))
    assert spans[0].confidence == 0.0
    assert spans[1].confidence == pytest.approx(0.4)


def test_no_confidences_leaves_every_word_at_one() -> None:
    tokens = _tokens([0, 2], [2, 4])
    spans = align_words(["a", "b"], [1, 1], tokens)
    assert all(span.confidence == 1.0 for span in spans)


def test_a_count_mismatch_is_refused() -> None:
    tokens = _tokens([0, 1], [1, 2])
    with pytest.raises(DecodeError, match="add to 3 but 2 tokens"):
        align_words(["a", "b"], [1, 2], tokens)


def test_a_word_count_mismatch_is_refused() -> None:
    with pytest.raises(DecodeError, match="cannot carry"):
        align_words(["a", "b"], [1], _tokens([0], [1]))


def test_a_negative_count_is_refused() -> None:
    with pytest.raises(DecodeError, match="negative number of units"):
        align_words(["a"], [-1], _tokens([], []))


def test_a_confidence_count_mismatch_is_refused() -> None:
    with pytest.raises(DecodeError, match="cannot describe"):
        align_words(["a"], [1], _tokens([0], [1]), confidences=(0.5, 0.5))


def test_a_word_span_rejects_an_invalid_range() -> None:
    with pytest.raises(DecodeError, match="invalid frame span"):
        WordSpan("a", 5, 2)


def test_a_word_span_rejects_an_out_of_range_confidence() -> None:
    with pytest.raises(DecodeError, match="must lie in"):
        WordSpan("a", 0, 1, (), 1.5)


def test_an_empty_alignment_is_allowed() -> None:
    assert align_words([], [], ()) == ()


def test_the_union_of_word_spans_covers_the_decode() -> None:
    tokens = _tokens([0, 2, 5, 5, 8], [2, 5, 5, 8, 11])
    spans = align_words(["ab", "c", "d"], [2, 2, 1], tokens)
    assert spans[0].start == tokens[0].start
    assert spans[-1].end == tokens[-1].end


def test_alignment_survives_a_whole_greedy_decode() -> None:
    import numpy as np

    from echoscribe.decode.confidence import probabilities_of, token_confidences
    from echoscribe.decode.ctc_greedy import ctc_greedy
    from echoscribe.decode.posteriors import scripted_posteriors

    matrix = scripted_posteriors([0, 1, 1, 0, 2, 3, 3, 0], num_symbols=5)
    result = ctc_greedy(matrix)
    confidences = token_confidences(matrix, result.tokens)
    counts = [1] * len(result.tokens)
    spans = align_words(
        [str(symbol) for symbol in result.symbols],
        counts,
        result.tokens,
        probabilities_of(confidences),
    )
    assert len(spans) == len(result.tokens)
    assert all(0.0 <= span.confidence <= 1.0 for span in spans)
    assert np.allclose([span.start for span in spans], [t.start for t in result.tokens])


def test_a_word_span_is_immutable() -> None:
    span = WordSpan("a", 0, 2)
    with pytest.raises(AttributeError):
        span.end = 4  # type: ignore[misc]
