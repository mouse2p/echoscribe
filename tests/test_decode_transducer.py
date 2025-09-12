"""The scripted joint is a fixture, so its shape contract is checked exactly."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.transducer import (
    Joint,
    TableJoint,
    TransducerConfig,
    TransducerResult,
    transducer_greedy,
)
from echoscribe.errors import DecodeError


def _table(frames: int = 6, labels: int = 4, symbols: int = 3, seed: int = 0) -> TableJoint:
    rng = np.random.default_rng(seed)
    return TableJoint(rng.uniform(-2.0, 2.0, size=(frames, labels + 1, symbols)))


def test_a_table_reports_its_dimensions() -> None:
    joint = _table(frames=5, labels=3, symbols=4)
    assert joint.num_frames == 5
    assert joint.num_symbols == 4
    assert joint.max_labels == 3


def test_a_table_satisfies_the_joint_protocol() -> None:
    assert isinstance(_table(), Joint)


def test_every_state_is_a_normalised_log_distribution() -> None:
    joint = _table()
    for frame in range(joint.num_frames):
        for emitted in range(joint.max_labels + 1):
            row = joint(frame, tuple(range(1, emitted + 1)))
            assert np.all(row <= 1e-12)
            assert np.exp(row).sum() == pytest.approx(1.0, rel=1e-12)


def test_the_history_only_selects_a_predictor_state() -> None:
    joint = _table(labels=4)
    assert np.array_equal(joint(2, (1, 1)), joint(2, (2, 1)))
    assert not np.array_equal(joint(2, (1,)), joint(2, (1, 1)))


def test_a_frame_outside_the_table_is_refused() -> None:
    joint = _table(frames=4)
    with pytest.raises(DecodeError, match="outside the 4-frame table"):
        joint(4, ())
    with pytest.raises(DecodeError, match="outside the 4-frame table"):
        joint(-1, ())


def test_a_history_longer_than_the_table_is_refused() -> None:
    with pytest.raises(DecodeError, match="covers 2 emitted labels"):
        _table(labels=2)(0, (1, 1, 1))


def test_a_bad_table_shape_is_refused() -> None:
    with pytest.raises(DecodeError, match="a joint table must be"):
        TableJoint(np.zeros((4, 3)))


def test_a_table_with_one_symbol_is_refused() -> None:
    with pytest.raises(DecodeError, match="cannot hold a blank and a label"):
        TableJoint(np.zeros((4, 2, 1)))


def test_a_non_finite_logit_is_refused() -> None:
    logits = np.zeros((3, 2, 3))
    logits[1, 0, 2] = np.nan
    with pytest.raises(DecodeError, match="only finite logits"):
        TableJoint(logits)


def test_a_blank_outside_the_alphabet_is_refused() -> None:
    with pytest.raises(DecodeError, match="outside the 3-symbol alphabet"):
        TableJoint(np.zeros((3, 2, 3)), blank_id=7)


def _blank_heavy(frames: int, symbols: int = 3) -> TableJoint:
    """A joint whose argmax is always the blank: nothing can be emitted."""
    logits = np.full((frames, 2, symbols), -5.0)
    logits[:, :, 0] = 5.0
    return TableJoint(logits)


def _label_then_blank(frames: int, symbol: int = 1, symbols: int = 3) -> TableJoint:
    """Emits ``symbol`` once, from frame 0, then blanks for the rest of the audio."""
    logits = np.full((frames, 3, symbols), -5.0)
    logits[:, 0, symbol] = 5.0
    logits[:, 1:, 0] = 5.0
    return TableJoint(logits)


def _two_labels_from_one_frame(frames: int = 3) -> TableJoint:
    """Emits 1 then 2 from frame 0, then blanks."""
    logits = np.full((frames, 4, 3), -5.0)
    logits[:, 0, 1] = 5.0
    logits[:, 1, 2] = 5.0
    logits[:, 2:, 0] = 5.0
    return TableJoint(logits)


def test_an_all_blank_joint_emits_nothing() -> None:
    result = transducer_greedy(_blank_heavy(5))
    assert isinstance(result, TransducerResult)
    assert result.symbols == ()
    assert result.num_frames == 5
    assert result.emissions_per_frame == (0, 0, 0, 0, 0)


def test_a_single_label_is_emitted_and_then_the_frames_run_out() -> None:
    result = transducer_greedy(_label_then_blank(4))
    assert result.symbols == (1,)
    assert result.frames == (0,)


def test_the_score_sums_every_chosen_log_probability_including_blanks() -> None:
    joint = _label_then_blank(3)
    result = transducer_greedy(joint)
    expected = float(joint(0, ())[1]) + sum(float(joint(frame, (1,))[0]) for frame in range(3))
    assert result.score == pytest.approx(expected)


def test_token_scores_count_labels_only() -> None:
    result = transducer_greedy(_two_labels_from_one_frame(4))
    assert sum(token.score for token in result.tokens) > result.score


def test_a_custom_blank_id_is_honoured() -> None:
    logits = np.full((3, 2, 3), -5.0)
    logits[:, :, 2] = 5.0
    result = transducer_greedy(TableJoint(logits, blank_id=2), TransducerConfig(blank_id=2))
    assert result.symbols == ()


def test_a_blank_id_outside_the_alphabet_is_refused() -> None:
    with pytest.raises(DecodeError, match="outside the 3-symbol alphabet"):
        transducer_greedy(_blank_heavy(2), TransducerConfig(blank_id=9))


def test_an_empty_joint_decodes_to_nothing() -> None:
    result = transducer_greedy(TableJoint(np.zeros((0, 2, 3))))
    assert result.symbols == ()
    assert result.num_frames == 0
    assert result.score == 0.0


def test_decoding_is_deterministic() -> None:
    joint = _table(labels=16, seed=3)
    config = TransducerConfig(max_symbols_per_frame=2)
    assert transducer_greedy(joint, config).symbols == transducer_greedy(joint, config).symbols


def test_a_config_rejects_a_bad_cap() -> None:
    with pytest.raises(DecodeError, match="max_symbols_per_frame must be positive"):
        TransducerConfig(max_symbols_per_frame=0)
    with pytest.raises(DecodeError, match="must be an integer"):
        TransducerConfig(max_symbols_per_frame=2.5)


def test_tokens_emitted_from_the_same_frame_get_a_zero_width_span() -> None:
    result = transducer_greedy(_two_labels_from_one_frame())
    assert result.symbols == (1, 2)
    assert result.frames == (0, 0)
    assert result.tokens[0].start == result.tokens[0].end == 0
    assert result.tokens[1].start == 0
    assert result.tokens[1].end == 3


def test_spans_never_run_backwards() -> None:
    result = transducer_greedy(_two_labels_from_one_frame(6))
    for earlier, later in zip(result.tokens, result.tokens[1:], strict=False):
        assert later.start >= earlier.start


def test_the_last_token_runs_to_the_end_of_the_utterance() -> None:
    result = transducer_greedy(_label_then_blank(7))
    assert result.tokens[-1].end == 7


def test_token_times_follow_the_frame_shift() -> None:
    result = transducer_greedy(_label_then_blank(5))
    # The single label was emitted from frame 0 and owns every frame after it, so
    # its span runs to the end of the five-frame utterance: 0.0 s to 0.05 s.
    assert result.tokens[0].times(10.0) == (0.0, pytest.approx(0.05))


def test_emissions_per_frame_counts_multiple_labels() -> None:
    assert transducer_greedy(_two_labels_from_one_frame(4)).emissions_per_frame == (2, 0, 0, 0)


def test_a_token_never_revisits_an_earlier_frame() -> None:
    joint = _table(frames=12, labels=40, seed=5)
    result = transducer_greedy(joint, TransducerConfig(max_symbols_per_frame=3))
    assert list(result.frames) == sorted(result.frames)
    assert all(0 <= frame < 12 for frame in result.frames)


def _never_blank(frames: int = 5, symbols: int = 3) -> TableJoint:
    """A joint that always prefers a label, so only the cap can stop it."""
    logits = np.full((frames, frames * 8 + 2, symbols), -5.0)
    logits[:, :, 1] = 5.0
    return TableJoint(logits)


def test_the_cap_bounds_how_many_labels_one_frame_can_emit() -> None:
    result = transducer_greedy(_never_blank(), TransducerConfig(max_symbols_per_frame=2))
    assert max(result.emissions_per_frame) <= 2


def test_the_cap_still_advances_time() -> None:
    result = transducer_greedy(_never_blank(frames=4), TransducerConfig(max_symbols_per_frame=2))
    assert result.num_frames == 4
    assert len(result.symbols) == 8
    assert sorted(set(result.frames)) == [0, 1, 2, 3]


def test_a_cap_of_one_emits_at_most_one_label_per_frame() -> None:
    result = transducer_greedy(_never_blank(frames=3), TransducerConfig(max_symbols_per_frame=1))
    assert result.emissions_per_frame == (1, 1, 1)


def test_a_generous_cap_does_not_truncate_a_normal_decode() -> None:
    joint = _label_then_blank(5)
    tight = transducer_greedy(joint, TransducerConfig(max_symbols_per_frame=10))
    loose = transducer_greedy(joint, TransducerConfig(max_symbols_per_frame=100))
    assert tight.symbols == loose.symbols


def test_the_cap_never_pushes_a_token_past_the_last_frame() -> None:
    result = transducer_greedy(_never_blank(frames=6), TransducerConfig(max_symbols_per_frame=3))
    assert all(token.end <= 6 for token in result.tokens)
    assert all(token.start < 6 for token in result.tokens)


def test_a_stuck_joint_never_runs_past_the_frame_count() -> None:
    for cap in (1, 2, 5):
        config = TransducerConfig(max_symbols_per_frame=cap)
        result = transducer_greedy(_never_blank(frames=4), config)
        assert max(result.frames) < 4
        assert len(result.symbols) == 4 * cap


def _eos_joint(frames: int = 6, eos: int = 2, emit: int = 2) -> TableJoint:
    """Emits ``emit`` labels from frame 0, then the end-of-sequence symbol.

    States past the EOS one prefer the blank, so a decode that ignores ``eos_id``
    still terminates and still consumes every frame.
    """
    logits = np.full((frames, frames * 8 + 2, 3), -5.0)
    for index in range(emit):
        logits[:, index, 1] = 5.0
    logits[:, emit, eos] = 5.0
    logits[:, emit + 1 :, 0] = 5.0
    return TableJoint(logits)


def test_an_eos_symbol_ends_the_decode_early() -> None:
    config = TransducerConfig(max_symbols_per_frame=5, eos_id=2)
    result = transducer_greedy(_eos_joint(frames=6), config)
    assert result.symbols == (1, 1)
    assert result.frames_consumed == 0
    assert result.stopped_early


def test_without_an_eos_id_the_decoder_runs_out_of_frames() -> None:
    result = transducer_greedy(_eos_joint(frames=6), TransducerConfig(max_symbols_per_frame=5))
    assert result.frames_consumed == 6
    assert not result.stopped_early


def test_an_eos_symbol_is_scored_but_not_emitted() -> None:
    joint = _eos_joint(frames=4)
    config = TransducerConfig(max_symbols_per_frame=5, eos_id=2)
    result = transducer_greedy(joint, config)
    expected = 2 * float(joint(0, ())[1]) + float(joint(0, (1, 1))[2])
    assert result.score == pytest.approx(expected)
    assert 2 not in result.symbols


def test_an_eos_on_the_first_frame_emits_nothing() -> None:
    logits = np.full((5, 2, 3), -5.0)
    logits[:, 0, 2] = 5.0
    result = transducer_greedy(TableJoint(logits), TransducerConfig(eos_id=2))
    assert result.symbols == ()
    assert result.frames_consumed == 0
    assert result.tokens == ()


def test_frames_consumed_equals_the_frame_count_when_nothing_stops_it() -> None:
    for frames in (1, 3, 6):
        result = transducer_greedy(_blank_heavy(frames))
        assert result.frames_consumed == frames
        assert not result.stopped_early


def test_the_decode_subpackage_re_exports_the_transducer() -> None:
    from echoscribe import decode

    assert decode.transducer_greedy is transducer_greedy
    assert decode.TableJoint is TableJoint
    assert decode.JointCache is not None
    for name in ("transducer_greedy", "TransducerConfig", "JointCache", "Joint"):
        assert name in decode.__all__
