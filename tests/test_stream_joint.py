"""An appendable joint has to look exactly like a fixed one to a decoder."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.transducer import Joint
from echoscribe.errors import DecodeError, StreamError
from echoscribe.stream.joint import AppendableJoint


def _logits(frames: int = 8, symbols: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-2.0, 2.0, size=(frames, symbols))


def test_an_empty_joint_has_no_frames() -> None:
    joint = AppendableJoint(4)
    assert joint.num_frames == 0
    assert joint.num_symbols == 4


def test_appending_grows_the_frame_count() -> None:
    joint = AppendableJoint(4)
    assert joint.append(_logits(3)) == 3
    assert joint.append(_logits(2)) == 5
    assert joint.num_frames == 5


def test_appended_rows_are_normalised() -> None:
    joint = AppendableJoint(4)
    joint.append(_logits(5))
    for frame in range(5):
        row = joint(frame, ())
        assert np.all(row <= 1e-12)
        assert np.exp(row).sum() == pytest.approx(1.0, rel=1e-12)


def test_an_appendable_joint_satisfies_the_joint_protocol() -> None:
    joint = AppendableJoint(4)
    joint.append(_logits(2))
    assert isinstance(joint, Joint)


def test_reading_an_unappended_frame_is_refused() -> None:
    joint = AppendableJoint(4)
    joint.append(_logits(3))
    with pytest.raises(DecodeError, match="outside the 3 frames appended so far"):
        joint(3, ())


def test_a_wrong_symbol_width_is_refused() -> None:
    joint = AppendableJoint(4)
    with pytest.raises(StreamError, match="the joint has 4"):
        joint.append(_logits(2, symbols=5))


def test_a_flat_vector_is_refused() -> None:
    with pytest.raises(StreamError, match="append needs a"):
        AppendableJoint(4).append(np.zeros(4))


def test_a_non_finite_logit_is_refused() -> None:
    logits = _logits(2)
    logits[1, 0] = np.inf
    with pytest.raises(StreamError, match="must be finite"):
        AppendableJoint(4).append(logits)


def test_a_bad_symbol_count_is_refused() -> None:
    with pytest.raises(StreamError, match="num_symbols must be positive"):
        AppendableJoint(0)


def test_a_reset_forgets_every_frame() -> None:
    joint = AppendableJoint(4)
    joint.append(_logits(4))
    joint.reset()
    assert joint.num_frames == 0


def test_a_predictor_shifts_the_choice() -> None:
    logits = np.array([[0.0, 0.1, 0.0, 0.0]])

    def prefer_symbol_two(history):
        return np.array([0.0, 0.0, 5.0, 0.0])

    plain = AppendableJoint(4)
    plain.append(logits)
    biased = AppendableJoint(4, predictor=prefer_symbol_two)
    biased.append(logits)
    assert int(np.argmax(plain(0, ()))) == 1
    assert int(np.argmax(biased(0, ()))) == 2


def test_a_predictor_sees_the_history() -> None:
    seen = []

    def record(history):
        seen.append(tuple(history))
        return np.zeros(3)

    joint = AppendableJoint(3, predictor=record)
    joint.append(_logits(2, symbols=3))
    joint(0, ())
    joint(1, (2, 2))
    assert seen == [(), (2, 2)]


def test_a_predictor_of_the_wrong_width_is_refused() -> None:
    joint = AppendableJoint(4, predictor=lambda history: np.zeros(3))
    joint.append(_logits(2))
    with pytest.raises(DecodeError, match="expected 4 symbols"):
        joint(0, ())


def test_without_a_predictor_the_history_is_ignored() -> None:
    joint = AppendableJoint(4)
    joint.append(_logits(3))
    assert np.array_equal(joint(1, ()), joint(1, (1, 2, 3)))
