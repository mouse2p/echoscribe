"""The scripted joint is a fixture, so its shape contract is checked exactly."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.transducer import Joint, TableJoint
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
