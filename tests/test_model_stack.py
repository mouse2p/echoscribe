"""Frame stacking is checked on a tensor small enough to compute by hand."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.errors import ModelError
from echoscribe.model.stack import stack_frames

pytestmark = pytest.mark.model


def test_the_hand_computed_two_frame_stack() -> None:
    features = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    stacked = stack_frames(features, 2)
    assert stacked.tolist() == [[[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]]]


def test_a_stack_of_one_is_the_identity() -> None:
    features = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert torch.equal(stack_frames(features, 1), features)


def test_stacking_keeps_time_and_multiplies_width() -> None:
    features = torch.zeros(4, 11, 5)
    assert stack_frames(features, 3).shape == (4, 11, 15)


def test_the_oldest_frame_comes_first() -> None:
    features = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    stacked = stack_frames(features, 3)
    # the last window holds frame 0, frame 1, frame 2 in that order
    assert stacked[0, 2].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_flat_feature_vector_is_refused() -> None:
    with pytest.raises(ModelError, match="batch, time, dim"):
        stack_frames(torch.zeros(5), 2)


def test_a_zero_stack_is_refused() -> None:
    with pytest.raises(ModelError, match="integer >= 1"):
        stack_frames(torch.zeros(1, 2, 3), 0)


def test_numpy_input_is_refused() -> None:
    import numpy as np

    with pytest.raises(ModelError, match="must be a tensor"):
        stack_frames(np.zeros((1, 2, 3)), 2)
