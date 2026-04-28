"""The CTC head is checked on shapes, conventions and hand-computed losses."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.errors import ConfigError, ModelError
from echoscribe.model.ctc import BLANK_INDEX, CtcConfig, CtcHead, ctc_loss

pytestmark = pytest.mark.model


def seeded_head(**overrides: int) -> CtcHead:
    torch.manual_seed(77)
    return CtcHead(CtcConfig(**overrides))


def test_the_head_emits_one_extra_class_for_the_blank() -> None:
    head = seeded_head(input_dim=6, vocab_size=3)
    logits = head(torch.randn(2, 5, 6))
    assert logits.shape == (2, 5, 4)
    assert head.config.output_dim == 4


def test_the_blank_sits_at_index_zero() -> None:
    assert BLANK_INDEX == 0


def test_log_posteriors_are_normalized_per_frame() -> None:
    head = seeded_head(input_dim=6, vocab_size=3)
    posteriors = head.log_posteriors(torch.randn(2, 5, 6))
    totals = posteriors.exp().sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)


def test_a_bad_config_is_refused() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        CtcConfig(vocab_size=0)


def test_the_config_round_trips_through_a_dict() -> None:
    config = CtcConfig(input_dim=12, vocab_size=7)
    assert CtcConfig.from_dict(config.to_dict()) == config


def test_the_head_stays_small() -> None:
    assert seeded_head().num_parameters() < 5_000


def test_a_single_frame_over_a_two_class_alphabet_is_ln_two() -> None:
    import math

    # One frame, uniform logits, one target symbol: the only aligning path
    # spends the frame on that symbol, so the loss is exactly -log(1/2).
    logits = torch.zeros(1, 1, 2)
    loss = ctc_loss(
        logits,
        torch.tensor([[1]]),
        torch.tensor([1]),
        torch.tensor([1]),
    )
    assert torch.isclose(loss, torch.tensor(math.log(2.0)), rtol=1e-5)


def test_a_blank_only_target_needs_no_frames_of_content() -> None:
    # An empty target under uniform logits over 3 frames: every frame must
    # be a blank, so the loss is exactly -3*log(1/4)... over four classes.
    import math

    logits = torch.zeros(1, 3, 4)
    loss = ctc_loss(
        logits, torch.zeros(1, 0, dtype=torch.long), torch.tensor([3]), torch.tensor([0])
    )
    assert torch.isclose(loss, torch.tensor(3 * math.log(4.0)), rtol=1e-5)


def test_wrong_shaped_inputs_are_refused() -> None:
    with pytest.raises(ModelError, match="batch, time, classes"):
        ctc_loss(
            torch.zeros(2, 4),
            torch.zeros(1, 1, dtype=torch.long),
            torch.tensor([1]),
            torch.tensor([1]),
        )
    with pytest.raises(ModelError, match="int64"):
        ctc_loss(torch.zeros(1, 2, 3), torch.zeros(1, 1), torch.tensor([2]), torch.tensor([1]))
    with pytest.raises(ModelError, match="batch of 1"):
        ctc_loss(
            torch.zeros(1, 2, 3),
            torch.zeros(2, 1, dtype=torch.long),
            torch.tensor([2, 2]),
            torch.tensor([1, 1]),
        )
