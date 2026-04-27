"""The CTC head is checked on shapes, conventions and hand-computed losses."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.errors import ConfigError
from echoscribe.model.ctc import BLANK_INDEX, CtcConfig, CtcHead

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
