"""The encoder is checked on shapes and seeds, never on accuracy."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.errors import ConfigError
from echoscribe.model.encoder import Encoder, EncoderConfig

pytestmark = pytest.mark.model


def seeded_encoder(**overrides: int) -> Encoder:
    torch.manual_seed(1234)
    return Encoder(EncoderConfig(**overrides))


def test_the_encoder_keeps_time_and_projects_the_width() -> None:
    encoder = seeded_encoder(input_dim=8, stack=2, hidden_dim=16, output_dim=12)
    output = encoder(torch.randn(2, 9, 8))
    assert output.shape == (2, 9, 12)


def test_the_config_knows_its_stacked_width() -> None:
    assert EncoderConfig(input_dim=8, stack=3).stacked_dim == 24


def test_a_single_frame_encodes_fine() -> None:
    encoder = seeded_encoder(input_dim=6, stack=2, hidden_dim=8, output_dim=4)
    assert encoder(torch.randn(1, 1, 6)).shape == (1, 1, 4)


def test_eval_mode_forward_passes_are_deterministic() -> None:
    encoder = seeded_encoder(input_dim=6, stack=1, hidden_dim=8, output_dim=4)
    encoder.eval()
    features = torch.randn(1, 5, 6)
    assert torch.equal(encoder(features), encoder(features))


def test_bad_config_values_are_refused() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        EncoderConfig(input_dim=0)
    with pytest.raises(ConfigError, match="stack"):
        EncoderConfig(stack=0)


def test_the_config_round_trips_through_a_dict() -> None:
    config = EncoderConfig(input_dim=12, stack=2, hidden_dim=8, output_dim=6)
    assert EncoderConfig.from_dict(config.to_dict()) == config
