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


def test_the_same_seed_builds_identical_encoders() -> None:
    left = seeded_encoder()
    right = seeded_encoder()
    assert all(
        torch.equal(one, two)
        for one, two in zip(left.state_dict().values(), right.state_dict().values(), strict=True)
    )


def test_different_seeds_build_different_encoders() -> None:
    torch.manual_seed(1)
    left = Encoder(EncoderConfig())
    torch.manual_seed(2)
    right = Encoder(EncoderConfig())
    assert any(
        not torch.equal(one, two)
        for one, two in zip(left.state_dict().values(), right.state_dict().values(), strict=True)
    )


def test_the_default_encoder_stays_under_two_hundred_thousand_parameters() -> None:
    encoder = seeded_encoder()
    assert encoder.num_parameters() < 200_000


def test_the_parameter_count_is_the_sum_of_every_tensor() -> None:
    encoder = seeded_encoder(input_dim=6, stack=2, hidden_dim=8, output_dim=4)
    assert encoder.num_parameters() == sum(
        tensor.numel() for tensor in encoder.state_dict().values()
    )


def test_an_incomplete_config_dict_is_refused() -> None:
    from echoscribe.errors import ModelError

    with pytest.raises(ModelError, match="missing 'stack'"):
        EncoderConfig.from_dict({"input_dim": 8, "hidden_dim": 4, "output_dim": 4})
