"""Predictor and joint are checked on shapes, states and hand-computed sums."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.errors import ConfigError, ModelError
from echoscribe.model.transducer import (
    BOS_INDEX,
    Joint,
    JointConfig,
    Predictor,
    PredictorConfig,
    transducer_loss,
)

pytestmark = pytest.mark.model


def seeded_predictor(**overrides: int) -> Predictor:
    torch.manual_seed(4242)
    return Predictor(PredictorConfig(**overrides))


def test_the_forward_pass_yields_one_state_per_prefix() -> None:
    predictor = seeded_predictor(vocab_size=5, embed_dim=4, hidden_dim=6)
    states = predictor(torch.tensor([[1, 2, 3], [4, 5, 1]]))
    assert states.shape == (2, 4, 6)


def test_the_embedding_has_one_row_per_symbol_plus_the_start() -> None:
    predictor = seeded_predictor(vocab_size=5, embed_dim=4, hidden_dim=6)
    assert predictor.embedding.weight.shape == (6, 4)
    assert BOS_INDEX == 0


def test_stepping_by_hand_matches_the_forward_pass() -> None:
    predictor = seeded_predictor(vocab_size=5, embed_dim=4, hidden_dim=6)
    symbols = torch.tensor([[2, 4]])
    states = predictor(symbols)
    state = predictor.initial_state(1)
    hand = []
    for symbol in [BOS_INDEX, 2, 4]:
        state = predictor.step(torch.tensor([symbol]), state)
        hand.append(state[0])
    assert torch.allclose(torch.stack(hand, dim=1), states, atol=1e-6)


def test_the_same_seed_builds_identical_predictors() -> None:
    left = seeded_predictor(vocab_size=5, embed_dim=4, hidden_dim=6)
    right = seeded_predictor(vocab_size=5, embed_dim=4, hidden_dim=6)
    assert all(
        torch.equal(one, two)
        for one, two in zip(left.state_dict().values(), right.state_dict().values(), strict=True)
    )


def test_a_float_symbols_tensor_is_refused() -> None:
    predictor = seeded_predictor(vocab_size=5, embed_dim=4, hidden_dim=6)
    with pytest.raises(ModelError, match="int64"):
        predictor(torch.zeros(1, 3))


def test_a_bad_config_is_refused() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        PredictorConfig(embed_dim=0)


def test_the_config_round_trips_through_a_dict() -> None:
    config = PredictorConfig(vocab_size=9, embed_dim=5, hidden_dim=7)
    assert PredictorConfig.from_dict(config.to_dict()) == config


def seeded_joint(**overrides: int) -> Joint:
    torch.manual_seed(31337)
    return Joint(JointConfig(**overrides))


def test_the_joint_scores_every_lattice_point() -> None:
    joint = seeded_joint(encoder_dim=4, predictor_dim=6, vocab_size=3, hidden_dim=5)
    logits = joint(torch.randn(2, 7, 4), torch.randn(2, 3, 6))
    assert logits.shape == (2, 7, 3, 4)


def test_moving_the_encoder_side_moves_the_logits() -> None:
    joint = seeded_joint(encoder_dim=4, predictor_dim=6, vocab_size=3, hidden_dim=5)
    predictor_states = torch.randn(1, 3, 6)
    left = joint(torch.randn(1, 5, 4), predictor_states)
    right = joint(torch.randn(1, 5, 4), predictor_states)
    assert not torch.allclose(left, right)


def test_moving_the_predictor_side_moves_the_logits() -> None:
    joint = seeded_joint(encoder_dim=4, predictor_dim=6, vocab_size=3, hidden_dim=5)
    encoder_states = torch.randn(1, 5, 4)
    left = joint(encoder_states, torch.randn(1, 3, 6))
    right = joint(encoder_states, torch.randn(1, 3, 6))
    assert not torch.allclose(left, right)


def test_a_batch_mismatch_is_refused() -> None:
    joint = seeded_joint(encoder_dim=4, predictor_dim=6, vocab_size=3, hidden_dim=5)
    with pytest.raises(ModelError, match="batches of"):
        joint(torch.randn(2, 5, 4), torch.randn(3, 3, 6))


def test_a_flat_encoder_state_is_refused() -> None:
    joint = seeded_joint(encoder_dim=4, predictor_dim=6, vocab_size=3, hidden_dim=5)
    with pytest.raises(ModelError, match="batch, time, dim"):
        joint(torch.randn(5, 4), torch.randn(1, 3, 6))


def test_the_joint_config_round_trips() -> None:
    config = JointConfig(encoder_dim=3, predictor_dim=4, vocab_size=5, hidden_dim=6)
    assert JointConfig.from_dict(config.to_dict()) == config
    assert config.output_dim == 6


def test_two_frames_one_symbol_uniform_logits() -> None:
    import math

    # Four classes, two frames, one target symbol: exactly two aligning paths
    # (symbol then blank blank, or blank then symbol then blank), each with
    # probability (1/4)**3, so the loss is -log(2 / 64) = log(32).
    logits = torch.zeros(1, 2, 2, 4)
    loss = transducer_loss(logits, torch.tensor([[1]]), torch.tensor([2]), torch.tensor([1]))
    assert torch.isclose(loss, torch.tensor(math.log(32.0)), rtol=1e-5)


def test_an_empty_target_is_the_all_blank_path() -> None:
    import math

    logits = torch.zeros(1, 3, 1, 4)
    loss = transducer_loss(
        logits, torch.zeros(1, 0, dtype=torch.long), torch.tensor([3]), torch.tensor([0])
    )
    assert torch.isclose(loss, torch.tensor(3 * math.log(4.0)), rtol=1e-5)


def test_a_planted_joint_beats_a_uniform_one() -> None:
    # Blank-dominated logits with the symbol planted at frame 1 of 4.
    planted = torch.full((1, 4, 2, 5), -6.0)
    planted[:, :, :, 0] = 6.0
    planted[0, 1, 0, 2] = 6.0
    uniform = torch.zeros(1, 4, 2, 5)
    targets = torch.tensor([[2]])
    lengths = torch.tensor([4])
    symbols = torch.tensor([1])
    planted_loss = transducer_loss(planted, targets, lengths, symbols)
    uniform_loss = transducer_loss(uniform, targets, lengths, symbols)
    assert planted_loss < uniform_loss


def test_the_loss_is_differentiable_through_the_joint() -> None:
    joint = seeded_joint(encoder_dim=4, predictor_dim=6, vocab_size=3, hidden_dim=5)
    logits = joint(torch.randn(1, 5, 4), torch.randn(1, 3, 6))
    loss = transducer_loss(logits, torch.tensor([[1, 2]]), torch.tensor([5]), torch.tensor([2]))
    loss.backward()
    gradient = joint.output.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_a_states_targets_mismatch_is_refused() -> None:
    with pytest.raises(ModelError, match="one state per prefix"):
        transducer_loss(
            torch.zeros(1, 2, 4, 3),
            torch.tensor([[1]]),
            torch.tensor([2]),
            torch.tensor([1]),
        )


def test_a_flat_joint_tensor_is_refused() -> None:
    with pytest.raises(ModelError, match="batch, time, states, classes"):
        transducer_loss(
            torch.zeros(2, 2, 3), torch.tensor([[1]]), torch.tensor([2]), torch.tensor([1])
        )
