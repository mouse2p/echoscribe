"""Training is checked on a laboratory small enough to stay quick."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.errors import ConfigError
from echoscribe.model.train import (
    TrainConfig,
    TrainExample,
    TrainHistory,
    laboratory_examples,
    pad_batch,
    should_stop,
)

pytestmark = pytest.mark.model


def test_the_same_seed_renders_the_same_examples() -> None:
    left, _ = laboratory_examples(seed=5, utterances=4)
    right, _ = laboratory_examples(seed=5, utterances=4)
    assert [example.utterance_id for example in left] == [example.utterance_id for example in right]
    for one, two in zip(left, right, strict=True):
        assert torch.equal(one.features, two.features)
        assert torch.equal(one.targets, two.targets)


def test_a_different_seed_renders_different_audio() -> None:
    left, _ = laboratory_examples(seed=5, utterances=4)
    right, _ = laboratory_examples(seed=6, utterances=4)
    assert any(
        not torch.equal(one.features, two.features) for one, two in zip(left, right, strict=True)
    )


def test_targets_are_vocabulary_ids_and_never_the_blank() -> None:
    examples, vocabulary = laboratory_examples(seed=7, utterances=6)
    assert vocabulary.blank_id == 0
    for example in examples:
        assert example.targets.numel() >= 1
        assert int(example.targets.min()) > 0
        assert int(example.targets.max()) < len(vocabulary)


def test_features_are_finite_and_centered() -> None:
    examples, _ = laboratory_examples(seed=8, utterances=3)
    for example in examples:
        assert torch.isfinite(example.features).all()
        assert abs(float(example.features.mean())) < 0.1


def test_examples_carry_their_utterance_id() -> None:
    examples, _ = laboratory_examples(seed=9, utterances=2)
    assert all(example.utterance_id for example in examples)
    assert isinstance(examples[0], TrainExample)


def example(identifier: str, frames: int, symbols: int, dim: int = 4) -> TrainExample:
    return TrainExample(
        identifier,
        torch.zeros(frames, dim),
        torch.arange(1, symbols + 1, dtype=torch.int64),
    )


def test_padding_keeps_the_true_extents() -> None:
    features, targets, input_lengths, target_lengths = pad_batch(
        [example("a", 5, 2), example("b", 3, 4)]
    )
    assert features.shape == (2, 5, 4)
    assert targets.shape == (2, 4)
    assert input_lengths.tolist() == [5, 3]
    assert target_lengths.tolist() == [2, 4]


def test_padding_fills_with_zeros() -> None:
    features, targets, _, _ = pad_batch([example("a", 5, 2), example("b", 3, 4)])
    assert features[1, 3:].abs().sum() == 0.0
    assert targets[0, 2:].tolist() == [0, 0]


def test_a_single_example_pads_nothing() -> None:
    features, targets, input_lengths, target_lengths = pad_batch([example("a", 3, 2)])
    assert features.shape == (1, 3, 4)
    assert targets.shape == (1, 2)
    assert input_lengths.tolist() == [3]
    assert target_lengths.tolist() == [2]


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one example"):
        pad_batch([])


def test_mixed_feature_widths_are_refused() -> None:
    with pytest.raises(ConfigError, match="disagree on feature width"):
        pad_batch([example("a", 3, 2, dim=4), example("b", 3, 2, dim=5)])


def test_the_history_reports_its_improvement() -> None:
    history = TrainHistory(losses=(4.0, 3.0, 2.0), seed=1)
    assert history.first_loss == 4.0
    assert history.last_loss == 2.0
    assert history.best_loss == 2.0
    assert history.improved
    assert history.total_improvement == 2.0
    assert history.epochs_completed == 3


def test_a_flat_history_does_not_claim_improvement() -> None:
    history = TrainHistory(losses=(2.0, 2.0), seed=1)
    assert not history.improved
    assert history.total_improvement == 0.0


def test_the_history_round_trips_through_a_dict() -> None:
    history = TrainHistory(losses=(4.0, 2.0), seed=9)
    payload = history.to_dict()
    assert payload["epochs_completed"] == 2
    assert TrainHistory.from_dict(payload) == history


def test_should_stop_waits_for_a_full_window() -> None:
    assert not should_stop([3.0, 2.0], 2)
    assert not should_stop([], 3)


def test_should_stop_fires_after_patience_flat_epochs() -> None:
    assert should_stop([3.0, 2.0, 1.0, 1.5, 1.6], 2)


def test_should_stop_stays_quiet_while_the_loss_falls() -> None:
    assert not should_stop([3.0, 2.5, 2.0, 1.5, 1.0], 2)


def test_a_bad_patience_is_refused() -> None:
    with pytest.raises(ConfigError, match="patience"):
        should_stop([1.0, 2.0], 0)


def test_a_train_config_rejects_zero_epochs() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        TrainConfig(epochs=0)


def test_a_train_config_rejects_a_negative_rate() -> None:
    with pytest.raises(ConfigError, match="positive"):
        TrainConfig(learning_rate=-0.1)


def test_a_train_config_round_trips_through_a_dict() -> None:
    config = TrainConfig(epochs=3, patience=2)
    assert TrainConfig(**config.to_dict()) == config
