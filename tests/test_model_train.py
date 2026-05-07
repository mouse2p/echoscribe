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


@pytest.mark.slow
def test_ctc_training_decreases_the_loss_on_synthetic_speech() -> None:
    from echoscribe.model.ctc import CtcConfig, CtcHead
    from echoscribe.model.encoder import Encoder, EncoderConfig
    from echoscribe.model.train import train_ctc

    examples, vocabulary = laboratory_examples(seed=11, utterances=16)
    torch.manual_seed(3)
    encoder = Encoder(EncoderConfig(input_dim=40, stack=2, hidden_dim=48, output_dim=48))
    head = CtcHead(CtcConfig(input_dim=48, vocab_size=len(vocabulary) - 1))
    history = train_ctc(
        encoder, head, examples, TrainConfig(epochs=8, learning_rate=0.01, batch_size=4)
    )
    assert history.epochs_completed == 8
    assert history.improved
    assert history.last_loss < history.first_loss * 0.9


def test_the_same_seed_trains_the_same_losses() -> None:
    from echoscribe.model.ctc import CtcConfig, CtcHead
    from echoscribe.model.encoder import Encoder, EncoderConfig
    from echoscribe.model.train import train_ctc

    examples, vocabulary = laboratory_examples(seed=12, utterances=8)

    def fresh() -> tuple:
        torch.manual_seed(4)
        encoder = Encoder(EncoderConfig(input_dim=40, stack=1, hidden_dim=16, output_dim=16))
        head = CtcHead(CtcConfig(input_dim=16, vocab_size=len(vocabulary) - 1))
        return encoder, head

    config = TrainConfig(epochs=2, batch_size=4, seed=77)
    left = train_ctc(*fresh(), examples, config)
    right = train_ctc(*fresh(), examples, config)
    assert left.losses == right.losses
    assert left.losses[0] != left.losses[-1] or left.epochs_completed == 1


@pytest.mark.slow
def test_transducer_training_decreases_the_loss_on_synthetic_speech() -> None:
    from echoscribe.model.encoder import Encoder, EncoderConfig
    from echoscribe.model.train import train_transducer
    from echoscribe.model.transducer import Joint, JointConfig, Predictor, PredictorConfig

    examples, vocabulary = laboratory_examples(seed=13, utterances=12)
    vocab_size = len(vocabulary) - 1
    torch.manual_seed(5)
    encoder = Encoder(EncoderConfig(input_dim=40, stack=2, hidden_dim=32, output_dim=32))
    predictor = Predictor(PredictorConfig(vocab_size=vocab_size, embed_dim=16, hidden_dim=32))
    joint = Joint(
        JointConfig(encoder_dim=32, predictor_dim=32, vocab_size=vocab_size, hidden_dim=32)
    )
    history = train_transducer(
        encoder,
        predictor,
        joint,
        examples,
        TrainConfig(epochs=4, learning_rate=0.01, batch_size=4, seed=99),
    )
    assert history.epochs_completed == 4
    assert history.improved
    assert history.last_loss < history.first_loss * 0.95
