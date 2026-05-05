"""Training is checked on a laboratory small enough to stay quick."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.model.train import TrainExample, laboratory_examples

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
