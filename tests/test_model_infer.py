"""The bridge is checked against the modules it reloads, not against itself."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.errors import ModelError
from echoscribe.model.ctc import CtcConfig, CtcHead
from echoscribe.model.encoder import Encoder, EncoderConfig
from echoscribe.model.infer import ctc_log_posteriors, load_ctc_model
from echoscribe.model.schema import save_bundle
from echoscribe.model.train import TrainHistory

pytestmark = pytest.mark.model


def make_bundle(tmp_path):
    """A seeded encoder+head saved the way the training loop saves them."""
    torch.manual_seed(8)
    encoder = Encoder(EncoderConfig(input_dim=8, stack=2, hidden_dim=6, output_dim=6))
    head = CtcHead(CtcConfig(input_dim=6, vocab_size=4))
    state = {f"encoder.{name}": tensor for name, tensor in encoder.state_dict().items()}
    state.update({f"head.{name}": tensor for name, tensor in head.state_dict().items()})
    target = save_bundle(
        tmp_path / "ctc.pt",
        "ctc",
        {"encoder": encoder.config.to_dict(), "head": head.config.to_dict()},
        state,
        history=TrainHistory(losses=(1.0,), seed=8).to_dict(),
    )
    return target, encoder, head


def test_the_bridge_reloads_what_training_saved(tmp_path) -> None:
    target, encoder, head = make_bundle(tmp_path)
    encoder.eval()
    head.eval()
    restored_encoder, restored_head, config = load_ctc_model(target)
    features = torch.randn(2, 5, 8)
    assert torch.allclose(
        head(encoder(features)), restored_head(restored_encoder(features)), atol=1e-6
    )
    assert config["encoder"]["input_dim"] == 8


def test_the_posteriors_are_numpy_log_distributions(tmp_path) -> None:
    target, _, _ = make_bundle(tmp_path)
    encoder, head, _ = load_ctc_model(target)
    features = np.random.default_rng(2).standard_normal((7, 8))
    posteriors = ctc_log_posteriors(encoder, head, features)
    assert posteriors.shape == (7, 5)
    assert posteriors.dtype == np.float64
    assert np.allclose(np.exp(posteriors).sum(axis=1), 1.0, atol=1e-5)
    assert np.all(posteriors <= 0.0)


def test_a_wrong_feature_width_is_refused(tmp_path) -> None:
    target, _, _ = make_bundle(tmp_path)
    encoder, head, _ = load_ctc_model(target)
    with pytest.raises(ModelError, match="the encoder reads"):
        ctc_log_posteriors(encoder, head, np.zeros((4, 9)))


def test_a_bundle_without_both_configs_is_refused(tmp_path) -> None:
    from echoscribe.model.schema import save_bundle as save

    target = save(tmp_path / "bad.pt", "ctc", {"head": {"input_dim": 4}}, {})
    with pytest.raises(ModelError, match="both the encoder and the head"):
        load_ctc_model(target)


def test_a_bundle_without_encoder_tensors_is_refused(tmp_path) -> None:
    from echoscribe.model.schema import save_bundle as save

    torch.manual_seed(1)
    head = CtcHead(CtcConfig(input_dim=4, vocab_size=2))
    state = {f"head.{name}": tensor for name, tensor in head.state_dict().items()}
    target = save(
        tmp_path / "half.pt",
        "ctc",
        {
            "encoder": EncoderConfig(input_dim=4, stack=1, hidden_dim=4, output_dim=4).to_dict(),
            "head": head.config.to_dict(),
        },
        state,
    )
    with pytest.raises(ModelError, match="no 'encoder' tensors"):
        load_ctc_model(target)
