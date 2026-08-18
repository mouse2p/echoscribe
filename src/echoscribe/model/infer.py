"""From bundle files back to the posterior matrices the NumPy decoders read.

This is the seam the architecture promises: a trained model is worth exactly
the posterior matrices it produces, and this module is the only place in the
package where a torch tensor becomes a NumPy array that the decoders
consume.  Everything runs under ``torch.no_grad()``, on CPU, in eval mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from echoscribe.errors import ModelError, require
from echoscribe.model.ctc import CtcConfig, CtcHead
from echoscribe.model.encoder import Encoder, EncoderConfig
from echoscribe.model.schema import load_bundle
from echoscribe.types import FEATURE_DTYPE, FloatArray, require_rank


def load_ctc_model(path: Path | str) -> tuple[Encoder, CtcHead, dict]:
    """The encoder and head a training bundle saved, both in eval mode."""
    payload = load_bundle(path, kind="ctc")
    config = payload["config"]
    require(
        isinstance(config, dict) and "encoder" in config and "head" in config,
        "the bundle config must describe both the encoder and the head",
        ModelError,
    )
    encoder = Encoder(EncoderConfig.from_dict(config["encoder"]))
    head = CtcHead(CtcConfig.from_dict(config["head"]))
    state = payload["state"]
    encoder.load_state_dict(prefixed_state(state, "encoder"))
    head.load_state_dict(prefixed_state(state, "head"))
    encoder.eval()
    head.eval()
    return encoder, head, config


def prefixed_state(state: dict, prefix: str) -> dict:
    """The slice of a combined state dict that belongs to one module."""
    entries = {
        key[len(prefix) + 1 :]: value
        for key, value in state.items()
        if key.startswith(prefix + ".")
    }
    require(bool(entries), f"the bundle state holds no {prefix!r} tensors", ModelError)
    return entries


def ctc_log_posteriors(encoder: Encoder, head: CtcHead, features: np.ndarray) -> FloatArray:
    """One utterance's ``(frames, symbols)`` log-posteriors, gradient-free.

    The result is exactly what :mod:`echoscribe.decode` consumes: natural-log
    rows over an alphabet with the blank at index 0, as float64 NumPy.  The
    torch side never leaks past this function.
    """
    matrix = np.asarray(features, dtype=FEATURE_DTYPE)
    require_rank("features", matrix, 2)
    require(
        matrix.shape[1] == encoder.config.input_dim,
        f"features have {matrix.shape[1]} bins but the encoder reads {encoder.config.input_dim}",
        ModelError,
    )
    with torch.no_grad():
        tensor = torch.from_numpy(matrix.astype(np.float32))[None, :, :]
        logits = head(encoder(tensor))
        posteriors = torch.log_softmax(logits, dim=-1)[0]
    return np.asarray(posteriors.numpy(), dtype=FEATURE_DTYPE)
