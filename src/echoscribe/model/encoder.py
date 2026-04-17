"""A compact bidirectional LSTM encoder.

One stacked-frame BiLSTM and a linear projection -- around a hundred
thousand parameters at its defaults, three orders of magnitude below
anything used for real speech.  The point is the plumbing around it: config
records that serialize, weights that round-trip through a versioned bundle,
forward passes whose shapes the tests pin.

It is bidirectional, and honesty requires saying what that means: this
encoder sees the whole utterance, so it belongs to the *offline* half of the
toolkit.  The streaming runtime elsewhere decodes with precomputed
posteriors frame by frame and never claims this encoder could serve a live
microphone without a chunked variant, which is future work rather than a
hidden feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from echoscribe.errors import ConfigError, ModelError, require
from echoscribe.model.schema import load_bundle, save_bundle
from echoscribe.model.stack import stack_frames


@dataclass(frozen=True)
class EncoderConfig:
    """Shape of the encoder, and exactly what a bundle file restores."""

    input_dim: int = 40
    stack: int = 3
    hidden_dim: int = 64
    output_dim: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("input_dim", self.input_dim),
            ("hidden_dim", self.hidden_dim),
            ("output_dim", self.output_dim),
        ):
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{name} must be a positive integer, got {value!r}",
                ConfigError,
            )
        require(
            isinstance(self.stack, int) and not isinstance(self.stack, bool) and self.stack >= 1,
            f"stack must be an integer >= 1, got {self.stack!r}",
            ConfigError,
        )

    @property
    def stacked_dim(self) -> int:
        """Width the LSTM actually sees after stacking."""
        return self.input_dim * self.stack

    def to_dict(self) -> dict:
        """A JSON-ready mapping of every field."""
        return {
            "input_dim": self.input_dim,
            "stack": self.stack,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> EncoderConfig:
        """Rebuild a config, refusing anything incomplete."""
        for name in ("input_dim", "stack", "hidden_dim", "output_dim"):
            require(
                name in payload,
                f"the encoder config is missing {name!r}",
                ModelError,
            )
        return cls(
            input_dim=int(payload["input_dim"]),
            stack=int(payload["stack"]),
            hidden_dim=int(payload["hidden_dim"]),
            output_dim=int(payload["output_dim"]),
        )


class Encoder(torch.nn.Module):
    """Frame-stacked BiLSTM with a linear projection on top."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = torch.nn.LSTM(
            input_size=config.stacked_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = torch.nn.Linear(2 * config.hidden_dim, config.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode ``(batch, time, input_dim)`` to ``(batch, time, output_dim)``."""
        stacked = stack_frames(features, self.config.stack)
        encoded, _ = self.lstm(stacked)
        return self.projection(encoded)

    def num_parameters(self) -> int:
        """Every scalar this encoder holds, biases included."""
        return sum(parameter.numel() for parameter in self.parameters())

    def save(self, path: Path | str) -> Path:
        """Write this encoder to a versioned bundle file."""
        return save_bundle(path, "encoder", self.config.to_dict(), self.state_dict())

    @classmethod
    def load(cls, path: Path | str) -> Encoder:
        """Rebuild an encoder from a bundle, weights and shape together."""
        payload = load_bundle(path, kind="encoder")
        config = EncoderConfig.from_dict(payload["config"])
        encoder = cls(config)
        encoder.load_state_dict(payload["state"])
        encoder.eval()
        return encoder
