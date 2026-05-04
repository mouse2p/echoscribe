"""The CTC head: one linear layer, a blank, and the lattice loss.

Index conventions match the NumPy decoders in :mod:`echoscribe.decode`:
index 0 is the blank, and the ``vocab_size`` real symbols occupy indices
1..vocab_size.  The head emits ``vocab_size + 1`` logits per frame, and the
loss wrapper here consumes exactly what the head emits.  Keeping one
convention on both sides means a posterior matrix from this head can be
handed to the NumPy greedy decoder without remapping, which is what the
integration checks do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from echoscribe.errors import ConfigError, ModelError, require
from echoscribe.model.schema import load_bundle, save_bundle

BLANK_INDEX = 0


@dataclass(frozen=True)
class CtcConfig:
    """Shape of the CTC head: what it reads and what it emits."""

    input_dim: int = 64
    vocab_size: int = 40

    def __post_init__(self) -> None:
        for name, value in (("input_dim", self.input_dim), ("vocab_size", self.vocab_size)):
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{name} must be a positive integer, got {value!r}",
                ConfigError,
            )

    @property
    def output_dim(self) -> int:
        """One logit per symbol, plus the blank."""
        return self.vocab_size + 1

    def to_dict(self) -> dict:
        """A JSON-ready mapping of every field."""
        return {"input_dim": self.input_dim, "vocab_size": self.vocab_size}

    @classmethod
    def from_dict(cls, payload: dict) -> CtcConfig:
        """Rebuild a config, refusing anything incomplete."""
        for name in ("input_dim", "vocab_size"):
            require(name in payload, f"the CTC config is missing {name!r}", ModelError)
        return cls(input_dim=int(payload["input_dim"]), vocab_size=int(payload["vocab_size"]))


class CtcHead(torch.nn.Module):
    """A linear projection to symbols-plus-blank, one distribution per frame."""

    def __init__(self, config: CtcConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = torch.nn.Linear(config.input_dim, config.output_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Raw logits for ``(batch, time, input_dim)`` encodings."""
        return self.projection(encoder_output)

    def log_posteriors(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Log-softmaxed frame distributions, the shape decoders read."""
        return torch.log_softmax(self.forward(encoder_output), dim=-1)

    def num_parameters(self) -> int:
        """Every scalar in the head, biases included."""
        return sum(parameter.numel() for parameter in self.parameters())

    def save(self, path: Path | str) -> Path:
        """Write this head to a versioned bundle file."""
        return save_bundle(path, "ctc", self.config.to_dict(), self.state_dict())

    @classmethod
    def load(cls, path: Path | str) -> CtcHead:
        """Rebuild a head from a bundle, weights and shape together."""
        payload = load_bundle(path, kind="ctc")
        config = CtcConfig.from_dict(payload["config"])
        head = cls(config)
        head.load_state_dict(payload["state"])
        head.eval()
        return head


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = BLANK_INDEX,
) -> torch.Tensor:
    """The CTC lattice loss over raw logits, averaged across the batch.

    ``logits`` are ``(batch, time, vocab_size + 1)`` and are log-softmaxed
    here, so callers pass exactly what :class:`CtcHead` emits and never
    pre-normalized values.  ``targets`` hold symbol indices in
    1..vocab_size -- the blank never appears in a reference -- padded to a
    common width whose true extents ``target_lengths`` gives.  Infinities
    from degenerate batches are zeroed rather than propagated, matching the
    convention of the underlying torch loss.
    """
    require(
        logits.dim() == 3,
        f"logits must be (batch, time, classes), got {tuple(logits.shape)}",
        ModelError,
    )
    require(
        logits.shape[-1] >= 2,
        "an alphabet needs the blank plus at least one symbol",
        ModelError,
    )
    require(
        targets.dim() == 2 and targets.shape[0] == logits.shape[0],
        f"targets must be (batch, symbols) for a batch of {logits.shape[0]}, "
        f"got {tuple(targets.shape)}",
        ModelError,
    )
    require(
        targets.dtype == torch.long,
        f"targets must hold int64 symbol indices, got {targets.dtype}",
        ModelError,
    )
    log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths.to(torch.long),
        target_lengths.to(torch.long),
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )
