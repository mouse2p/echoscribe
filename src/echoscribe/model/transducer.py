"""The transducer's predictor and joint, after Graves (2012).

A transducer scores one symbol per lattice position ``(t, u)``: ``t`` counts
acoustic frames, ``u`` counts symbols emitted so far.  The predictor turns
the symbol history into a state per position ``u``; the joint combines that
state with the encoder's frame state into a distribution over the next
symbol or the blank.  The conventions match :mod:`echoscribe.model.ctc`:
the blank is index 0, real symbols are 1..vocab_size, and the joint emits
``vocab_size + 1`` logits.

The predictor embeds the blank row as its start-of-sequence input.  That is
safe because a transducer never emits the blank as a *label* -- blanks only
advance time -- so embedding row 0 has no second meaning to conflict with.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from echoscribe.errors import ConfigError, ModelError, require

BOS_INDEX = 0


@dataclass(frozen=True)
class PredictorConfig:
    """Shape of the predictor: its alphabet and its two widths."""

    vocab_size: int = 40
    embed_dim: int = 32
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("embed_dim", self.embed_dim),
            ("hidden_dim", self.hidden_dim),
        ):
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{name} must be a positive integer, got {value!r}",
                ConfigError,
            )

    def to_dict(self) -> dict:
        """A JSON-ready mapping of every field."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> PredictorConfig:
        """Rebuild a config, refusing anything incomplete."""
        for name in ("vocab_size", "embed_dim", "hidden_dim"):
            require(name in payload, f"the predictor config is missing {name!r}", ModelError)
        return cls(
            vocab_size=int(payload["vocab_size"]),
            embed_dim=int(payload["embed_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
        )


class Predictor(torch.nn.Module):
    """An embedding plus one LSTM cell, advanced one symbol at a time."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config.vocab_size + 1, config.embed_dim)
        self.cell = torch.nn.LSTMCell(config.embed_dim, config.hidden_dim)

    def initial_state(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The zero state every sequence starts from."""
        weight = self.embedding.weight
        zeros = torch.zeros(batch, self.config.hidden_dim, dtype=weight.dtype)
        return zeros, torch.zeros_like(zeros)

    def step(
        self, symbols: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume one symbol per sequence and return the new cell state."""
        return self.cell(self.embedding(symbols), state)

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        """States for every prefix: ``(batch, symbols + 1, hidden)``.

        State 0 has consumed only the start input; state ``u`` has consumed
        the first ``u`` symbols -- exactly the states the joint indexes by
        lattice position.
        """
        require(
            symbols.dim() == 2 and symbols.dtype == torch.long,
            f"symbols must be a (batch, length) int64 tensor, got {tuple(symbols.shape)}",
            ModelError,
        )
        batch = symbols.shape[0]
        start = torch.full((batch, 1), BOS_INDEX, dtype=symbols.dtype)
        inputs = torch.cat([start, symbols], dim=1)
        state = self.initial_state(batch)
        states = []
        for position in range(inputs.shape[1]):
            state = self.step(inputs[:, position], state)
            states.append(state[0])
        return torch.stack(states, dim=1)


@dataclass(frozen=True)
class JointConfig:
    """Shape of the joint: both input widths, the alphabet, the mixing width."""

    encoder_dim: int = 64
    predictor_dim: int = 64
    vocab_size: int = 40
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("encoder_dim", self.encoder_dim),
            ("predictor_dim", self.predictor_dim),
            ("vocab_size", self.vocab_size),
            ("hidden_dim", self.hidden_dim),
        ):
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
        return {
            "encoder_dim": self.encoder_dim,
            "predictor_dim": self.predictor_dim,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> JointConfig:
        """Rebuild a config, refusing anything incomplete."""
        for name in ("encoder_dim", "predictor_dim", "vocab_size", "hidden_dim"):
            require(name in payload, f"the joint config is missing {name!r}", ModelError)
        return cls(
            encoder_dim=int(payload["encoder_dim"]),
            predictor_dim=int(payload["predictor_dim"]),
            vocab_size=int(payload["vocab_size"]),
            hidden_dim=int(payload["hidden_dim"]),
        )


class Joint(torch.nn.Module):
    """The transducer joint: a tanh-MLP over one frame and one prefix state.

    ``logits[b, t, u] = W_out * tanh(W_enc * e[b, t] + W_pred * p[b, u])``

    -- the structure Graves (2012) defined the transducer with.  The sum
    inside the tanh is what makes the joint cheap to evaluate over a whole
    ``(t, u)`` lattice at once: both projections are computed once per row
    and broadcast against each other.
    """

    def __init__(self, config: JointConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder_projection = torch.nn.Linear(config.encoder_dim, config.hidden_dim)
        self.predictor_projection = torch.nn.Linear(config.predictor_dim, config.hidden_dim)
        self.output = torch.nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, encoder_states: torch.Tensor, predictor_states: torch.Tensor) -> torch.Tensor:
        """Score every lattice point: ``(B, T, He)`` x ``(B, U, Hp)`` -> ``(B, T, U, V+1)``."""
        require(
            encoder_states.dim() == 3 and predictor_states.dim() == 3,
            "the joint needs (batch, time, dim) and (batch, prefix, dim) states",
            ModelError,
        )
        require(
            encoder_states.shape[0] == predictor_states.shape[0],
            f"the joint got batches of {encoder_states.shape[0]} and {predictor_states.shape[0]}",
            ModelError,
        )
        mixed = (
            self.encoder_projection(encoder_states)[:, :, None, :]
            + self.predictor_projection(predictor_states)[:, None, :, :]
        )
        return self.output(torch.tanh(mixed))

    def num_parameters(self) -> int:
        """Every scalar in the joint, biases included."""
        return sum(parameter.numel() for parameter in self.parameters())
