"""Seeded training loops over the synthetic laboratory.

What a decreasing loss here proves, and what it does not.  It proves the
plumbing is real: the forward pass, the lattice loss and the backward pass
agree with each other, and gradients on planted data point downhill.  It
proves nothing about recognizing speech -- the vocabulary is tiny, the audio
is rendered formants, and the models are a few tens of thousands of
parameters.  Every number this module produces is reproducible from its seed,
on CPU, with no downloaded weights anywhere in the loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from echoscribe.cmvn import CmvnStatistics, apply_cmvn
from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.features import FeatureConfig, extract_log_mel
from echoscribe.framing import FrameConfig
from echoscribe.model.ctc import CtcHead, ctc_loss
from echoscribe.model.encoder import Encoder
from echoscribe.model.transducer import Joint, Predictor, transducer_loss
from echoscribe.seeds import Seed
from echoscribe.synth.corpus import CorpusConfig, generate_corpus
from echoscribe.synth.render import RenderConfig, render_utterance
from echoscribe.text.units import unit_vocabulary, utterance_units
from echoscribe.text.vocab import Vocabulary


@dataclass(frozen=True)
class TrainExample:
    """One utterance as the loops consume it: normalized frames and unit ids."""

    utterance_id: str
    features: torch.Tensor
    targets: torch.Tensor


def laboratory_examples(
    seed: int = 2025,
    utterances: int = 16,
    num_mel_bins: int = 40,
    min_tokens: int = 1,
    max_tokens: int = 3,
    level: str = "syllable",
) -> tuple[tuple[TrainExample, ...], Vocabulary]:
    """Render a seeded corpus into ``(features, targets)`` pairs.

    Features are log-mels normalized by their own utterance's CMVN
    statistics -- per utterance rather than per corpus, so an example is the
    same tensor whichever batch it later lands in.  Targets are modelling-unit
    ids from the lexicon-derived vocabulary, whose blank sits at index 0 and
    therefore never appears among them.
    """
    require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        f"seed must be a non-negative integer, got {seed!r}",
        ConfigError,
    )
    corpus = generate_corpus(
        CorpusConfig(
            utterances=utterances,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            seed=Seed(seed, "model.train.laboratory"),
        )
    )
    vocabulary = unit_vocabulary(level=level)
    render = RenderConfig()
    feature_config = FeatureConfig(
        num_mel_bins=num_mel_bins, frame=FrameConfig(sample_rate=render.sample_rate)
    )
    examples: list[TrainExample] = []
    for utterance in corpus:
        audio = render_utterance(utterance, render)
        features = extract_log_mel(audio, feature_config)
        statistics = CmvnStatistics(dimension=num_mel_bins)
        statistics.update(features)
        normalized = apply_cmvn(features, statistics)
        units = utterance_units(utterance.tokens, utterance.languages, level=level)
        ids = vocabulary.encode(units, allow_unk=False)
        examples.append(
            TrainExample(
                utterance_id=utterance.utterance_id,
                features=torch.tensor(normalized, dtype=torch.float32),
                targets=torch.tensor(ids, dtype=torch.int64),
            )
        )
    return tuple(examples), vocabulary


def pad_batch(
    examples: Sequence[TrainExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One right-padded batch plus the true extents of both sides.

    Padding value zero is safe on both sides: padded frames are excluded by
    ``input_lengths`` and padded symbols are index 0, the blank, which no
    loss reads past ``target_lengths``.
    """
    require(len(examples) > 0, "a batch needs at least one example", ConfigError)
    widths = {example.features.shape[1] for example in examples}
    require(
        len(widths) == 1,
        f"examples disagree on feature width: {sorted(widths)}",
        ConfigError,
    )
    longest_frames = max(example.features.shape[0] for example in examples)
    longest_targets = max(example.targets.shape[0] for example in examples)
    features = torch.zeros(len(examples), longest_frames, widths.pop())
    targets = torch.zeros(len(examples), longest_targets, dtype=torch.int64)
    input_lengths = torch.zeros(len(examples), dtype=torch.long)
    target_lengths = torch.zeros(len(examples), dtype=torch.long)
    for index, example in enumerate(examples):
        frames = example.features.shape[0]
        symbols = example.targets.shape[0]
        features[index, :frames] = example.features
        targets[index, :symbols] = example.targets
        input_lengths[index] = frames
        target_lengths[index] = symbols
    return features, targets, input_lengths, target_lengths


LOSS_DECIMALS = 6


@dataclass(frozen=True)
class TrainConfig:
    """Loop knobs, validated once, serialized with any bundle they produce."""

    epochs: int = 12
    learning_rate: float = 0.01
    batch_size: int = 4
    seed: int = 20250
    gradient_clip: float = 5.0
    patience: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{name} must be a positive integer, got {value!r}",
                ConfigError,
            )
        require_positive("learning_rate", self.learning_rate, ConfigError)
        require_positive("gradient_clip", self.gradient_clip, ConfigError)
        if self.patience is not None:
            require(
                isinstance(self.patience, int)
                and not isinstance(self.patience, bool)
                and self.patience >= 1,
                f"patience must be an integer >= 1 or None, got {self.patience!r}",
                ConfigError,
            )

    def to_dict(self) -> dict:
        """A JSON-ready mapping of every knob."""
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "gradient_clip": self.gradient_clip,
            "patience": self.patience,
        }


@dataclass(frozen=True)
class TrainHistory:
    """Per-epoch mean losses, rounded where they are recorded."""

    losses: tuple[float, ...]
    seed: int

    @property
    def epochs_completed(self) -> int:
        """How many epochs actually ran, which patience may cut short."""
        return len(self.losses)

    @property
    def first_loss(self) -> float:
        """The loss of the first epoch."""
        return self.losses[0]

    @property
    def last_loss(self) -> float:
        """The loss of the last epoch that ran."""
        return self.losses[-1]

    @property
    def best_loss(self) -> float:
        """The lowest epoch mean seen."""
        return min(self.losses)

    @property
    def improved(self) -> bool:
        """Whether the last epoch beat the first -- the honest headline."""
        return self.last_loss < self.first_loss

    @property
    def total_improvement(self) -> float:
        """first minus last, rounded at the source."""
        return round(self.first_loss - self.last_loss, LOSS_DECIMALS)

    def to_dict(self) -> dict:
        """A JSON-ready mapping, suitable for a bundle's history slot."""
        return {
            "seed": self.seed,
            "losses": list(self.losses),
            "epochs_completed": self.epochs_completed,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TrainHistory:
        """Rebuild a history from :meth:`to_dict`."""
        return cls(
            losses=tuple(float(loss) for loss in payload["losses"]),
            seed=int(payload["seed"]),
        )


def should_stop(losses: Sequence[float], patience: int) -> bool:
    """True when the last ``patience`` epochs all failed to beat the earlier best.

    A pure function of the loss list so the stopping rule can be tested
    without running a single gradient step.
    """
    require(
        isinstance(patience, int) and not isinstance(patience, bool) and patience >= 1,
        f"patience must be an integer >= 1, got {patience!r}",
        ConfigError,
    )
    if len(losses) <= patience:
        return False
    best_before = min(losses[: len(losses) - patience])
    return all(loss >= best_before for loss in losses[len(losses) - patience :])


def train_ctc(
    encoder: Encoder,
    head: CtcHead,
    examples: Sequence[TrainExample],
    config: TrainConfig | None = None,
) -> TrainHistory:
    """Train an encoder and CTC head, recording one mean loss per epoch.

    Seeded twice over: ``torch.manual_seed`` for anything the loop itself
    draws, and a dedicated generator for the per-epoch shuffle order, so the
    epoch sequence is a function of the config seed alone and does not
    depend on what the caller did before.
    """
    settings = TrainConfig() if config is None else config
    require(len(examples) > 0, "training needs at least one example", ConfigError)
    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    parameters = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
    losses: list[float] = []
    for _ in range(settings.epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        total, batches = 0.0, 0
        for start in range(0, len(order), settings.batch_size):
            chunk = [examples[index] for index in order[start : start + settings.batch_size]]
            features, targets, input_lengths, target_lengths = pad_batch(chunk)
            optimizer.zero_grad()
            loss = ctc_loss(head(encoder(features)), targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(round(total / batches, LOSS_DECIMALS))
        if settings.patience is not None and should_stop(losses, settings.patience):
            break
    return TrainHistory(losses=tuple(losses), seed=settings.seed)


def train_transducer(
    encoder: Encoder,
    predictor: Predictor,
    joint: Joint,
    examples: Sequence[TrainExample],
    config: TrainConfig | None = None,
) -> TrainHistory:
    """Train encoder, predictor and joint against the exact RNN-T loss.

    The predictor sees each batch's padded targets and produces one state per
    prefix, which is exactly the state axis the joint scores the lattice
    over; nothing about the loop is transducer-specific except the loss line,
    and that is the point of keeping the two loops side by side.
    """
    settings = TrainConfig() if config is None else config
    require(len(examples) > 0, "training needs at least one example", ConfigError)
    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    parameters = (
        list(encoder.parameters()) + list(predictor.parameters()) + list(joint.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
    losses: list[float] = []
    for _ in range(settings.epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        total, batches = 0.0, 0
        for start in range(0, len(order), settings.batch_size):
            chunk = [examples[index] for index in order[start : start + settings.batch_size]]
            features, targets, input_lengths, target_lengths = pad_batch(chunk)
            optimizer.zero_grad()
            logits = joint(encoder(features), predictor(targets))
            loss = transducer_loss(logits, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(round(total / batches, LOSS_DECIMALS))
        if settings.patience is not None and should_stop(losses, settings.patience):
            break
    return TrainHistory(losses=tuple(losses), seed=settings.seed)
