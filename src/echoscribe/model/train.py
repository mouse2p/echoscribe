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
from echoscribe.errors import ConfigError, require
from echoscribe.features import FeatureConfig, extract_log_mel
from echoscribe.framing import FrameConfig
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
