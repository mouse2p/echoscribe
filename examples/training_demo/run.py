#!/usr/bin/env python3
"""Train a tiny CTC model on synthetic speech and show the loss actually falling.

Needs the optional torch extra (CPU wheels are enough); without it the demo
prints an installation note and exits successfully rather than failing.

The decrease below is real -- gradients computed by torch on rendered formant
audio with planted transcripts -- and it is evidence about the training
plumbing, not about recognition.  The vocabulary is the synthetic lexicon's,
the corpus is seconds long, and nothing is downloaded anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echoscribe.model import TORCH_AVAILABLE, require_torch


def main() -> int:
    """Render a laboratory, train on it, print the loss curve."""
    parser = argparse.ArgumentParser(description="tiny CTC training demo")
    parser.add_argument("--workdir", required=True, help="directory for the history file")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--utterances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("the torch extra is not installed; this demo needs it:")
        print('  pip install "echoscribe[torch]"  (CPU wheels; see the README)')
        print("nothing was trained and no number below should be believed")
        return 0

    require_torch()
    import json

    import torch

    from echoscribe.model.ctc import CtcConfig, CtcHead
    from echoscribe.model.encoder import Encoder, EncoderConfig
    from echoscribe.model.train import TrainConfig, laboratory_examples, train_ctc

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    examples, vocabulary = laboratory_examples(seed=args.seed, utterances=args.utterances)
    print(
        f"laboratory: {len(examples)} utterances, vocabulary of {len(vocabulary)} tokens "
        f"(planted transcripts)"
    )

    torch.manual_seed(3)
    encoder = Encoder(EncoderConfig(input_dim=40, stack=2, hidden_dim=32, output_dim=32))
    head = CtcHead(CtcConfig(input_dim=32, vocab_size=len(vocabulary) - 1))
    print(
        f"model: {encoder.num_parameters() + head.num_parameters()} parameters, "
        f"CPU only, no pretrained weights"
    )

    history = train_ctc(
        encoder,
        head,
        examples,
        TrainConfig(epochs=args.epochs, learning_rate=0.01, batch_size=4, seed=3),
    )
    for epoch, loss in enumerate(history.losses, start=1):
        print(f"  epoch {epoch}: mean CTC loss {loss:.4f}")
    (workdir / "train-history.json").write_text(
        json.dumps(history.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not history.improved:
        print("the loss did not fall; the demo refuses to dress that up as success")
        return 1
    print(
        f"loss fell {history.first_loss:.4f} -> {history.last_loss:.4f} "
        f"({history.total_improvement:.4f} total)"
    )
    print(
        "note: this decrease shows forward, loss and backward agree on planted "
        "synthetic data; it is not recognizer accuracy"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
