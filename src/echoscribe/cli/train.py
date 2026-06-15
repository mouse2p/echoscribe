"""``echoscribe train``: train a tiny model on the synthetic laboratory.

The torch imports live inside the command rather than at the module top, so
a CLI without the torch extra still prints this help and fails at run time
with an installation instruction rather than failing at import time with a
traceback.  The numbers this command prints are losses on planted synthetic
data: a real decrease, evidence the plumbing works, and nothing about
recognizer accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echoscribe.cli.common import EXIT_OK, write_json
from echoscribe.model import require_torch

METHODS = ("ctc", "transducer")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``train`` to the top-level parser."""
    parser = subparsers.add_parser(
        "train",
        help="train a tiny CTC or transducer model on synthetic speech (torch extra)",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory to write the bundle into")
    parser.add_argument("--method", choices=METHODS, default="ctc", help="which head to train")
    parser.add_argument("--epochs", type=int, default=8, help="passes over the corpus")
    parser.add_argument("--utterances", type=int, default=16, help="corpus size to render")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam step size")
    parser.add_argument("--batch-size", type=int, default=4, help="utterances per gradient step")
    parser.add_argument("--seed", type=int, default=2025, help="corpus seed")
    parser.add_argument("--model-seed", type=int, default=3, help="weight-initialisation seed")
    parser.add_argument("--hidden", type=int, default=48, help="encoder and joint width")
    parser.add_argument("--out", default="model.pt", help="bundle file to write")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Render a laboratory, train on it, and write bundle plus history."""
    require_torch()
    import torch

    from echoscribe.model.ctc import CtcConfig, CtcHead
    from echoscribe.model.encoder import Encoder, EncoderConfig
    from echoscribe.model.train import (
        TrainConfig,
        laboratory_examples,
        save_training_bundle,
        train_ctc,
        train_transducer,
    )
    from echoscribe.model.transducer import Joint, JointConfig, Predictor, PredictorConfig

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    examples, vocabulary = laboratory_examples(seed=args.seed, utterances=args.utterances)
    vocab_size = len(vocabulary) - 1
    torch.manual_seed(args.model_seed)
    encoder = Encoder(
        EncoderConfig(input_dim=40, stack=2, hidden_dim=args.hidden, output_dim=args.hidden)
    )
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.model_seed,
    )
    if args.method == "ctc":
        head = CtcHead(CtcConfig(input_dim=args.hidden, vocab_size=vocab_size))
        history = train_ctc(encoder, head, examples, config)
        modules = {"encoder": encoder, "head": head}
        configs = {
            "encoder": encoder.config.to_dict(),
            "head": head.config.to_dict(),
            "train": config.to_dict(),
        }
    else:
        predictor = Predictor(
            PredictorConfig(vocab_size=vocab_size, embed_dim=16, hidden_dim=args.hidden)
        )
        joint = Joint(
            JointConfig(
                encoder_dim=args.hidden,
                predictor_dim=args.hidden,
                vocab_size=vocab_size,
                hidden_dim=args.hidden,
            )
        )
        history = train_transducer(encoder, predictor, joint, examples, config)
        modules = {"encoder": encoder, "predictor": predictor, "joint": joint}
        configs = {
            "encoder": encoder.config.to_dict(),
            "predictor": predictor.config.to_dict(),
            "joint": joint.config.to_dict(),
            "train": config.to_dict(),
        }
    target = save_training_bundle(workdir / args.out, args.method, configs, modules, history)
    write_json(workdir / "train-history.json", history.to_dict())
    print(
        f"echoscribe train ({args.method}): loss {history.first_loss:.4f} -> "
        f"{history.last_loss:.4f} over {history.epochs_completed} epochs"
    )
    print(f"  bundle: {target}")
    print(
        "  note: synthetic speech with planted transcripts; the decrease shows the "
        "loop works, not that a recognizer does"
    )
    return EXIT_OK
