#!/usr/bin/env python3
"""Decode one scripted posterior three ways: CTC greedy, CTC prefix beam, transducer.

Pure NumPy -- no torch, no audio, no model.  The posterior matrix is a
*fixture*: its per-frame argmax path is planted by construction, so all three
decoders must replay exactly that sequence.  Agreement is guaranteed by the
fixture; what the example validates is the decoders' arithmetic, and what it
shows is the shape of each decoder's output (collapsed symbols, an ordered
n-best list, transducer emissions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from echoscribe.decode.ctc_greedy import ctc_greedy
from echoscribe.decode.ctc_prefix import ctc_prefix_search
from echoscribe.decode.posteriors import scripted_posteriors
from echoscribe.decode.transducer import TableJoint, TransducerConfig, transducer_greedy

PLANTED = (3, 1, 4, 1, 5)
NUM_SYMBOLS = 8
FRAMES = 60


def ctc_path(units: tuple[int, ...], frames: int, blank: int = 0) -> list[int]:
    """Runs of one unit per symbol, single blanks between: a CTC-shaped path."""
    separators = len(units) - 1
    base, extra = divmod(frames - separators, len(units))
    path: list[int] = []
    for index, unit in enumerate(units):
        if index:
            path.append(blank)
        path.extend([unit] * (base + (1 if index < extra else 0)))
    return path


def sparse_path(units: tuple[int, ...], frames: int, blank: int = 0) -> list[int]:
    """One frame per unit, blanks elsewhere: a transducer-shaped path."""
    gap = frames // len(units)
    path = [blank] * frames
    for index, unit in enumerate(units):
        path[index * gap] = unit
    return path


def joint_table(units: tuple[int, ...], frames: int, num_symbols: int) -> np.ndarray:
    """A history-indexed joint whose argmax walks the sparse path once."""
    table = np.zeros((frames, len(units) + 1, num_symbols))
    table[:, :, 0] = 4.0  # blank dominates every state...
    gap = frames // len(units)
    for index, unit in enumerate(units):
        table[index * gap, index, unit] = 9.0  # ...except the planted emission
    return table


def main() -> int:
    """Run all three decoders over the same planted answer."""
    parser = argparse.ArgumentParser(description="CTC vs transducer decoding comparison")
    parser.add_argument("--workdir", required=True, help="directory for the comparison file")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    posteriors = scripted_posteriors(ctc_path(PLANTED, FRAMES), num_symbols=NUM_SYMBOLS)
    greedy = ctc_greedy(posteriors)
    prefix = ctc_prefix_search(posteriors, beam_size=4)
    print(f"planted sequence:      {PLANTED}")
    print(f"CTC greedy:            {greedy.symbols}   path score {greedy.score:.4f}")
    print(f"CTC prefix (beam 4):   {prefix.symbols}   score {prefix.best.score:.4f}")
    for rank, hypothesis in enumerate(prefix.n_best(3), start=1):
        print(f"  n-best {rank}: {hypothesis.tokens} score {hypothesis.score:.4f}")
    scores = [hypothesis.score for hypothesis in prefix.n_best(3)]
    assert scores == sorted(scores, reverse=True), "the n-best must stay score-ordered"

    joint = TableJoint(logits=joint_table(PLANTED, FRAMES, NUM_SYMBOLS))
    transducer = transducer_greedy(joint, TransducerConfig(max_symbols_per_frame=1))
    print(f"transducer greedy:     {transducer.symbols}   score {transducer.score:.4f}")

    assert greedy.symbols == PLANTED == prefix.symbols == transducer.symbols
    print("all three decoders replayed the planted sequence exactly")

    comparison = {
        "planted": list(PLANTED),
        "ctc_greedy": list(greedy.symbols),
        "ctc_prefix": list(prefix.symbols),
        "transducer": list(transducer.symbols),
        "n_best": [list(hypothesis.tokens) for hypothesis in prefix.n_best(3)],
    }
    (workdir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"comparison written to {workdir / 'comparison.json'}")
    print(
        "note: the posterior is a scripted fixture; agreement validates decoder "
        "arithmetic, not any acoustic model"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
