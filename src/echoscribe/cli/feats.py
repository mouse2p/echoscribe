"""``echoscribe feats``: log-mel features for every wav in a rendered corpus.

The output is one ``.npz``: an array per utterance keyed by its id, plus a
``meta`` entry holding the configuration as JSON.  One file rather than one
per utterance because a decoding or scoring stage wants the whole corpus in
memory anyway, and a single archive is a single thing to checksum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from echoscribe.audio import read_wav
from echoscribe.cli.common import EXIT_OK, fail
from echoscribe.delta import DeltaConfig, stack_features
from echoscribe.features import FeatureConfig, extract_log_mel
from echoscribe.framing import FrameConfig
from echoscribe.synth.manifest import read_manifest


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``feats`` to the top-level parser."""
    parser = subparsers.add_parser(
        "feats",
        help="extract log-mel features from a rendered corpus",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument(
        "--workdir", required=True, help="directory holding manifest.jsonl and wav/"
    )
    parser.add_argument("--num-mel-bins", type=int, default=80, help="mel bins per frame")
    parser.add_argument(
        "--deltas", action="store_true", help="stack delta features, widening each frame"
    )
    parser.add_argument(
        "--delta-order", type=int, default=2, help="derivative orders stacked with --deltas"
    )
    parser.add_argument(
        "--out", default="features.npz", help="npz file to write inside the work directory"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Extract, archive and summarize one corpus of features."""
    workdir = Path(args.workdir)
    manifest_path = workdir / "manifest.jsonl"
    if not manifest_path.exists():
        return fail(f"no manifest at {manifest_path}; run `echoscribe synth` first")
    records = read_manifest(manifest_path)
    if not records:
        return fail(f"the manifest at {manifest_path} holds no utterances")
    rates = {record.sample_rate for record in records}
    if len(rates) != 1:
        return fail(f"the manifest mixes sample rates {sorted(rates)}")
    config = FeatureConfig(
        num_mel_bins=args.num_mel_bins, frame=FrameConfig(sample_rate=records[0].sample_rate)
    )
    arrays: dict[str, np.ndarray] = {}
    frames = 0
    for record in records:
        audio = read_wav(workdir / record.wav_path)
        features = extract_log_mel(audio, config)
        if args.deltas:
            features = stack_features(features, DeltaConfig(order=args.delta_order))
        arrays[record.utterance_id] = features
        frames += features.shape[0]
    width = next(iter(arrays.values())).shape[1]
    meta = {
        "num_mel_bins": args.num_mel_bins,
        "deltas": bool(args.deltas),
        "delta_order": args.delta_order if args.deltas else 0,
        "utterances": len(records),
        "frames": frames,
        "width": width,
    }
    target = workdir / args.out
    # Typed as dict[str, Any] because np.savez's stubs model **kwds narrowly;
    # the values are all arrays, and the keys are utterance ids plus "meta".
    archive: dict[str, Any] = dict(arrays)
    archive["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(target, **archive)
    print(
        f"echoscribe feats: {len(records)} utterances, {frames} frames of width {width} -> {target}"
    )
    return EXIT_OK
