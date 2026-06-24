#!/usr/bin/env python3
"""Generate a synthetic corpus, render it, and extract its features.

Everything is seeded: the same seed renders the same waveforms, the same
manifest digest and the same feature matrices, on any machine.  The
transcripts are planted ground truth from the synthetic lexicon -- no real
speech is involved anywhere in this walkthrough.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from echoscribe.audio import read_wav
from echoscribe.cmvn import CmvnStatistics, apply_cmvn
from echoscribe.delta import DeltaConfig, stack_features
from echoscribe.features import FeatureConfig, extract_log_mel
from echoscribe.framing import FrameConfig
from echoscribe.seeds import Seed
from echoscribe.synth.corpus import CorpusConfig, generate_corpus
from echoscribe.synth.manifest import manifest_digest, read_manifest
from echoscribe.synth.render import RenderConfig, render_corpus


def main() -> int:
    """Run the walkthrough; return a process exit code."""
    parser = argparse.ArgumentParser(description="synthetic corpus + feature extraction")
    parser.add_argument("--workdir", required=True, help="directory to render into")
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num-mel-bins", type=int, default=40)
    args = parser.parse_args()
    workdir = Path(args.workdir)

    corpus = generate_corpus(
        CorpusConfig(
            utterances=args.utterances,
            min_tokens=1,
            max_tokens=4,
            seed=Seed(args.seed, "example.corpus"),
        )
    )
    print(f"corpus: {len(corpus)} utterances, language mix {corpus.language_mix()}")

    records = render_corpus(corpus, RenderConfig(), workdir)
    total_seconds = sum(record.duration for record in records)
    print(f"rendered {total_seconds:.2f} s of audio into {workdir / 'wav'}")
    print(f"manifest digest: {manifest_digest(records)}")

    config = FeatureConfig(
        num_mel_bins=args.num_mel_bins,
        frame=FrameConfig(sample_rate=records[0].sample_rate),
    )
    print(f"features: {args.num_mel_bins} log-mel bins, 10 ms shift, deltas stacked")
    for record in records[:3]:
        audio = read_wav(workdir / record.wav_path)
        log_mel = extract_log_mel(audio, config)
        statistics = CmvnStatistics(dimension=args.num_mel_bins)
        statistics.update(log_mel)
        normalized = apply_cmvn(log_mel, statistics)
        stacked = stack_features(normalized, DeltaConfig(order=1))
        print(
            f"  {record.utterance_id}: {log_mel.shape[0]} frames, "
            f"{log_mel.shape[1]} bins -> {stacked.shape[1]} stacked"
        )
        print(f"    planted text: {record.text}")

    again = render_corpus(corpus, RenderConfig(), workdir / "again")
    assert manifest_digest(again) == manifest_digest(records)
    print("re-rendering the same corpus reproduced the same manifest digest")

    reread = read_manifest(workdir / "manifest.jsonl")
    assert len(reread) == len(records)
    print(f"the manifest reads back with all {len(reread)} records")
    print("note: every transcript above is planted ground truth, not a transcription")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
