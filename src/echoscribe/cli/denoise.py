"""``echoscribe denoise``: spectral denoising over a rendered corpus.

Two honest modes.  Without a clean reference the command reports only how
much energy it kept -- a denoiser that sees just the noisy signal cannot
claim an SNR.  With ``--clean-workdir`` pointing at a clean render of the
*same corpus* (one seed, two renders), it measures before/after SNR against
the planted clean audio, which is the only kind of SNR claim this toolkit
makes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from echoscribe.audio import read_wav, write_wav
from echoscribe.cli.common import EXIT_OK, fail
from echoscribe.errors import ConfigError, require
from echoscribe.noise.denoise import DenoiseConfig, denoise
from echoscribe.noise.gains import GAIN_METHODS
from echoscribe.noise.snr import snr_improvement_db
from echoscribe.synth.manifest import read_manifest
from echoscribe.waveform import Audio


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``denoise`` to the top-level parser."""
    parser = subparsers.add_parser(
        "denoise",
        help="spectrally denoise a rendered corpus, with optional planted SNR measurement",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument(
        "--workdir", required=True, help="directory holding manifest.jsonl and wav/"
    )
    parser.add_argument(
        "--method", choices=GAIN_METHODS, default="subtraction", help="gain to apply"
    )
    parser.add_argument("--oversubtraction", type=float, default=1.5, help="subtraction strength")
    parser.add_argument("--floor", type=float, default=0.05, help="gain floor in [0, 1)")
    parser.add_argument(
        "--clean-workdir",
        default=None,
        help="a clean render of the same corpus; enables measured SNR improvement",
    )
    parser.add_argument("--out-dir", default="denoised", help="directory for the enhanced wavs")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Denoise every utterance and report what can honestly be reported."""
    root = Path(args.workdir)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        return fail(f"no manifest at {manifest_path}; run `echoscribe synth` first")
    records = read_manifest(manifest_path)
    clean_root = Path(args.clean_workdir) if args.clean_workdir else None
    if clean_root is not None and not (clean_root / "manifest.jsonl").exists():
        return fail(f"no manifest at {clean_root}; --clean-workdir must hold a clean render")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = DenoiseConfig(
        method=args.method, oversubtraction=args.oversubtraction, floor=args.floor
    )
    kept_ratios: list[float] = []
    improvements: list[float] = []
    for record in records:
        noisy = read_wav(root / record.wav_path)
        result = denoise(noisy.samples.astype(np.float64), config)
        enhanced = Audio(result.samples.astype(np.float32), noisy.sample_rate)
        write_wav(out_dir / f"{record.utterance_id}.wav", enhanced)
        energy_in = float(np.sum(noisy.samples.astype(np.float64) ** 2))
        kept = float(np.sum(result.samples**2)) / energy_in if energy_in > 0 else 1.0
        kept_ratios.append(kept)
        if clean_root is not None:
            clean = read_wav(clean_root / record.wav_path)
            require(
                clean.num_frames == noisy.num_frames,
                f"{record.utterance_id}: the clean render has {clean.num_frames} samples "
                f"but the noisy one has {noisy.num_frames}; are the seeds the same?",
                ConfigError,
            )
            _, _, improvement = snr_improvement_db(clean.samples, noisy.samples, enhanced.samples)
            improvements.append(improvement)
    print(f"echoscribe denoise ({args.method}): {len(records)} utterances -> {out_dir}")
    print(f"  mean energy kept: {sum(kept_ratios) / len(kept_ratios):.3f}")
    if improvements:
        mean_gain = sum(improvements) / len(improvements)
        print(
            f"  mean measured SNR improvement: {mean_gain:+.2f} dB over "
            f"{len(improvements)} planted clean/noisy pairs"
        )
    else:
        print("  no clean reference given: the energy ratio is not an SNR claim")
    return EXIT_OK
