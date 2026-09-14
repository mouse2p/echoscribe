"""``echoscribe decode``: CTC decoding of a prepared corpus.

Two methods share one door.  ``greedy`` takes the per-frame argmax, collapses
repeats and drops blanks -- the baseline a streaming system actually runs.
``prefix`` keeps the beam_size most probable collapsed sequences instead,
which costs more and buys an n-best list; on the unambiguous fixtures the two
must agree, and a test says so.

What the agreement is worth: on a fixture whose argmax path is planted,
greedy and prefix *have* to coincide, so a disagreement is a bug in one of
the two searches rather than a modelling question.  On a trained model's
posteriors the same check says nothing -- there the methods differ by design
-- which is why the invariant lives in the tests over fixtures and not in
the command's output.

Each row also carries surface text.  A row whose hypothesis matches the
planted units carries the reference text forward: the same unit sequence
renders to the same surface text under the lexicon, so this is a
reconstruction, not a peek at the answer.  A row that does *not* match has no
reliable surface form -- modelling units do not invert to words -- so its
text is the unit sequence joined with spaces, and a text-level score reads
that as the error it is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from echoscribe.cli.common import EXIT_OK, fail, read_json, write_jsonl
from echoscribe.cli.scripted import load_scripted_corpus
from echoscribe.cmvn import CmvnStatistics, apply_cmvn
from echoscribe.decode.confidence import token_confidences, utterance_confidence
from echoscribe.decode.ctc_greedy import ctc_greedy
from echoscribe.decode.ctc_prefix import ctc_prefix_search
from echoscribe.model import require_torch
from echoscribe.synth.manifest import read_manifest
from echoscribe.text.units import utterance_units
from echoscribe.text.vocab import Vocabulary

METHODS = ("greedy", "prefix")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``decode`` to the top-level parser."""
    parser = subparsers.add_parser(
        "decode",
        help="CTC-decode a prepared corpus of scripted posteriors",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--workdir", required=True, help="directory prepared by synth and feats")
    parser.add_argument("--method", choices=METHODS, default="greedy", help="decoding method")
    parser.add_argument("--beam", type=int, default=5, help="beam width for the prefix method")
    parser.add_argument("--out", default="decode.jsonl", help="hypothesis file to write")
    parser.add_argument(
        "--bundle",
        default=None,
        help="trained CTC bundle to decode through instead of the scripted fixtures",
    )
    parser.add_argument(
        "--confidences",
        action="store_true",
        help="attach per-token posterior confidences (scripted greedy path)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Decode every utterance and write one JSONL line per hypothesis."""
    if args.bundle:
        return run_with_bundle(args)
    if args.confidences and args.method != "greedy":
        return fail(
            "confidences need the greedy method; the prefix beam keeps no single token path"
        )
    utterances, vocabulary = load_scripted_corpus(args.workdir)
    rows: list[dict] = []
    exact = 0
    for item in utterances:
        greedy = None
        beam = None
        if args.method == "greedy":
            greedy = ctc_greedy(item.posteriors)
            symbols = greedy.symbols
        else:
            beam = ctc_prefix_search(item.posteriors, beam_size=args.beam)
            symbols = beam.symbols
        matched = symbols == item.unit_ids
        exact += int(matched)
        row = {
            "utterance_id": item.utterance_id,
            "method": args.method,
            "reference_units": list(item.units),
            "hypothesis_units": [vocabulary.token_at(symbol) for symbol in symbols],
            "hypothesis_ids": list(symbols),
            "exact_match": matched,
            "reference_text": item.text,
            "hypothesis_text": item.text
            if matched
            else " ".join(vocabulary.token_at(symbol) for symbol in symbols),
        }
        if greedy is not None and args.confidences:
            confidences = token_confidences(item.posteriors, greedy.tokens)
            row["token_confidences"] = [
                round(float(confidence.probability), 6) for confidence in confidences
            ]
            row["utterance_confidence"] = round(float(utterance_confidence(confidences)), 6)
        if beam is not None:
            row["n_best"] = [
                [list(hypothesis.tokens), round(float(hypothesis.score), 6)]
                for hypothesis in beam.n_best(3)
            ]
        rows.append(row)
    target = write_jsonl(Path(args.workdir) / args.out, rows)
    print(
        f"echoscribe decode ({args.method}): {len(rows)} utterances, "
        f"{exact} replayed their planted transcript exactly"
    )
    print(f"  hypotheses: {target}")
    print("  note: scripted fixture posteriors, not the output of a trained model")
    return EXIT_OK


def run_with_bundle(args: argparse.Namespace) -> int:
    """Decode through a trained bundle: real model posteriors, no fixtures.

    The features are normalized per utterance with their own CMVN statistics,
    exactly as the training bridge normalized them, so the model sees the
    distribution it was trained on.
    """
    require_torch()
    from echoscribe.model.infer import ctc_log_posteriors, load_ctc_model

    root = Path(args.workdir)
    vocab_path = root / "vocab.json"
    if not vocab_path.exists():
        return fail(f"no vocabulary at {vocab_path}; `echoscribe train` writes it")
    features_path = root / "features.npz"
    if not features_path.exists():
        return fail(f"no features at {features_path}; run `echoscribe feats` first")
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        return fail(f"no manifest at {manifest_path}; run `echoscribe synth` first")
    vocab_payload = read_json(vocab_path)
    if not isinstance(vocab_payload, dict):
        return fail(f"{vocab_path} does not look like an echoscribe vocabulary file")
    vocabulary = Vocabulary.from_dict(vocab_payload)
    encoder, head, _ = load_ctc_model(args.bundle)
    records = read_manifest(manifest_path)
    rows: list[dict] = []
    exact = 0
    with np.load(features_path) as archive:
        meta = json.loads(str(archive["meta"]))
        if meta["num_mel_bins"] != encoder.config.input_dim:
            return fail(
                f"the features hold {meta['num_mel_bins']} bins but the bundle "
                f"encoder reads {encoder.config.input_dim}; rerun feats with "
                f"--num-mel-bins {encoder.config.input_dim}"
            )
        for record in records:
            features = np.asarray(archive[record.utterance_id])
            statistics = CmvnStatistics(dimension=features.shape[1])
            statistics.update(features)
            normalized = apply_cmvn(features, statistics)
            posteriors = ctc_log_posteriors(encoder, head, normalized)
            symbols = list(ctc_greedy(posteriors).symbols)
            reference = list(utterance_units(record.tokens, record.languages))
            matched = symbols == list(vocabulary.encode(reference, allow_unk=False))
            exact += int(matched)
            units = [vocabulary.token_at(symbol) for symbol in symbols]
            rows.append(
                {
                    "utterance_id": record.utterance_id,
                    "method": "bundle-ctc",
                    "reference_units": reference,
                    "hypothesis_units": units,
                    "hypothesis_ids": symbols,
                    "exact_match": matched,
                    "reference_text": record.text,
                    "hypothesis_text": record.text if matched else " ".join(units),
                }
            )
    target = write_jsonl(root / args.out, rows)
    print(
        f"echoscribe decode (bundle-ctc): {len(rows)} utterances, "
        f"{exact} matched their planted transcript exactly"
    )
    print(f"  hypotheses: {target}")
    print("  note: the bundle was trained on this synthetic laboratory, so a low error")
    print("  rate here is memorization of planted data, not recognition accuracy")
    return EXIT_OK
