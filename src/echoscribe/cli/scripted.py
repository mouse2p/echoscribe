"""Scripted posteriors from planted transcripts, for the model-free stages.

Everything the decode, bias and stream commands do without a trained acoustic
model runs on matrices built here: an utterance's planted unit sequence is
spread across its frames, and
:func:`~echoscribe.decode.posteriors.scripted_posteriors` turns that path
into log-posteriors whose per-frame argmax is the path by construction.

These matrices are *fixtures*, and every command that uses them says so in
its output.  They exercise decoders, schedulers and scorers against a known
answer; a low error rate measured this way is a statement about the decoding
mechanics, never about acoustic modelling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echoscribe.decode.posteriors import scripted_posteriors
from echoscribe.errors import ConfigError, require
from echoscribe.synth.manifest import ManifestRecord, read_manifest
from echoscribe.text.units import unit_vocabulary, utterance_units
from echoscribe.text.vocab import Vocabulary


@dataclass(frozen=True)
class ScriptedUtterance:
    """One utterance's planted path and the posteriors that carry it."""

    utterance_id: str
    text: str
    units: tuple[str, ...]
    unit_ids: tuple[int, ...]
    num_frames: int
    posteriors: np.ndarray

    @property
    def num_symbols(self) -> int:
        """Alphabet width the posteriors were built for."""
        return int(self.posteriors.shape[1])


def scripted_path(unit_ids: Sequence[int], num_frames: int, blank_id: int = 0) -> list[int]:
    """Spread units across frames: runs of one unit, blanks between them.

    Each unit holds a run of consecutive frames and, when there is room, a
    single blank separates two runs -- which is what a real CTC path looks
    like, labels not touching.  When the frames are too tight to separate
    every unit (fewer than ``2n - 1`` of them for ``n`` units), separators
    are dropped rather than units lost, and repeated adjacent units then
    legitimately collapse on decoding.
    """
    count = len(unit_ids)
    require(count > 0, "a scripted path needs at least one unit", ConfigError)
    require(
        isinstance(num_frames, int) and not isinstance(num_frames, bool) and num_frames >= count,
        f"{num_frames} frames cannot carry {count} units",
        ConfigError,
    )
    separators = count - 1 if num_frames >= 2 * count - 1 else 0
    base, extra = divmod(num_frames - separators, count)
    path: list[int] = []
    for index, unit in enumerate(unit_ids):
        if index > 0 and separators:
            path.append(blank_id)
        path.extend([int(unit)] * (base + (1 if index < extra else 0)))
    return path


def load_scripted_corpus(
    workdir: Path | str, level: str = "syllable"
) -> tuple[tuple[ScriptedUtterance, ...], Vocabulary]:
    """Rebuild every planted decode input from a prepared work directory.

    The directory must hold a rendered manifest (``echoscribe synth``) and a
    feature archive (``echoscribe feats``); the frame count per utterance
    comes from the archive, so a decode always runs over the frame grid the
    features defined.
    """
    root = Path(workdir)
    manifest_path = root / "manifest.jsonl"
    require(
        manifest_path.exists(),
        f"no manifest at {manifest_path}; run `echoscribe synth` first",
        ConfigError,
    )
    features_path = root / "features.npz"
    require(
        features_path.exists(),
        f"no features at {features_path}; run `echoscribe feats` first",
        ConfigError,
    )
    records: Sequence[ManifestRecord] = read_manifest(manifest_path)
    require(len(records) > 0, f"the manifest at {manifest_path} holds no utterances", ConfigError)
    with np.load(features_path) as bundle:
        frame_counts = {
            key: int(np.asarray(bundle[key]).shape[0]) for key in bundle.files if key != "meta"
        }
    vocabulary = unit_vocabulary(level=level)
    utterances: list[ScriptedUtterance] = []
    for record in records:
        units = utterance_units(record.tokens, record.languages, level=level)
        ids = tuple(vocabulary.encode(units, allow_unk=False))
        require(
            record.utterance_id in frame_counts,
            f"{record.utterance_id} is in the manifest but not in {features_path}",
            ConfigError,
        )
        num_frames = frame_counts[record.utterance_id]
        path = scripted_path(ids, num_frames)
        utterances.append(
            ScriptedUtterance(
                utterance_id=record.utterance_id,
                text=record.text,
                units=units,
                unit_ids=ids,
                num_frames=num_frames,
                posteriors=scripted_posteriors(path, num_symbols=len(vocabulary)),
            )
        )
    return tuple(utterances), vocabulary
