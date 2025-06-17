"""JSONL corpus manifests with a schema version and a content checksum.

A manifest is the contract between the laboratory and everything downstream of it:
the decoder reads the audio path, the scorer reads the transcript, and a rerun
needs to know whether the audio on disk is still the audio that was scored.  Three
things make that checkable:

``schema``
    a version on every line, so a reader can refuse a manifest it does not
    understand instead of silently mis-mapping a field;
``checksum``
    SHA-256 over each wav file's bytes, recorded at render time;
``condition``
    the acoustic settings the utterance was rendered under, so a result can always
    be attributed to a noise level or a room rather than to "the corpus".

Lines are JSON objects, one per utterance, which keeps a manifest streamable: a
corpus can be appended to without rewriting what is already there.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from echoscribe.errors import CorpusError, SchemaError, require

#: Bumped whenever the shape of a manifest line changes.
MANIFEST_SCHEMA_VERSION = 1

#: Fields every manifest line must carry.
REQUIRED_FIELDS = (
    "schema",
    "utterance_id",
    "text",
    "tokens",
    "languages",
    "wav_path",
    "sample_rate",
    "num_samples",
    "duration",
    "seed",
    "checksum",
    "condition",
)


@dataclass(frozen=True)
class ManifestRecord:
    """One utterance's entry in a corpus manifest."""

    utterance_id: str
    text: str
    tokens: tuple[str, ...]
    languages: tuple[str, ...]
    wav_path: str
    sample_rate: int
    num_samples: int
    duration: float
    seed: int
    checksum: str
    condition: dict

    def __post_init__(self) -> None:
        require(bool(self.utterance_id), "a manifest record needs an utterance id", CorpusError)
        require(
            self.sample_rate > 0,
            f"sample_rate must be positive, got {self.sample_rate}",
            CorpusError,
        )
        require(
            self.num_samples >= 0,
            f"num_samples must not be negative, got {self.num_samples}",
            CorpusError,
        )
        require(
            len(self.tokens) == len(self.languages),
            f"{self.utterance_id}: {len(self.tokens)} tokens but {len(self.languages)} labels",
            CorpusError,
        )
        require(
            len(self.checksum) == 64,
            f"{self.utterance_id}: checksum is not sha256 hex",
            CorpusError,
        )


def checksum_of_bytes(payload: bytes) -> str:
    """SHA-256 hex digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def checksum_of_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's contents, read in blocks."""
    resolved = Path(path)
    require(resolved.is_file(), f"cannot checksum a missing file: {resolved}", CorpusError)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_to_dict(record: ManifestRecord) -> dict:
    """Render a record as a JSON-ready mapping, tagged with the schema version."""
    payload = asdict(record)
    payload["tokens"] = list(record.tokens)
    payload["languages"] = list(record.languages)
    payload["duration"] = round(float(record.duration), 9)
    return {"schema": MANIFEST_SCHEMA_VERSION, **payload}


def record_from_dict(payload: dict) -> ManifestRecord:
    """Rebuild a record, refusing a schema this build does not read."""
    version = payload.get("schema")
    require(
        version == MANIFEST_SCHEMA_VERSION,
        f"manifest schema {version!r} is not supported; this build reads {MANIFEST_SCHEMA_VERSION}",
        SchemaError,
    )
    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    require(not missing, f"manifest line is missing {missing}", SchemaError)
    return ManifestRecord(
        utterance_id=str(payload["utterance_id"]),
        text=str(payload["text"]),
        tokens=tuple(payload["tokens"]),
        languages=tuple(payload["languages"]),
        wav_path=str(payload["wav_path"]),
        sample_rate=int(payload["sample_rate"]),
        num_samples=int(payload["num_samples"]),
        duration=float(payload["duration"]),
        seed=int(payload["seed"]),
        checksum=str(payload["checksum"]),
        condition=dict(payload["condition"]),
    )
