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
import json
from collections.abc import Sequence
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


def write_manifest(path: str | Path, records: Sequence[ManifestRecord]) -> str:
    """Write one JSON line per record and return the corpus digest.

    The digest covers the ordered ``(utterance_id, checksum)`` pairs rather than the
    file bytes, so reformatting a manifest -- different key order, different float
    rendering -- does not change it, while replacing or reordering an utterance
    does.
    """
    require(len(records) > 0, "refusing to write an empty manifest", CorpusError)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return manifest_digest(records)


def manifest_digest(records: Sequence[ManifestRecord]) -> str:
    """A single hex digest identifying an ordered collection of records."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.utterance_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(record.checksum.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read a manifest back, reporting the offending line number on any failure."""
    resolved = Path(path)
    require(resolved.is_file(), f"manifest not found: {resolved}", CorpusError)
    records = []
    with resolved.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise CorpusError(f"{resolved}:{number}: not valid JSON ({error})") from error
            require(
                isinstance(payload, dict),
                f"{resolved}:{number}: a manifest line must be a JSON object",
                CorpusError,
            )
            try:
                records.append(record_from_dict(payload))
            except SchemaError as error:
                raise CorpusError(f"{resolved}:{number}: {error}") from error
    require(bool(records), f"{resolved}: the manifest holds no records", CorpusError)
    return records


def verify_manifest(root: str | Path, records: Sequence[ManifestRecord]) -> list[str]:
    """Re-hash every referenced wav and return the ids that no longer match.

    A missing file counts as a mismatch, and is reported under the same key as a
    changed one: from a manifest's point of view both mean "this line does not
    describe the audio on disk".
    """
    base = Path(root)
    broken = []
    for record in records:
        path = base / record.wav_path
        if not path.is_file() or checksum_of_file(path) != record.checksum:
            broken.append(record.utterance_id)
    return broken
