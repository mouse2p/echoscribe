"""Manifests are a serialized contract, so every field is pinned."""

from __future__ import annotations

import hashlib

import pytest

from echoscribe.errors import CorpusError
from echoscribe.synth.manifest import (
    REQUIRED_FIELDS,
    ManifestRecord,
    checksum_of_bytes,
    checksum_of_file,
)


def _record(**overrides) -> ManifestRecord:
    fields = {
        "utterance_id": "utt-00001",
        "text": "hello 你好",
        "tokens": ("hello", "你", "好"),
        "languages": ("en", "zh", "zh"),
        "wav_path": "wav/utt-00001.wav",
        "sample_rate": 16000,
        "num_samples": 16000,
        "duration": 1.0,
        "seed": 12345,
        "checksum": hashlib.sha256(b"audio").hexdigest(),
        "condition": {"noise_kind": None},
    }
    fields.update(overrides)
    return ManifestRecord(**fields)


def test_a_record_keeps_every_field() -> None:
    record = _record()
    assert record.utterance_id == "utt-00001"
    assert record.tokens == ("hello", "你", "好")
    assert record.languages == ("en", "zh", "zh")
    assert record.duration == 1.0
    assert len(record.checksum) == 64


def test_required_fields_cover_the_whole_record() -> None:
    assert "schema" in REQUIRED_FIELDS
    for name in ManifestRecord.__dataclass_fields__:
        assert name in REQUIRED_FIELDS


def test_a_record_needs_an_id() -> None:
    with pytest.raises(CorpusError, match="needs an utterance id"):
        _record(utterance_id="")


def test_a_record_rejects_a_mismatched_label_count() -> None:
    with pytest.raises(CorpusError, match="tokens but 2 labels"):
        _record(tokens=("a", "b", "c"), languages=("en", "zh"))


def test_a_record_rejects_a_short_checksum() -> None:
    with pytest.raises(CorpusError, match="not sha256 hex"):
        _record(checksum="abc")


def test_a_record_rejects_a_bad_sample_rate_or_length() -> None:
    with pytest.raises(CorpusError, match="sample_rate must be positive"):
        _record(sample_rate=0)
    with pytest.raises(CorpusError, match="num_samples must not be negative"):
        _record(num_samples=-1)


def test_the_byte_checksum_matches_hashlib() -> None:
    assert checksum_of_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_the_file_checksum_matches_the_byte_checksum(tmp_path) -> None:
    target = tmp_path / "audio.wav"
    payload = bytes(range(256)) * 64
    target.write_bytes(payload)
    assert checksum_of_file(target) == checksum_of_bytes(payload)


def test_a_missing_file_cannot_be_checksummed(tmp_path) -> None:
    with pytest.raises(CorpusError, match="cannot checksum a missing file"):
        checksum_of_file(tmp_path / "absent.wav")


def test_a_record_is_immutable() -> None:
    record = _record()
    with pytest.raises(AttributeError):
        record.duration = 2.0  # type: ignore[misc]
