"""Manifests are a serialized contract, so every field is pinned."""

from __future__ import annotations

import hashlib
import json

import pytest

from echoscribe.errors import CorpusError, SchemaError
from echoscribe.synth.manifest import (
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_FIELDS,
    ManifestRecord,
    checksum_of_bytes,
    checksum_of_file,
    manifest_digest,
    read_manifest,
    record_from_dict,
    record_to_dict,
    verify_manifest,
    write_manifest,
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


def test_serialization_round_trips() -> None:
    record = _record()
    assert record_from_dict(record_to_dict(record)) == record


def test_a_serialized_record_names_its_schema() -> None:
    payload = record_to_dict(_record())
    assert payload["schema"] == 1
    assert set(payload) == set(REQUIRED_FIELDS)


def test_a_serialized_record_is_json_encodable() -> None:
    import json

    text = json.dumps(record_to_dict(_record()), ensure_ascii=False)
    assert "你好" in text


def test_a_foreign_schema_version_is_refused() -> None:
    payload = record_to_dict(_record())
    payload["schema"] = 99
    with pytest.raises(SchemaError, match="is not supported"):
        record_from_dict(payload)


def test_a_missing_field_is_reported_by_name() -> None:
    payload = record_to_dict(_record())
    del payload["checksum"]
    with pytest.raises(SchemaError, match=r"missing \['checksum'\]"):
        record_from_dict(payload)


def test_a_list_field_comes_back_as_a_tuple() -> None:
    restored = record_from_dict(record_to_dict(_record()))
    assert isinstance(restored.tokens, tuple)
    assert isinstance(restored.languages, tuple)


def test_the_duration_is_rounded_where_it_is_produced() -> None:
    record = _record(duration=1.0 / 3.0)
    assert record_to_dict(record)["duration"] == 0.333333333


def test_the_schema_version_is_the_one_the_lines_carry() -> None:
    assert MANIFEST_SCHEMA_VERSION == record_to_dict(_record())["schema"]


def test_serialization_does_not_mutate_the_record() -> None:
    record = _record()
    before = record_to_dict(record)
    record_to_dict(record)
    assert record_to_dict(record) == before


def test_a_manifest_round_trips(tmp_path) -> None:
    target = tmp_path / "corpus" / "manifest.jsonl"
    records = [
        _record(utterance_id="utt-00000", checksum=hashlib.sha256(b"a").hexdigest()),
        _record(utterance_id="utt-00001", checksum=hashlib.sha256(b"b").hexdigest()),
    ]
    digest = write_manifest(target, records)
    assert read_manifest(target) == records
    assert digest == manifest_digest(records)


def test_a_manifest_has_one_line_per_record(tmp_path) -> None:
    target = tmp_path / "manifest.jsonl"
    write_manifest(target, [_record(utterance_id=f"utt-{index:05d}") for index in range(5)])
    assert len(target.read_text(encoding="utf-8").splitlines()) == 5


def test_a_manifest_keeps_non_ascii_transcripts(tmp_path) -> None:
    target = tmp_path / "manifest.jsonl"
    write_manifest(target, [_record(text="你好 world")])
    assert "你好 world" in target.read_text(encoding="utf-8")
    assert read_manifest(target)[0].text == "你好 world"


def test_the_digest_changes_when_a_checksum_changes() -> None:
    left = [_record(checksum=hashlib.sha256(b"a").hexdigest())]
    right = [_record(checksum=hashlib.sha256(b"b").hexdigest())]
    assert manifest_digest(left) != manifest_digest(right)


def test_the_digest_depends_on_the_order() -> None:
    first = _record(utterance_id="utt-00000", checksum=hashlib.sha256(b"a").hexdigest())
    second = _record(utterance_id="utt-00001", checksum=hashlib.sha256(b"b").hexdigest())
    assert manifest_digest([first, second]) != manifest_digest([second, first])


def test_the_digest_ignores_the_condition_metadata() -> None:
    left = [_record(condition={"noise_kind": None})]
    right = [_record(condition={"noise_kind": "white", "noise_snr_db": 10.0})]
    assert manifest_digest(left) == manifest_digest(right)


def test_writing_an_empty_manifest_is_refused(tmp_path) -> None:
    with pytest.raises(CorpusError, match="refusing to write an empty manifest"):
        write_manifest(tmp_path / "m.jsonl", [])


def test_reading_a_missing_manifest_is_refused(tmp_path) -> None:
    with pytest.raises(CorpusError, match="manifest not found"):
        read_manifest(tmp_path / "absent.jsonl")


def test_a_broken_line_is_reported_with_its_number(tmp_path) -> None:
    target = tmp_path / "manifest.jsonl"
    write_manifest(target, [_record()])
    target.write_text(target.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8")
    with pytest.raises(CorpusError, match=":2: not valid JSON"):
        read_manifest(target)


def test_a_foreign_schema_line_is_reported_with_its_number(tmp_path) -> None:
    target = tmp_path / "manifest.jsonl"
    payload = record_to_dict(_record())
    payload["schema"] = 7
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match=":1: manifest schema 7"):
        read_manifest(target)


def test_a_manifest_of_only_blank_lines_is_refused(tmp_path) -> None:
    target = tmp_path / "manifest.jsonl"
    target.write_text("\n\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="holds no records"):
        read_manifest(target)


def test_verification_passes_for_untouched_audio(tmp_path) -> None:
    audio = tmp_path / "wav" / "utt-00001.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF....")
    record = _record(wav_path="wav/utt-00001.wav", checksum=checksum_of_file(audio))
    assert verify_manifest(tmp_path, [record]) == []


def test_verification_reports_a_changed_file(tmp_path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"original")
    record = _record(wav_path="a.wav", checksum=checksum_of_file(audio))
    audio.write_bytes(b"re-rendered")
    assert verify_manifest(tmp_path, [record]) == ["utt-00001"]


def test_verification_reports_a_missing_file(tmp_path) -> None:
    record = _record(wav_path="gone.wav", checksum=hashlib.sha256(b"x").hexdigest())
    assert verify_manifest(tmp_path, [record]) == ["utt-00001"]


def test_verification_checks_every_record(tmp_path) -> None:
    good = tmp_path / "good.wav"
    good.write_bytes(b"good")
    records = [
        _record(utterance_id="ok", wav_path="good.wav", checksum=checksum_of_file(good)),
        _record(utterance_id="bad", wav_path="bad.wav", checksum="0" * 64),
    ]
    assert verify_manifest(tmp_path, records) == ["bad"]
