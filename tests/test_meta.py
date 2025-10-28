"""A fingerprint is only useful if it is stable, so stability is what is tested."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.meta import (
    FLOAT_DIGITS,
    METADATA_SCHEMA_VERSION,
    RunMetadata,
    fingerprint,
    package_versions,
    run_metadata,
    stable_repr,
)


@dataclass(frozen=True)
class _Inner:
    alpha: int = 1
    beta: str = "x"


@dataclass(frozen=True)
class _Outer:
    name: str = "run"
    inner: _Inner = field(default_factory=_Inner)
    values: tuple[float, ...] = (0.5, 1.5)


def test_package_versions_reports_the_required_entries() -> None:
    versions = package_versions()
    assert {"echoscribe", "numpy", "python", "torch"} <= set(versions)
    assert all(isinstance(value, str) and value for value in versions.values())


def test_an_absent_optional_dependency_is_recorded_not_omitted() -> None:
    versions = package_versions()
    assert versions["torch"] == "absent" or versions["torch"]


def test_stable_repr_expands_a_nested_dataclass() -> None:
    rendered = stable_repr(_Outer())
    assert "_Inner(alpha=1, beta='x')" in rendered
    assert rendered.startswith("_Outer(")


def test_stable_repr_sorts_mapping_keys() -> None:
    assert stable_repr({"b": 1, "a": 2}) == stable_repr({"a": 2, "b": 1})


def test_stable_repr_rounds_floats() -> None:
    assert stable_repr(1.0 / 3.0) == repr(round(1.0 / 3.0, FLOAT_DIGITS))


def test_stable_repr_summarises_an_array() -> None:
    rendered = stable_repr(np.arange(6.0))
    assert "shape=(6,)" in rendered
    assert "sha256=" in rendered


def test_two_arrays_with_the_same_bytes_share_a_rendering() -> None:
    assert stable_repr(np.ones(4)) == stable_repr(np.ones(4))
    assert stable_repr(np.ones(4)) != stable_repr(np.zeros(4))


def test_stable_repr_refuses_an_opaque_object() -> None:
    with pytest.raises(ConfigError, match="cannot fingerprint a value of type"):
        stable_repr(object())


def test_a_fingerprint_is_stable_across_key_order() -> None:
    left = fingerprint({"a": 1, "b": _Outer()})
    right = fingerprint({"b": _Outer(), "a": 1})
    assert left == right
    assert len(left) == 16


def test_a_fingerprint_changes_with_the_configuration() -> None:
    assert fingerprint({"x": 1}) != fingerprint({"x": 2})
    assert fingerprint({"x": 1}) != fingerprint({"y": 1})


def test_a_fingerprint_of_nothing_is_still_a_hash() -> None:
    assert len(fingerprint({})) == 16


def test_a_non_string_key_is_refused() -> None:
    with pytest.raises(ConfigError, match="keys must be strings"):
        fingerprint({1: "a"})


def test_private_dataclass_fields_are_left_out() -> None:
    @dataclass(frozen=True)
    class _Cached:
        value: int = 1
        _cache: dict = field(default_factory=dict, repr=False, compare=False)

    assert "_cache" not in stable_repr(_Cached())


def test_the_schema_version_is_declared() -> None:
    assert METADATA_SCHEMA_VERSION == 1


def test_run_metadata_keeps_its_fields() -> None:
    meta = run_metadata("synthetic-corpus", 20250106, frames=_Outer())
    assert meta.label == "synthetic-corpus"
    assert meta.seed == 20250106
    assert meta.versions["echoscribe"]


def test_the_digest_is_stable_for_identical_metadata() -> None:
    left = run_metadata("run", 7, a=1, b=_Outer())
    right = run_metadata("run", 7, b=_Outer(), a=1)
    assert left.digest() == right.digest()
    assert len(left.digest()) == 16


def test_the_digest_changes_with_the_seed_or_label() -> None:
    base = run_metadata("run", 7, a=1)
    assert base.digest() != run_metadata("run", 8, a=1).digest()
    assert base.digest() != run_metadata("other", 7, a=1).digest()
    assert base.digest() != run_metadata("run", 7, a=2).digest()


def test_with_config_returns_a_new_object() -> None:
    base = run_metadata("run", 1)
    extended = base.with_config("frame", _Outer())
    assert base.configs == {}
    assert "frame" in extended.configs
    assert extended.digest() != base.digest()


def test_serialization_round_trips_the_digest() -> None:
    meta = run_metadata("run", 3, a=_Outer())
    payload = meta.to_dict()
    assert payload["schema"] == METADATA_SCHEMA_VERSION
    assert set(payload) == {"schema", "label", "seed", "versions", "configs", "digest"}
    assert RunMetadata.from_dict(payload).digest() == meta.digest()


def test_configs_are_serialized_as_canonical_strings() -> None:
    payload = run_metadata("run", 3, frame=_Outer()).to_dict()
    assert payload["configs"]["frame"] == stable_repr(_Outer())


def test_the_payload_is_json_encodable() -> None:
    import json

    text = json.dumps(run_metadata("run", 3, a=1, b=(1.0, 2.0)).to_dict())
    assert json.loads(text)["label"] == "run"


def test_a_foreign_schema_is_refused() -> None:
    from echoscribe.errors import SchemaError

    payload = run_metadata("run", 3).to_dict()
    payload["schema"] = METADATA_SCHEMA_VERSION + 1
    with pytest.raises(SchemaError, match="is not supported"):
        RunMetadata.from_dict(payload)


def test_a_missing_field_is_refused() -> None:
    from echoscribe.errors import SchemaError

    payload = run_metadata("run", 3).to_dict()
    del payload["seed"]
    with pytest.raises(SchemaError, match="missing 'seed'"):
        RunMetadata.from_dict(payload)


def test_a_tampered_label_fails_the_digest() -> None:
    from echoscribe.errors import SchemaError

    payload = run_metadata("run", 3).to_dict()
    payload["label"] = "tampered"
    with pytest.raises(SchemaError, match="digest mismatch"):
        RunMetadata.from_dict(payload)


def test_metadata_needs_a_label_and_an_integer_seed() -> None:
    with pytest.raises(ConfigError, match="needs a label"):
        RunMetadata("", 1, {}, {})
    with pytest.raises(ConfigError, match="seed must be an integer"):
        RunMetadata("run", 1.5, {}, {})


def test_a_config_entry_needs_a_name() -> None:
    with pytest.raises(ConfigError, match="needs a name"):
        run_metadata("run", 1).with_config("", 1)
