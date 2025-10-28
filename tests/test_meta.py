"""A fingerprint is only useful if it is stable, so stability is what is tested."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from echoscribe.errors import ConfigError
from echoscribe.meta import (
    FLOAT_DIGITS,
    METADATA_SCHEMA_VERSION,
    fingerprint,
    package_versions,
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
