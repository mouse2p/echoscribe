"""Run metadata: what produced a result, in a form that can be checked.

A measured number is worthless without the conditions it was measured under, and
"the conditions" is a bundle of things that are easy to lose: the package version,
the NumPy version, the seed, and every configuration object that touched the run.
This module turns that bundle into one short digest that can be printed in a report,
stored next to a result, and compared later.

Two properties are the point:

*deterministic.*  The same configuration produces the same digest on every run and
every machine.  That rules out timestamps, dict iteration order and object
addresses, all of which would make a digest change without anything meaningful
having changed.
*canonical.*  Nested dataclasses are rendered by sorted field name, so two
configurations that differ only in the order their fields were passed in are the
same configuration and digest identically.

There is deliberately no timestamp here.  A report that needs one should add it at
write time, outside the digest, so that re-running the same experiment produces the
same digest and the difference is visible.

The alternative -- stamping the run and digesting everything -- is the more common
design and it is worse for exactly the use case that matters.  A digest that changes
on every run cannot answer "is this the same experiment as the one that produced
this number?", which is the only question a reproducibility claim needs answered.
Version control already records when a commit happened; a results file does not have
to duplicate that, and doing so costs the ability to compare.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from importlib import metadata

import numpy as np

from echoscribe.errors import ConfigError, SchemaError, require

#: Bumped when the serialized shape of :class:`RunMetadata` changes.
METADATA_SCHEMA_VERSION = 1

#: Digits kept when rendering a float, so a digest does not depend on the last bit.
FLOAT_DIGITS = 12

#: Separator between the parts of a canonical rendering.
LINE_SEPARATOR = chr(10)


@lru_cache(maxsize=1)
def _installed_versions() -> tuple[tuple[str, str], ...]:
    """Memoised version table; versions cannot change inside one process."""
    return tuple(sorted(package_versions().items()))


def cached_versions() -> dict[str, str]:
    """The version table, served from a one-entry cache after the first call."""
    return dict(_installed_versions())


def package_versions() -> dict[str, str]:
    """Versions of everything that could change a result.

    ``torch`` is reported through its installed distribution rather than imported,
    because importing it costs a second and a fingerprint should be cheap enough to
    compute on every run.  An absent optional dependency is recorded as ``absent``
    rather than omitted, so a digest taken with and without the extra differs.
    """
    import echoscribe

    versions = {
        "echoscribe": echoscribe.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    for optional in ("torch",):
        try:
            versions[optional] = metadata.version(optional)
        except metadata.PackageNotFoundError:
            versions[optional] = "absent"
    return versions


def stable_repr(value: object) -> str:
    """Render a configuration value as a canonical string.

    Dataclasses are expanded by sorted field name, mappings by sorted key, and
    floats are rounded to :data:`FLOAT_DIGITS` so a value that differs only in the
    bits nobody chose does not change the digest.  Arrays are reduced to their shape
    and a digest of their bytes, because a fingerprint is not the place to embed a
    feature matrix.
    """
    if is_dataclass(value) and not isinstance(value, type):
        parts = [
            f"{item.name}={stable_repr(getattr(value, item.name))}"
            for item in fields(value)
            if not item.name.startswith("_") and item.repr
        ]
        return f"{type(value).__name__}({', '.join(sorted(parts))})"
    if isinstance(value, Mapping):
        entries = [
            f"{stable_repr(key)}: {stable_repr(item)}"
            for key, item in sorted(value.items(), key=lambda pair: stable_repr(pair[0]))
        ]
        return "{" + ", ".join(entries) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(stable_repr(item) for item in value) + "]"
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]
        return f"array(shape={value.shape},dtype={value.dtype},sha256={digest})"
    if isinstance(value, float):
        return repr(round(value, FLOAT_DIGITS))
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return repr(value)
    raise ConfigError(f"cannot fingerprint a value of type {type(value).__name__}")


def fingerprint(configs: Mapping[str, object]) -> str:
    """A short hex digest identifying a bundle of configuration values."""
    require(
        all(isinstance(key, str) for key in configs),
        "configuration keys must be strings",
        ConfigError,
    )
    rendered = "\n".join(f"{key}={stable_repr(configs[key])}" for key in sorted(configs))
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RunMetadata:
    """Everything a result needs in order to be reproduced or refuted."""

    label: str
    seed: int
    versions: Mapping[str, str]
    configs: Mapping[str, object]

    def __post_init__(self) -> None:
        require(bool(self.label), "run metadata needs a label", ConfigError)
        require(
            isinstance(self.seed, int) and not isinstance(self.seed, bool),
            f"seed must be an integer, got {self.seed!r}",
            ConfigError,
        )

    def canonical_configs(self) -> dict:
        """Every configuration value rendered to its canonical string form.

        A value that is already a string passes through unchanged, which is what
        makes canonicalisation idempotent: metadata read back from a file -- where
        the configs are stored as these very strings -- digests identically to the
        object that wrote it.
        """
        return {
            key: value if isinstance(value, str) else stable_repr(value)
            for key, value in self.configs.items()
        }

    def canonical(self) -> str:
        """The canonical rendering :meth:`digest` hashes.

        Exposed separately so a test can show *why* two digests differ instead of
        only asserting that they do.
        """
        parts = (
            f"label={self.label}",
            f"seed={self.seed}",
            f"versions={stable_repr(dict(self.versions))}",
            f"configs={stable_repr(self.canonical_configs())}",
        )
        return LINE_SEPARATOR.join(parts)

    def digest(self) -> str:
        """A short digest over the label, seed, versions and configuration bundle."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]

    def with_config(self, name: str, value: object) -> RunMetadata:
        """A copy with one more named configuration entry."""
        require(bool(name), "a configuration entry needs a name", ConfigError)
        configs = dict(self.configs)
        configs[name] = value
        return RunMetadata(self.label, self.seed, dict(self.versions), configs)

    def to_dict(self) -> dict:
        """A JSON-ready mapping, tagged with the schema version."""
        return {
            "schema": METADATA_SCHEMA_VERSION,
            "label": self.label,
            "seed": self.seed,
            "versions": dict(self.versions),
            "configs": self.canonical_configs(),
            "digest": self.digest(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunMetadata:
        """Rebuild metadata, refusing a schema this build does not read.

        Configurations come back as their canonical *strings*, not as the objects
        they were: a reader in a later version has no way to reconstruct a dataclass
        it has never seen, and pretending otherwise is how a report becomes
        unreadable.  The digest is checked so a hand-edited file is caught.
        """
        version = payload.get("schema")
        require(
            version == METADATA_SCHEMA_VERSION,
            f"run metadata schema {version!r} is not supported; this build reads "
            f"{METADATA_SCHEMA_VERSION}",
            SchemaError,
        )
        for name in ("label", "seed", "versions", "configs"):
            require(name in payload, f"run metadata is missing {name!r}", SchemaError)
        seed = payload["seed"]
        versions = payload["versions"]
        configs = payload["configs"]
        # Explicit raises rather than require(): the checker only narrows the
        # types of seed/versions/configs when it can see the branch itself.
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SchemaError(f"run metadata seed must be an integer, got {seed!r}")
        if not isinstance(versions, dict) or not isinstance(configs, dict):
            raise SchemaError(
                f"run metadata versions and configs must be mappings, got "
                f"{type(versions).__name__} and {type(configs).__name__}"
            )
        restored = cls(
            label=str(payload["label"]),
            seed=seed,
            versions=dict(versions),
            configs=dict(configs),
        )
        recorded = payload.get("digest")
        require(
            recorded is None or recorded == restored.digest(),
            f"run metadata digest mismatch: the file says {recorded!r}, the contents "
            f"hash to {restored.digest()!r}",
            SchemaError,
        )
        return restored


def run_metadata(label: str, seed: int, **configs: object) -> RunMetadata:
    """Build metadata for a run, capturing the current package versions."""
    return RunMetadata(label, seed, package_versions(), dict(configs))
