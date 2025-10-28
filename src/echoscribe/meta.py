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
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib import metadata

import numpy as np

from echoscribe.errors import ConfigError, require

#: Bumped when the serialized shape of :class:`RunMetadata` changes.
METADATA_SCHEMA_VERSION = 1

#: Digits kept when rendering a float, so a digest does not depend on the last bit.
FLOAT_DIGITS = 12

#: Separator between the parts of a canonical rendering.
LINE_SEPARATOR = chr(10)


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
