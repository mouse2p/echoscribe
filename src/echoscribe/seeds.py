"""Deterministic seeding for every randomized stage of the toolkit.

A single user-facing integer is never enough: corpus generation, noise mixing,
perturbation, subword training and model initialisation all consume randomness,
and they must not interfere.  :class:`Seed` therefore behaves like a *path* --
each stage derives a child seed from its parent plus a stable label -- so the
whole run is reproducible from one root while individual stages stay
independent.

Seeds are derived with SHA-256 over the label path rather than with Python's
``hash``, which is salted per process and would break reproducibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from echoscribe.errors import ConfigError, require

SEED_BITS = 63
SEED_MODULUS = 1 << SEED_BITS


@dataclass(frozen=True)
class Seed:
    """An immutable integer seed plus the label path that produced it."""

    value: int
    label: str = "root"

    def __post_init__(self) -> None:
        require(
            isinstance(self.value, int) and 0 <= self.value < SEED_MODULUS,
            f"seed value must be an int in [0, 2**{SEED_BITS}), got {self.value!r}",
            ConfigError,
        )
        require(bool(self.label), "seed label must not be empty", ConfigError)

    def digest(self) -> str:
        """Return the full hex digest backing this seed (used in reports)."""
        return hashlib.sha256(self.label.encode("utf-8")).hexdigest() + ":" + str(self.value)

    def fork(self, *parts: object) -> Seed:
        """Derive an independent child seed for a sub-stage of the same run.

        The child mixes the parent's *value* as well as its label path, so two
        runs with different roots never share a derived stage seed even when the
        stage labels are identical.
        """
        require(len(parts) > 0, "fork needs at least one label part")
        rendered = "/".join(repr(part) for part in parts)
        material = f"{self.value}\x00{self.label}\x00{rendered}"
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") % SEED_MODULUS
        return Seed(value, f"{self.label}/{rendered}")


def seed_of(*parts: object) -> Seed:
    """Build a seed from an arbitrary label path.

    ``seed_of("corpus", 3, "noise")`` is stable across processes and Python
    versions: parts are rendered with ``repr`` and hashed with SHA-256, then
    reduced modulo :data:`SEED_MODULUS`.
    """
    require(len(parts) > 0, "seed_of needs at least one label part")
    rendered = "/".join(repr(part) for part in parts)
    digest = hashlib.sha256(rendered.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % SEED_MODULUS
    return Seed(value, rendered)


#: Root seed used by examples and by the CLI when no seed is supplied.  Pinning
#: it keeps published synthetic numbers reproducible from a single integer.
DEFAULT_ROOT = Seed(20250106, "echoscribe")


def new_rng(seed: int | Seed | None = None) -> np.random.Generator:
    """Return a fresh ``numpy`` generator for ``seed``.

    ``None`` means :data:`DEFAULT_ROOT`, *not* OS entropy: echoscribe never
    produces an unreproducible run by accident.
    """
    if seed is None:
        resolved = DEFAULT_ROOT
    elif isinstance(seed, int):
        resolved = Seed(seed, f"int/{seed}")
    else:
        resolved = seed
    return np.random.default_rng(np.random.SeedSequence(resolved.value))
