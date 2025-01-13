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
