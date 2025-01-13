"""Seed derivation is stable across processes, which is the whole point."""

from __future__ import annotations

import pytest

from echoscribe.seeds import SEED_MODULUS, Seed


def test_seed_keeps_its_value_and_label() -> None:
    seed = Seed(1234, "corpus")
    assert seed.value == 1234
    assert seed.label == "corpus"


def test_seed_defaults_to_a_named_root() -> None:
    assert Seed(0).label == "root"


def test_seed_rejects_out_of_range_values() -> None:
    for bad in (-1, SEED_MODULUS, SEED_MODULUS + 7):
        with pytest.raises(ValueError):
            Seed(bad)


def test_seed_rejects_non_integer_values() -> None:
    with pytest.raises(ValueError):
        Seed(1.5)  # type: ignore[arg-type]


def test_seed_rejects_an_empty_label() -> None:
    with pytest.raises(ValueError):
        Seed(1, "")


def test_digest_mixes_label_and_value() -> None:
    left = Seed(7, "a").digest()
    right = Seed(7, "b").digest()
    assert left != right
    assert left.endswith(":7")
    assert len(left.split(":")[0]) == 64
