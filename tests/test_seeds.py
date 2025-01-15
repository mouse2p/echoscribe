"""Seed derivation is stable across processes, which is the whole point."""

from __future__ import annotations

import pytest

from echoscribe.seeds import DEFAULT_ROOT, SEED_MODULUS, Seed, new_rng, seed_of


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


def test_seed_of_is_stable_for_the_same_path() -> None:
    assert seed_of("corpus", 3).value == seed_of("corpus", 3).value


def test_seed_of_distinguishes_order_and_types() -> None:
    values = {
        seed_of("a", "b").value,
        seed_of("b", "a").value,
        seed_of(1, 2).value,
        seed_of("1", "2").value,
    }
    assert len(values) == 4


def test_seed_of_labels_carry_the_full_path() -> None:
    assert seed_of("corpus", 3, "noise").label == "'corpus'/3/'noise'"


def test_seed_of_values_stay_inside_the_declared_range() -> None:
    for index in range(64):
        value = seed_of("range", index).value
        assert 0 <= value < SEED_MODULUS


def test_seed_of_golden_values() -> None:
    # Pinned so an accidental change to the derivation is caught deliberately.
    assert seed_of("echoscribe", "golden").value == 7850220504658308925
    assert seed_of("corpus", 3).value == 7689283219961177129


def test_seed_of_distribution_is_not_degenerate() -> None:
    values = [seed_of("spread", index).value for index in range(512)]
    assert len(set(values)) == len(values)
    # A 63-bit space sampled 512 times should spread over the whole range.
    assert max(values) > SEED_MODULUS // 2
    assert min(values) < SEED_MODULUS // 2


def test_fork_changes_the_value_and_extends_the_label() -> None:
    parent = Seed(11, "root")
    child = parent.fork("noise")
    assert child.value != parent.value
    assert child.label == "root/'noise'"


def test_fork_is_deterministic_for_the_same_path() -> None:
    assert Seed(5, "root").fork("a", 2).value == Seed(5, "root").fork("a", 2).value


def test_fork_is_order_sensitive() -> None:
    parent = Seed(5, "root")
    assert parent.fork("a", "b").value != parent.fork("b", "a").value


def test_forks_of_the_same_parent_are_independent() -> None:
    parent = Seed(5, "root")
    values = {parent.fork(name).value for name in ("noise", "speed", "volume", "rir")}
    assert len(values) == 4


def test_the_same_stage_label_differs_across_roots() -> None:
    assert Seed(1, "run").fork("noise").value != Seed(2, "run").fork("noise").value


def test_fork_requires_a_label_part() -> None:
    with pytest.raises(ValueError, match="at least one label part"):
        Seed(3).fork()


def test_nested_forks_build_a_readable_path() -> None:
    seed = Seed(9, "run").fork("corpus").fork("utterance", 4)
    assert seed.label == "run/'corpus'/'utterance'/4"


def test_new_rng_replays_the_same_stream() -> None:
    first = new_rng(Seed(4, "a")).standard_normal(8)
    second = new_rng(Seed(4, "a")).standard_normal(8)
    assert list(first) == list(second)


def test_new_rng_accepts_a_bare_integer() -> None:
    assert list(new_rng(7).random(3)) == list(new_rng(Seed(7, "int/7")).random(3))


def test_new_rng_defaults_to_the_pinned_root() -> None:
    assert list(new_rng().random(4)) == list(new_rng(DEFAULT_ROOT).random(4))


def test_new_rng_streams_differ_across_forks() -> None:
    root = Seed(4, "a")
    assert list(new_rng(root.fork("x")).random(4)) != list(new_rng(root.fork("y")).random(4))


def test_default_root_is_a_fixed_documented_seed() -> None:
    assert DEFAULT_ROOT.value == 20250106
    assert DEFAULT_ROOT.label == "echoscribe"


def test_independent_generators_do_not_advance_each_other() -> None:
    root = Seed(21, "run")
    left = new_rng(root.fork("left"))
    right = new_rng(root.fork("right"))
    expected = list(new_rng(root.fork("right")).random(4))
    left.random(64)
    assert list(right.random(4)) == expected


def test_seed_values_never_collide_for_a_stage_sweep() -> None:
    root = Seed(101, "sweep")
    values = [root.fork("utterance", index).value for index in range(256)]
    assert len(set(values)) == 256


def test_generators_produce_usable_noise_statistics() -> None:
    # A sanity check that the seeded stream is a real generator, not a constant.
    draws = new_rng(Seed(33, "stats")).standard_normal(4096)
    assert abs(float(draws.mean())) < 0.1
    assert 0.8 < float(draws.std()) < 1.2
