"""The cache has to be transparent: same answer, fewer model calls."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.cache import JointCache
from echoscribe.decode.transducer import TableJoint, TransducerConfig, transducer_greedy
from echoscribe.errors import DecodeError


def _joint(seed: int = 0) -> TableJoint:
    rng = np.random.default_rng(seed)
    return TableJoint(rng.uniform(-2.0, 2.0, size=(6, 20, 3)))


def test_a_repeat_lookup_returns_the_identical_array() -> None:
    cache = JointCache(_joint())
    assert cache(0, ()) is cache(0, ())


def test_the_cached_value_matches_the_underlying_joint() -> None:
    joint = _joint()
    assert np.array_equal(JointCache(joint)(2, (1,)), joint(2, (1,)))


def test_hits_and_misses_are_counted() -> None:
    cache = JointCache(_joint())
    cache(0, ())
    cache(0, ())
    cache(1, ())
    assert cache.misses == 2
    assert cache.hits == 1
    assert cache.hit_ratio == pytest.approx(1 / 3)


def test_the_hit_ratio_of_a_fresh_cache_is_zero() -> None:
    assert JointCache(_joint()).hit_ratio == 0.0


def test_different_histories_are_different_keys() -> None:
    cache = JointCache(_joint())
    assert cache(0, (1,)) is not cache(0, (2,))
    assert cache.misses == 2


def test_the_capacity_bounds_the_entry_count() -> None:
    cache = JointCache(_joint(), capacity=2)
    cache(0, ())
    cache(1, ())
    cache(2, ())
    assert len(cache) == 2
    assert cache.evictions == 1


def test_eviction_is_least_recently_used() -> None:
    cache = JointCache(_joint(), capacity=2)
    first = cache(0, ())
    cache(1, ())
    cache(0, ())  # refresh the first entry so the second is now the oldest
    cache(2, ())  # must evict (1, ()) rather than (0, ())
    assert cache(0, ()) is first
    assert cache.hits == 2


def test_clearing_drops_entries_but_keeps_the_counters() -> None:
    cache = JointCache(_joint())
    cache(0, ())
    cache(0, ())
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 1
    assert cache.misses == 1


def test_a_cache_forwards_the_joint_dimensions() -> None:
    joint = _joint()
    cache = JointCache(joint)
    assert cache.num_frames == joint.num_frames
    assert cache.num_symbols == joint.num_symbols


def test_an_unbounded_cache_never_evicts() -> None:
    cache = JointCache(_joint(), capacity=None)
    for frame in range(6):
        cache(frame, ())
    assert len(cache) == 6
    assert cache.evictions == 0


def test_a_bad_capacity_is_refused() -> None:
    with pytest.raises(DecodeError, match="positive integer or None"):
        JointCache(_joint(), capacity=0)


def test_a_cached_joint_decodes_identically_to_the_raw_one() -> None:
    joint = _joint(seed=11)
    config = TransducerConfig(max_symbols_per_frame=3)
    direct = transducer_greedy(joint, config)
    cached = transducer_greedy(JointCache(joint, capacity=8), config)
    assert direct.symbols == cached.symbols
    assert direct.score == pytest.approx(cached.score)
