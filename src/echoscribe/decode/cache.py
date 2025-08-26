"""A bounded cache in front of a transducer joint.

The joint is called once per decoder step, and a search that revisits a state -- a
beam, a lattice rescore, or a re-decode of the same audio under a different bias --
asks for the same ``(frame, history)`` more than once.  Answering means running the
predictor and the joint network, so the repeat is not free.

The cache is keyed on the exact history tuple, which is the honest key: two histories
of the same length are different predictor states.  It is bounded, because an
unbounded cache over a long utterance grows with the number of distinct histories the
search touched and has no natural stopping point.  Eviction is least-recently-used,
which matches how a beam search revisits states.

Cached arrays are returned by reference, not copied.  A caller that writes into one
corrupts the cache for everyone else, so treat the result as read-only.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from echoscribe.decode.transducer import Joint
from echoscribe.errors import DecodeError, require


class JointCache:
    """An LRU memo for :class:`~echoscribe.decode.transducer.Joint` calls."""

    def __init__(self, joint: Joint, capacity: int | None = 4096) -> None:
        if capacity is not None:
            require(
                isinstance(capacity, int) and not isinstance(capacity, bool) and capacity >= 1,
                f"capacity must be a positive integer or None, got {capacity!r}",
                DecodeError,
            )
        self.joint = joint
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._entries: OrderedDict[tuple[int, tuple[int, ...]], np.ndarray] = OrderedDict()

    @property
    def num_frames(self) -> int:
        """Forwarded from the wrapped joint, so a cache is a drop-in replacement."""
        return self.joint.num_frames

    @property
    def num_symbols(self) -> int:
        """Forwarded from the wrapped joint."""
        return self.joint.num_symbols

    @property
    def hit_ratio(self) -> float:
        """Fraction of lookups served from the cache; zero before any lookup."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._entries)

    def __call__(self, frame: int, history: tuple[int, ...]) -> np.ndarray:
        """Return the joint output for one state, computing it at most once."""
        key = (int(frame), tuple(history))
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return cached
        self.misses += 1
        value = self.joint(frame, history)
        self._entries[key] = value
        if self.capacity is not None and len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self.evictions += 1
        return value

    def clear(self) -> None:
        """Drop every entry; the counters stay, because they describe the run."""
        self._entries.clear()
