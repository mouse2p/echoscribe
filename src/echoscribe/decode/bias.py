"""Hotword biasing by shallow fusion over a trie.

A recognizer that has to hear a name, a product or a command reliably needs more
than a better acoustic model: it needs to be *told* what to listen for.  Context
biasing does that by adding a bonus to the log-probability of any label that
continues a listed hotword, so a hypothesis spelling one is preferred when the
acoustic evidence is close.

The mechanism is shallow fusion inside the decoding step rather than a rescoring
pass afterwards.  That distinction matters: rescoring can only reorder hypotheses
the first pass already produced, so a hotword the acoustic model never considered
cannot be recovered.  Biasing during the search can.

The trie carries one number per node -- the bonus for *arriving* there -- which
keeps the update rule to a single addition per emitted label and makes the scheme
explainable: the bias a hypothesis collects is the sum of the node bonuses along
the path its labels trace through the trie.

A caution the numbers bear out: a bonus large enough to guarantee a hotword is also
large enough to insert it where it was never said.  Biasing trades recall on the
listed words for precision on everything else, and the trade is set by the bonus,
not by the algorithm.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from echoscribe.errors import ConfigError, DecodeError, require


@dataclass(frozen=True)
class TrieNode:
    """One trie position: where each label leads, and what arriving here earns."""

    children: Mapping[int, TrieNode] = field(default_factory=dict)
    bonus: float = 0.0
    completes: bool = False

    def __post_init__(self) -> None:
        require(
            all(symbol >= 0 for symbol in self.children),
            "a trie edge must be a non-negative symbol id",
            ConfigError,
        )

    def child(self, symbol: int) -> TrieNode | None:
        """The node a label leads to, or ``None`` when no hotword continues."""
        return self.children.get(symbol)


@dataclass
class _MutableNode:
    """Building block for :func:`build_trie`, frozen into a :class:`TrieNode`."""

    children: dict[int, _MutableNode] = field(default_factory=dict)
    bonus: float = 0.0
    completes: bool = False

    def freeze(self) -> TrieNode:
        """Return the immutable form of this subtree."""
        return TrieNode(
            children={symbol: node.freeze() for symbol, node in sorted(self.children.items())},
            bonus=self.bonus,
            completes=self.completes,
        )


def build_trie(hotwords: Iterable[Sequence[int]], bonus: float = 1.0) -> TrieNode:
    """Build a trie from label sequences, giving every hotword the same total bonus.

    The bonus is split evenly along the hotword's edges rather than paid only at the
    end.  Paying at the end looks cleaner but does not work: the decoder has to
    choose the hotword's *first* symbol before any reward is in reach, so an
    end-only bonus can never pull a hypothesis onto a multi-symbol hotword.  Split
    evenly, completing the hotword still collects exactly ``bonus``, and a prefix
    collects the fraction of it that the evidence so far has earned.

    Two hotwords sharing a prefix add their shares on the common edges, which is the
    right behaviour: a symbol that continues either hotword is worth more than one
    that continues neither.
    """
    root = _MutableNode()
    for hotword in hotwords:
        symbols = tuple(int(symbol) for symbol in hotword)
        require(len(symbols) > 0, "a hotword must contain at least one symbol", ConfigError)
        require(
            all(symbol >= 0 for symbol in symbols),
            f"hotword {symbols} contains a negative symbol id",
            ConfigError,
        )
        share = float(bonus) / len(symbols)
        node = root
        for symbol in symbols:
            node = node.children.setdefault(symbol, _MutableNode())
            node.bonus += share
        node.completes = True
    return root.freeze()


def apply_bias(logits: np.ndarray, node: TrieNode) -> np.ndarray:
    """Return ``logits`` with each continuing label's bonus added.

    The input is never modified.  A decoder keeps the unbiased logits for scoring
    and biases only the *choice*, because reporting a biased score as if it were a
    posterior would make every confidence estimate downstream too high.
    """
    require(
        logits.ndim == 1,
        f"apply_bias expects one frame of logits, got shape {logits.shape}",
        DecodeError,
    )
    boosted = np.array(logits, dtype=np.float64, copy=True)
    for symbol, child in node.children.items():
        if symbol < boosted.shape[0]:
            boosted[symbol] += child.bonus
    return boosted


def advance(node: TrieNode, symbol: int, root: TrieNode | None = None) -> TrieNode:
    """Move to the child for ``symbol``, falling back to ``root`` on a miss.

    Resetting to the root rather than staying put is what lets a hotword start at
    any position: after a miss the next label is matched against every hotword's
    first symbol again.  When ``root`` is omitted the node is returned unchanged,
    which is what a caller tracking its own reset wants.
    """
    matched = node.child(symbol)
    if matched is not None:
        return matched
    fallback = node if root is None else root
    rematch = fallback.child(symbol)
    return fallback if rematch is None else rematch


def collect_bonus(node: TrieNode) -> float:
    """Bonus earned by arriving at ``node``, which a caller should bank once."""
    return float(node.bonus)
