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

Two limits of this implementation are worth naming:

* the bonus is added to the logits and the row is renormalised, so a biased score
  is a log-probability under the *biased* distribution.  It ranks correctly and it
  is not a calibrated posterior: anything downstream that reads a decoder score as
  a confidence -- a calibration curve, a rejection threshold -- has to know whether
  biasing was on, which is why :func:`bias_greedy` reports the unbiased decode next
  to the biased one.
* the trie position is memoised per label history rather than threaded through the
  decoder loop, because the decoder only ever sees ``(frame, history)``.  That costs
  a dict lookup per state; a production decoder carries the node in its beam entry
  and pays nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from echoscribe.decode.transducer import Joint, TransducerConfig, transducer_greedy
from echoscribe.errors import ConfigError, DecodeError, require
from echoscribe.numeric import log_softmax


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


@dataclass(frozen=True)
class BiasConfig:
    """How strongly a hotword list is allowed to move the decode."""

    bonus: float = 2.0

    def __post_init__(self) -> None:
        require(
            self.bonus >= 0.0,
            f"a bias bonus must not be negative, got {self.bonus!r}",
            ConfigError,
        )


@dataclass(frozen=True)
class BiasReport:
    """What biasing changed, which is what makes the trade-off visible."""

    unbiased_symbols: tuple[int, ...]
    biased_symbols: tuple[int, ...]
    hotword_symbols: tuple[tuple[int, ...], ...]
    score_difference: float

    @property
    def changed(self) -> bool:
        """True when the emitted label sequence differs."""
        return self.unbiased_symbols != self.biased_symbols

    def hotwords_matched(self) -> int:
        """How many listed hotwords appear as a contiguous run in the output."""
        found = 0
        for hotword in self.hotword_symbols:
            for start in range(len(self.biased_symbols) - len(hotword) + 1):
                if self.biased_symbols[start : start + len(hotword)] == hotword:
                    found += 1
                    break
        return found


class BiasedJoint:
    """Wraps a joint and boosts the logits of labels that continue a hotword.

    The trie position is a pure function of the label history, so the wrapper can
    stay stateless with respect to the decoder loop and still be correct: it walks
    the history from the root, memoising each position it has already computed.
    """

    def __init__(self, joint: Joint, trie: TrieNode) -> None:
        self.joint = joint
        self.trie = trie
        self._positions: dict[tuple[int, ...], TrieNode] = {}

    @property
    def num_frames(self) -> int:
        """Forwarded from the wrapped joint."""
        return self.joint.num_frames

    @property
    def num_symbols(self) -> int:
        """Forwarded from the wrapped joint."""
        return self.joint.num_symbols

    def position(self, history: tuple[int, ...]) -> TrieNode:
        """Trie node reached by ``history``, computed at most once per history."""
        node = self._positions.get(history)
        if node is None:
            node = self.trie
            for symbol in history:
                node = advance(node, symbol, self.trie)
            self._positions[history] = node
        return node

    def __call__(self, frame: int, history: tuple[int, ...]) -> np.ndarray:
        """Biased log-probabilities for one state.

        The bonus is added and the row is renormalised, so what comes back is a real
        log distribution and a decoder can record it as a score.  Without that step
        a boosted symbol can score above zero, which is not a log-probability and
        would be rejected by every downstream check that assumes one.
        """
        boosted = apply_bias(self.joint(frame, history), self.position(history))
        return log_softmax(boosted)


def bias_greedy(
    joint: Joint,
    hotwords: Iterable[Sequence[int]],
    config: TransducerConfig | None = None,
    bias: BiasConfig | None = None,
) -> BiasReport:
    """Decode greedily with hotword bonuses, alongside the unbiased baseline.

    Both runs are reported because a biased hypothesis with nothing to compare it to
    is not a result: the interesting quantity is what the bonus changed, and that
    needs the baseline next to it.
    """
    settings = bias if bias is not None else BiasConfig()
    listed = tuple(tuple(int(symbol) for symbol in hotword) for hotword in hotwords)
    plain = transducer_greedy(joint, config)
    boosted = transducer_greedy(BiasedJoint(joint, build_trie(listed, settings.bonus)), config)
    return BiasReport(plain.symbols, boosted.symbols, listed, boosted.score - plain.score)
