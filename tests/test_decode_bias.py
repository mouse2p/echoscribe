"""The trie is checked as a state machine: advance, match, reset."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.bias import (
    TrieNode,
    advance,
    apply_bias,
    build_trie,
    collect_bonus,
)
from echoscribe.errors import ConfigError, DecodeError


def test_an_empty_trie_has_no_children() -> None:
    root = build_trie([])
    assert root.children == {}
    assert root.bonus == 0.0
    assert not root.completes


def test_a_single_symbol_hotword_pays_its_whole_bonus_at_once() -> None:
    root = build_trie([[4]], bonus=2.5)
    assert root.children[4].bonus == pytest.approx(2.5)
    assert root.children[4].completes


def test_a_multi_symbol_hotword_spreads_its_bonus_along_its_edges() -> None:
    root = build_trie([[1, 2, 3]], bonus=3.0)
    assert root.children[1].bonus == pytest.approx(1.0)
    assert not root.children[1].completes
    assert root.children[1].children[2].bonus == pytest.approx(1.0)
    final = root.children[1].children[2].children[3]
    assert final.bonus == pytest.approx(1.0)
    assert final.completes


def test_shared_prefixes_add_their_shares() -> None:
    root = build_trie([[1, 2], [1, 3]], bonus=2.0)
    assert set(root.children[1].children) == {2, 3}
    assert root.children[1].bonus == pytest.approx(2.0)


def test_a_prefix_shared_by_two_hotwords_of_different_length_splits_evenly() -> None:
    root = build_trie([[1], [1, 2]], bonus=3.0)
    assert root.children[1].bonus == pytest.approx(3.0 + 1.5)
    assert root.children[1].completes


def test_advance_follows_a_matching_edge() -> None:
    root = build_trie([[1, 2]], bonus=1.0)
    assert advance(root, 1) is root.children[1]


def test_advance_stays_put_when_nothing_matches_and_no_root_is_given() -> None:
    root = build_trie([[1, 2]], bonus=1.0)
    node = advance(root, 1)
    assert advance(node, 7) is node


def test_advance_resets_to_the_root_when_one_is_given() -> None:
    root = build_trie([[1, 2]], bonus=1.0)
    assert advance(advance(root, 1), 7, root) is root


def test_a_hotword_can_restart_immediately_after_a_miss() -> None:
    root = build_trie([[5, 6]], bonus=1.0)
    node = advance(advance(root, 5), 9, root)
    assert node is root
    assert advance(node, 5, root) is root.children[5]


def test_a_reset_that_matches_from_the_root_goes_straight_there() -> None:
    root = build_trie([[1, 2]], bonus=1.0)
    node = advance(root, 1)
    assert advance(node, 2, root) is root.children[1].children[2]


def test_apply_bias_adds_only_to_continuing_labels() -> None:
    root = build_trie([[2]], bonus=3.0)
    boosted = apply_bias(np.array([-1.0, -1.0, -1.0, -1.0]), root)
    assert boosted[2] == pytest.approx(2.0)
    assert np.allclose(boosted[[0, 1, 3]], -1.0)


def test_apply_bias_does_not_modify_its_input() -> None:
    logits = np.zeros(3)
    apply_bias(logits, build_trie([[1]], bonus=2.0))
    assert np.all(logits == 0.0)


def test_apply_bias_on_an_empty_trie_is_the_identity() -> None:
    logits = np.array([0.5, -1.5])
    assert np.array_equal(apply_bias(logits, build_trie([])), logits)


def test_apply_bias_ignores_a_symbol_outside_the_alphabet() -> None:
    assert np.all(apply_bias(np.zeros(3), build_trie([[9]], bonus=1.0)) == 0.0)


def test_apply_bias_rejects_a_matrix() -> None:
    with pytest.raises(DecodeError, match="one frame of logits"):
        apply_bias(np.zeros((2, 3)), build_trie([]))


def test_collect_bonus_reads_the_node_value() -> None:
    root = build_trie([[1, 2]], bonus=1.5)
    assert collect_bonus(root) == 0.0
    assert collect_bonus(advance(advance(root, 1), 2)) == pytest.approx(0.75)


def test_an_empty_hotword_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one symbol"):
        build_trie([[]])


def test_a_negative_symbol_is_refused() -> None:
    with pytest.raises(ConfigError, match="negative symbol id"):
        build_trie([[1, -2]])


def test_a_built_trie_is_immutable() -> None:
    with pytest.raises(AttributeError):
        TrieNode().bonus = 1.0  # type: ignore[misc]
