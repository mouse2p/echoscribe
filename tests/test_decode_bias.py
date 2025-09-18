"""The trie is checked as a state machine: advance, match, reset."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.decode.bias import (
    BiasConfig,
    BiasedJoint,
    BiasReport,
    TrieNode,
    advance,
    apply_bias,
    bias_greedy,
    build_trie,
    collect_bonus,
)
from echoscribe.decode.transducer import Joint, TableJoint
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


def _near_tie_joint(frames: int = 4) -> TableJoint:
    """Prefers symbol 2 by a hair, so a modest bonus flips it to symbol 1."""
    logits = np.full((frames, 8, 3), -5.0)
    logits[:, :, 0] = 4.5
    logits[:, 0, 2] = 4.1
    logits[:, 0, 1] = 4.0
    return TableJoint(logits)


def test_biasing_flips_a_close_choice() -> None:
    report = bias_greedy(_near_tie_joint(), [[1]], bias=BiasConfig(bonus=1.0))
    assert isinstance(report, BiasReport)
    assert report.changed
    assert 1 in report.biased_symbols
    assert 1 not in report.unbiased_symbols


def test_a_zero_bonus_changes_nothing() -> None:
    report = bias_greedy(_near_tie_joint(), [[1]], bias=BiasConfig(bonus=0.0))
    assert not report.changed
    assert report.biased_symbols == report.unbiased_symbols


def test_no_hotwords_changes_nothing() -> None:
    report = bias_greedy(_near_tie_joint(), [])
    assert not report.changed
    assert report.hotwords_matched() == 0


def test_a_matched_hotword_is_counted() -> None:
    report = bias_greedy(_near_tie_joint(), [[1]], bias=BiasConfig(bonus=1.0))
    assert report.hotwords_matched() == 1
    assert report.hotword_symbols == ((1,),)


def test_a_bonus_must_not_be_negative() -> None:
    with pytest.raises(ConfigError, match="must not be negative"):
        BiasConfig(bonus=-0.5)


def test_a_larger_bonus_does_not_weaken_the_effect() -> None:
    joint = _near_tie_joint()
    small = bias_greedy(joint, [[1]], bias=BiasConfig(bonus=0.5))
    large = bias_greedy(joint, [[1]], bias=BiasConfig(bonus=4.0))
    assert large.hotwords_matched() >= small.hotwords_matched()


def test_the_biased_joint_returns_a_normalised_log_distribution() -> None:
    wrapped = BiasedJoint(_near_tie_joint(), build_trie([[1]], 4.0))
    row = wrapped(0, ())
    assert np.all(row <= 1e-12)
    assert float(np.exp(row).sum()) == pytest.approx(1.0, rel=1e-12)


def test_a_large_bonus_does_not_produce_a_positive_score() -> None:
    from echoscribe.decode.transducer import transducer_greedy

    joint = _near_tie_joint()
    report = bias_greedy(joint, [[1]], bias=BiasConfig(bonus=50.0))
    assert report.biased_symbols == (1,)
    # The renormalisation is what makes this hold: without it the boosted symbol
    # would carry a logit above zero and every per-token score check downstream
    # would reject the decode.
    biased = transducer_greedy(BiasedJoint(joint, build_trie([[1]], 50.0)))
    assert all(token.score <= 0.0 for token in biased.tokens)
    assert biased.score <= 0.0


def test_apply_bias_alone_is_unnormalised_and_the_wrapper_is_not() -> None:
    joint = _near_tie_joint()
    root = build_trie([[1]], 4.0)
    raw = apply_bias(joint(0, ()), root)
    wrapped = BiasedJoint(joint, root)(0, ())
    assert float(np.exp(raw).sum()) > 1.0
    assert float(np.exp(wrapped).sum()) == pytest.approx(1.0, rel=1e-12)


def test_biasing_is_deterministic() -> None:
    joint = _near_tie_joint()
    first = bias_greedy(joint, [[1]], bias=BiasConfig(bonus=1.0))
    assert bias_greedy(joint, [[1]], bias=BiasConfig(bonus=1.0)).biased_symbols == (
        first.biased_symbols
    )


def test_the_biased_joint_satisfies_the_joint_protocol() -> None:
    wrapped = BiasedJoint(_near_tie_joint(), build_trie([[1]], 1.0))
    assert isinstance(wrapped, Joint)
    assert wrapped.num_frames == 4
    assert wrapped.num_symbols == 3


def test_the_biased_joint_memoises_its_trie_position() -> None:
    wrapped = BiasedJoint(_near_tie_joint(), build_trie([[1, 2]], 1.0))
    assert wrapped.position((1,)) is wrapped.position((1,))
    assert wrapped.position((1,)) is not wrapped.position((1, 2))


def test_a_multi_symbol_hotword_is_matched_across_frames() -> None:
    # Two frames that each narrowly prefer symbol 2; the hotword (1, 2) pays only
    # on its second symbol, so the bonus has to carry the decode through both.
    logits = np.full((2, 8, 3), -5.0)
    logits[:, :, 0] = 4.5
    logits[0, 0, 1] = 4.0
    logits[1, 1, 2] = 4.0
    joint = TableJoint(logits)
    report = bias_greedy(joint, [[1, 2]], bias=BiasConfig(bonus=1.5))
    assert report.unbiased_symbols == ()
    assert report.biased_symbols == (1, 2)
    assert report.hotwords_matched() == 1


def test_the_memo_grows_with_distinct_histories_only() -> None:
    joint = _near_tie_joint(frames=6)
    wrapped = BiasedJoint(joint, build_trie([[1, 2]], 1.0))
    for frame in range(6):
        wrapped(frame, ())
        wrapped(frame, (1,))
    assert len(wrapped._positions) == 2


def test_the_memo_returns_the_same_node_for_the_same_history() -> None:
    wrapped = BiasedJoint(_near_tie_joint(), build_trie([[1, 2, 1]], 1.0))
    first = wrapped.position((1, 2))
    for _ in range(4):
        assert wrapped.position((1, 2)) is first


def test_a_long_history_still_resolves_to_the_right_node() -> None:
    root = build_trie([[1, 2, 1]], 3.0)
    wrapped = BiasedJoint(_near_tie_joint(), root)
    final = root.children[1].children[2].children[1]
    assert wrapped.position((1, 2, 1)) is final
    assert final.bonus == pytest.approx(1.0)
    assert final.completes


def test_biasing_a_long_decode_stays_fast_enough_to_test() -> None:
    rng = np.random.default_rng(1)
    logits = rng.uniform(-1.0, 1.0, size=(60, 200, 4))
    joint = TableJoint(logits)
    from echoscribe.decode.transducer import TransducerConfig

    report = bias_greedy(
        joint, [[1, 2], [3, 1]], TransducerConfig(max_symbols_per_frame=3), BiasConfig(0.5)
    )
    assert isinstance(report.biased_symbols, tuple)


def test_bias_greedy_accepts_a_generator_of_hotwords() -> None:
    # The hotword list is materialised once, so a single-pass iterable works.
    report = bias_greedy(_near_tie_joint(), (hotword for hotword in [[1]]))
    assert report.hotword_symbols == ((1,),)
    assert report.hotwords_matched() >= 0


def test_bias_greedy_accepts_tuples_and_lists_alike() -> None:
    joint = _near_tie_joint()
    from_list = bias_greedy(joint, [[1]], bias=BiasConfig(bonus=1.0))
    from_tuple = bias_greedy(joint, [(1,)], bias=BiasConfig(bonus=1.0))
    assert from_list.biased_symbols == from_tuple.biased_symbols
