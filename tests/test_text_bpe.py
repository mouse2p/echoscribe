"""BPE is a pure function of the corpus, so every stage is pinned."""

from __future__ import annotations

from collections import Counter

import pytest

from echoscribe.errors import ConfigError, SchemaError
from echoscribe.text.bpe import (
    BPE_SCHEMA_VERSION,
    WORD_END,
    BpeModel,
    count_pairs,
    detokenize,
    learn_bpe,
    merge_pair,
    tokenize,
    word_symbols,
)


def test_a_word_is_spelled_with_an_end_marker() -> None:
    assert word_symbols("ab") == ("a", "b", WORD_END)


def test_a_single_character_word_has_two_symbols() -> None:
    assert word_symbols("a") == ("a", WORD_END)


def test_an_empty_word_is_refused() -> None:
    with pytest.raises(ConfigError, match="cannot spell an empty word"):
        word_symbols("")


def test_a_non_string_word_is_refused() -> None:
    with pytest.raises(ConfigError, match="must be a string"):
        word_symbols(7)  # type: ignore[arg-type]


def test_pair_counts_cover_every_adjacent_pair() -> None:
    assert count_pairs(["ab"]) == Counter({("a", "b"): 1, ("b", WORD_END): 1})


def test_repeats_are_counted() -> None:
    counts = count_pairs(["ab", "ab", "ab"])
    assert counts[("a", "b")] == 3


def test_a_single_character_word_has_one_pair() -> None:
    assert count_pairs(["a"]) == Counter({("a", WORD_END): 1})


def test_pair_counts_of_nothing_are_empty() -> None:
    assert count_pairs([]) == Counter()


def test_a_merge_joins_every_occurrence() -> None:
    assert merge_pair(("a", "b", "a", "b"), ("a", "b")) == ("ab", "ab")


def test_a_merge_does_not_overlap_itself() -> None:
    assert merge_pair(("a", "a", "a"), ("a", "a")) == ("aa", "a")


def test_a_merge_that_does_not_occur_changes_nothing() -> None:
    symbols = ("x", "y", WORD_END)
    assert merge_pair(symbols, ("a", "b")) == symbols


def test_a_merge_can_produce_the_end_marker_pair() -> None:
    assert merge_pair(("a", WORD_END), ("a", WORD_END)) == (f"a{WORD_END}",)


def test_merging_an_empty_sequence_is_empty() -> None:
    assert merge_pair((), ("a", "b")) == ()


LOW = 3 * ["low"] + 2 * ["lowest"] + 4 * ["newer"] + 4 * ["wider"] + 1 * ["new"]


def test_the_first_merge_is_the_most_frequent_pair() -> None:
    merges = learn_bpe(LOW, 1)
    counts = count_pairs(LOW)
    best = max(counts.values())
    tied = sorted(pair for pair, count in counts.items() if count == best)
    assert merges[0] == tied[0]


def test_learning_stops_early_when_nothing_is_left_to_merge() -> None:
    assert learn_bpe(["ab"], 500) == (("a", "b"), ("ab", WORD_END))


def test_more_merges_are_learned_when_asked_for() -> None:
    assert len(learn_bpe(LOW, 3)) == 3
    assert len(learn_bpe(LOW, 10)) > 3


def test_the_merge_order_is_a_function_of_the_counts() -> None:
    import random

    shuffled = list(LOW)
    random.Random(0).shuffle(shuffled)
    assert learn_bpe(shuffled, 12) == learn_bpe(LOW, 12)


def test_learning_from_nothing_yields_no_merges() -> None:
    assert learn_bpe([], 10) == ()


def test_a_non_positive_merge_count_is_refused() -> None:
    with pytest.raises(ConfigError, match="num_merges must be positive"):
        learn_bpe(LOW, 0)


def test_merges_are_pairs_of_strings() -> None:
    for pair in learn_bpe(LOW, 6):
        assert isinstance(pair, tuple) and len(pair) == 2
        assert all(isinstance(part, str) and part for part in pair)


def test_a_tie_is_broken_lexicographically() -> None:
    # All four pairs occur once.  The smallest by (first, second) wins, and "</w>"
    # sorts before any letter because "<" is below "a" -- a consequence worth
    # pinning, since it decides the tail of every merge list.
    assert learn_bpe(["ab", "ba"], 1) == (("a", WORD_END),)


def test_frequency_beats_lexicographic_order() -> None:
    # (z, z) occurs four times against (a, b) once, so frequency wins outright.
    assert learn_bpe(["zzz", "zzz", "ab"], 1) == (("z", "z"),)


def test_a_longer_corpus_produces_a_prefix_of_the_same_order() -> None:
    short = learn_bpe(LOW, 4)
    long = learn_bpe(LOW, 12)
    assert long[:4] == short


def test_learning_twice_gives_an_identical_tuple() -> None:
    assert learn_bpe(LOW, 8) == learn_bpe(LOW, 8)


def test_a_corpus_of_one_repeated_word_still_merges_to_the_end_marker() -> None:
    merges = learn_bpe(["abc"] * 5, 10)
    assert merges[-1] == ("abc", WORD_END) or merges[-1][1] == WORD_END


def test_an_untrained_model_spells_a_word_out() -> None:
    assert BpeModel().encode("abc") == ("a", "b", "c", WORD_END)


def test_a_trained_model_merges_what_it_learned() -> None:
    model = BpeModel.learn(LOW, 10)
    pieces = model.encode("low")
    assert len(pieces) < 4
    assert BpeModel.decode(pieces) == "low"


def test_every_corpus_word_round_trips() -> None:
    model = BpeModel.learn(LOW, 20)
    for word in set(LOW):
        assert BpeModel.decode(model.encode(word)) == word


def test_an_unseen_word_still_round_trips() -> None:
    model = BpeModel.learn(LOW, 20)
    for word in ("newest", "lower", "zzz", "你好"):
        assert BpeModel.decode(model.encode(word)) == word


def test_encoding_never_produces_an_empty_piece() -> None:
    model = BpeModel.learn(LOW, 20)
    for word in ("low", "newer", "wider", "unseen"):
        assert all(model.encode(word))


def test_the_end_marker_survives_as_a_suffix() -> None:
    model = BpeModel.learn(LOW, 20)
    assert model.encode("low")[-1].endswith(WORD_END)


def test_encoding_is_deterministic() -> None:
    model = BpeModel.learn(LOW, 15)
    assert model.encode("newer") == model.encode("newer")


def test_ranks_follow_the_merge_order() -> None:
    model = BpeModel.learn(LOW, 6)
    for rank, pair in enumerate(model.merges):
        assert model.rank_of(pair) == rank
    assert model.rank_of(("\u0000", "\u0001")) is None


def test_a_model_reports_its_merge_count() -> None:
    assert len(BpeModel.learn(LOW, 5)) == 5


def test_a_duplicated_merge_is_refused() -> None:
    with pytest.raises(ConfigError, match="repeats"):
        BpeModel((("a", "b"), ("a", "b")))


def test_a_malformed_merge_is_refused() -> None:
    with pytest.raises(ConfigError, match="must be a pair of non-empty strings"):
        BpeModel((("a",),))  # type: ignore[arg-type]


def test_the_schema_version_is_declared() -> None:
    assert BPE_SCHEMA_VERSION == 1


def test_a_model_round_trips_through_a_dict() -> None:
    model = BpeModel.learn(LOW, 12)
    restored = BpeModel.from_dict(model.to_dict())
    assert restored == model
    assert restored.merges == model.merges


def test_the_payload_shape_is_fixed() -> None:
    payload = BpeModel.learn(LOW, 3).to_dict()
    assert set(payload) == {"schema", "merges", "digest"}
    assert payload["schema"] == BPE_SCHEMA_VERSION
    assert len(payload["digest"]) == 16


def test_the_digest_covers_the_merge_order() -> None:
    forward = BpeModel((("a", "b"), ("c", "d")))
    backward = BpeModel((("c", "d"), ("a", "b")))
    assert forward.digest() != backward.digest()


def test_the_digest_is_stable_for_the_same_merges() -> None:
    assert BpeModel.learn(LOW, 8).digest() == BpeModel.learn(LOW, 8).digest()


def test_an_empty_model_has_a_digest_too() -> None:
    assert len(BpeModel().digest()) == 16


def test_a_foreign_schema_is_refused() -> None:
    payload = BpeModel.learn(LOW, 3).to_dict()
    payload["schema"] = BPE_SCHEMA_VERSION + 1
    with pytest.raises(SchemaError, match="is not supported"):
        BpeModel.from_dict(payload)


def test_a_tampered_merge_list_fails_the_digest() -> None:
    payload = BpeModel.learn(LOW, 5).to_dict()
    payload["merges"][0] = ["z", "z"]
    with pytest.raises(SchemaError, match="digest mismatch"):
        BpeModel.from_dict(payload)


def test_a_missing_merges_field_is_refused() -> None:
    payload = BpeModel.learn(LOW, 3).to_dict()
    del payload["merges"]
    with pytest.raises(SchemaError, match="missing 'merges'"):
        BpeModel.from_dict(payload)


def test_the_serialized_form_is_json_encodable() -> None:
    import json

    text = json.dumps(BpeModel.learn(LOW + ["你好"], 8).to_dict(), ensure_ascii=False)
    assert "你好" in text or WORD_END in text


UTTERANCES = [
    "hello world",
    "你好 世界",
    "hello 你好 world",
    "please open the file",
    "don't stop",
    "mixed 混合 text 文本",
]


def test_detokenize_recovers_the_segmented_words() -> None:
    from echoscribe.text.normalize import normalize_text
    from echoscribe.text.segment import segment, token_texts

    model = BpeModel.learn(LOW, 20)
    for text in UTTERANCES:
        expected = token_texts(segment(normalize_text(text)))
        assert detokenize(tokenize(text, model)) == expected


def test_tokenize_puts_an_end_marker_on_every_word() -> None:
    model = BpeModel.learn(LOW, 20)
    pieces = tokenize("hello world", model)
    assert sum(1 for piece in pieces if piece.endswith(WORD_END)) == 2


def test_tokenize_normalises_before_splitting() -> None:
    model = BpeModel()
    assert tokenize("ＨＥＬＬＯ", model) == tokenize("hello", model)


def test_tokenize_of_an_empty_string_has_no_pieces() -> None:
    assert tokenize("", BpeModel()) == ()


def test_detokenize_refuses_a_truncated_piece_sequence() -> None:
    with pytest.raises(ConfigError, match="ends without"):
        detokenize(("h", "e"))


def test_detokenize_of_nothing_is_nothing() -> None:
    assert detokenize(()) == ()


def test_a_chinese_utterance_keeps_one_word_per_character() -> None:
    model = BpeModel.learn(LOW + ["你", "好"], 30)
    pieces = tokenize("你好", model)
    # Segmentation makes each character its own token and BPE only ever merges
    # inside a token, so no merge can join two characters together.
    assert detokenize(pieces) == ("你", "好")
    assert "".join(piece.replace(WORD_END, "") for piece in pieces) == "你好"


def test_tokenize_is_deterministic() -> None:
    model = BpeModel.learn(LOW, 15)
    assert tokenize("hello 你好 world", model) == tokenize("hello 你好 world", model)


def test_the_refactored_scan_still_picks_the_lowest_rank() -> None:
    # "ab" is merged before "bc", so "abc" must become "ab" + "c" and not
    # "a" + "bc", whichever way the scan is written.
    model = BpeModel((("a", "b"), ("b", "c")))
    assert model.encode("abc") == ("ab", "c", WORD_END)


def test_a_later_merge_still_applies_after_an_earlier_one() -> None:
    model = BpeModel((("a", "b"), ("ab", "c")))
    assert model.encode("abc") == ("abc", WORD_END)


def test_the_scan_reports_no_merge_for_an_unknown_word() -> None:
    model = BpeModel((("a", "b"),))
    assert model._next_merge(["x", "y"]) == -1
    assert model.encode("xy") == ("x", "y", WORD_END)


def test_the_text_subpackage_re_exports_the_tokenizer() -> None:
    from echoscribe import text

    assert text.BpeModel is BpeModel
    assert text.tokenize is tokenize
    assert text.WORD_END == WORD_END
    for name in ("BpeModel", "tokenize", "detokenize", "learn_bpe", "WORD_END"):
        assert name in text.__all__
