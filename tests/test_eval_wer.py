"""The scorer's arithmetic is checked against the alignment it came from."""

from __future__ import annotations

import pytest

from echoscribe.errors import EvalError
from echoscribe.eval.wer import cer, format_summary, score_corpus, score_tokens, wer


def test_the_breakdown_sums_to_the_error_count() -> None:
    report = score_tokens("the cat sat".split(), "a cat sat down".split())
    assert report.errors == report.substitutions + report.deletions + report.insertions
    assert report.errors == report.alignment.errors


def test_the_deletion_insertion_gap_explains_the_length_difference() -> None:
    for ref, hyp in [
        ("a b c", "a b c"),
        ("a b c", "a b"),
        ("a b", "a b c"),
        ("a b c d", "b d e"),
    ]:
        report = score_tokens(ref.split(), hyp.split())
        assert report.deletions - report.insertions == report.reference_length - len(hyp.split())


def test_the_alignment_inside_the_report_still_rebuilds_the_hypothesis() -> None:
    reference = "the cat sat on the mat".split()
    hypothesis = "a cat sat on the mat now".split()
    report = score_tokens(reference, hypothesis)
    assert report.alignment.apply(reference) == tuple(hypothesis)


def test_an_identical_pair_scores_zero() -> None:
    report = score_tokens(["a", "b"], ["a", "b"])
    assert report.errors == 0
    assert report.error_rate == 0.0


def test_scoring_is_a_pure_function_of_its_inputs() -> None:
    first = score_tokens("a b c".split(), "a x c".split())
    second = score_tokens("a b c".split(), "a x c".split())
    assert first == second


def test_the_report_keeps_its_tokens_as_tuples() -> None:
    report = score_tokens(["a", "b"], ["b"])
    assert report.reference_tokens == ("a", "b")
    assert report.hypothesis_tokens == ("b",)


def test_a_report_cannot_be_modified_after_the_fact() -> None:
    from dataclasses import FrozenInstanceError

    report = score_tokens(["a"], ["b"])
    with pytest.raises(FrozenInstanceError):
        report.substitutions = 0


@pytest.mark.parametrize("seed", range(6))
def test_random_pairs_keep_every_invariant(seed: int) -> None:
    import random

    vocabulary = ["a", "b", "c", "d"]
    rng = random.Random(seed)
    for _ in range(25):
        reference = [rng.choice(vocabulary) for _ in range(rng.randint(1, 8))]
        hypothesis = [rng.choice(vocabulary) for _ in range(rng.randint(0, 8))]
        report = score_tokens(reference, hypothesis)
        assert report.errors == report.alignment.errors
        assert report.deletions - report.insertions == len(reference) - len(hypothesis)
        assert report.alignment.apply(reference) == tuple(hypothesis)
        assert report.error_rate >= 0.0


def test_an_empty_reference_with_a_hypothesis_is_refused() -> None:
    with pytest.raises(EvalError, match="the reference is empty"):
        score_tokens([], ["anything"])


def test_two_empty_sequences_score_zero() -> None:
    report = score_tokens([], [])
    assert report.errors == 0
    assert report.error_rate == 0.0
    assert report.alignment.operations == ()


def test_an_empty_hypothesis_against_a_reference_is_all_deletions() -> None:
    report = score_tokens(["a", "b", "c"], [])
    assert report.deletions == 3
    assert report.substitutions == 0
    assert report.insertions == 0
    assert report.error_rate == 1.0


def test_a_single_deletion_over_three_words() -> None:
    report = wer("the cat sat", "the cat")
    assert (report.substitutions, report.deletions, report.insertions) == (0, 1, 0)
    assert report.error_rate == round(1 / 3, 6)


def test_the_kitten_sitting_example_in_characters() -> None:
    report = cer("kitten", "sitting")
    assert report.errors == 3
    assert report.reference_length == 6
    assert report.error_rate == 0.5


def test_a_chinese_sentence_missing_one_character() -> None:
    report = cer("今天天气不错", "今天气不错")
    assert report.deletions == 1
    assert report.errors == 1
    assert report.reference_length == 6


def test_whitespace_never_counts_as_a_character() -> None:
    assert cer("a b", "ab").errors == 0


def test_wer_splits_on_any_whitespace_run() -> None:
    report = wer("the   cat\nsat", "the cat sat")
    assert report.errors == 0


def test_wer_and_cer_disagree_by_design_on_the_same_strings() -> None:
    # One word replaced by another is a single word error but several character
    # errors; neither number is wrong, they answer different questions.
    word_report = wer("hello world", "hello word")
    char_report = cer("hello world", "hello word")
    assert word_report.errors == 1
    assert char_report.errors == 1
    assert word_report.reference_length == 2
    assert char_report.reference_length == 10


def test_the_corpus_rate_is_micro_averaged() -> None:
    pairs = [
        (["a", "b", "c", "d"], ["a", "b", "c", "x"]),  # 1 error over 4 tokens
        (["y"], ["z"]),  # 1 error over 1 token
    ]
    report = score_corpus(pairs)
    assert report.errors == 2
    assert report.reference_length == 5
    assert report.error_rate == 0.4
    per_utterance_mean = sum(item.error_rate for item in report.reports) / 2
    assert report.error_rate != per_utterance_mean


def test_corpus_totals_are_the_sums_of_the_parts() -> None:
    pairs = [
        (["a", "b"], ["a"]),
        (["c"], ["c", "d"]),
        (["e", "f"], ["e", "f"]),
    ]
    report = score_corpus(pairs)
    assert report.utterances == 3
    assert report.deletions == 1
    assert report.insertions == 1
    assert report.substitutions == 0
    assert report.errors == 2
    assert report.reference_length == 5


def test_an_empty_corpus_scores_zero_without_dividing() -> None:
    report = score_corpus([])
    assert report.utterances == 0
    assert report.error_rate == 0.0


def test_a_corpus_pair_with_an_empty_reference_is_still_refused() -> None:
    with pytest.raises(EvalError, match="the reference is empty"):
        score_corpus([(["a"], ["a"]), ([], ["b"])])


def test_the_corpus_report_keeps_every_utterance_report() -> None:
    pairs = [(list("abc"), list("axc")), (list("de"), list("de"))]
    report = score_corpus(pairs)
    assert [item.errors for item in report.reports] == [1, 0]


def test_the_summary_line_has_a_fixed_format() -> None:
    report = wer("the cat sat on the mat", "the cat sit on a mat")
    assert format_summary(report) == "WER 33.33% | S2 D0 I0 | N6"


def test_the_label_is_caller_chosen() -> None:
    report = cer("今天天气不错", "今天气不错")
    assert format_summary(report, label="CER") == "CER 16.67% | S0 D1 I0 | N6"


def test_a_perfect_hypothesis_reads_as_zero() -> None:
    assert format_summary(wer("a b", "a b")) == "WER 0.00% | S0 D0 I0 | N2"


def test_the_summary_of_an_empty_pair_is_defined() -> None:
    assert format_summary(score_tokens([], [])) == "WER 0.00% | S0 D0 I0 | N0"


def test_a_rate_above_one_hundred_percent_is_shown_not_capped() -> None:
    report = wer("hello", "hello world how are you")
    assert format_summary(report) == "WER 400.00% | S0 D0 I4 | N1"


def test_lists_and_tuples_score_identically() -> None:
    from_list = score_tokens(["a", "b"], ["a", "c"])
    from_tuple = score_tokens(("a", "b"), ("a", "c"))
    assert from_list == from_tuple


def test_a_generator_of_pairs_can_build_a_corpus() -> None:
    pairs = ((list(ref), list(hyp)) for ref, hyp in [("ab", "ac"), ("de", "de")])
    report = score_corpus(pairs)
    assert report.utterances == 2
    assert report.errors == 1
    assert report.reference_length == 4
    assert report.error_rate == 0.25
