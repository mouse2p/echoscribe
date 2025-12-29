"""Mixed tokenization is the contract every code-switch score rests on."""

from __future__ import annotations

import pytest

from echoscribe.eval.codeswitch import (
    count_code_switches,
    format_code_switch,
    mixed_token_sequence,
    score_mixed,
)
from echoscribe.eval.wer import cer, wer


def test_han_characters_become_individual_tokens() -> None:
    assert mixed_token_sequence("今天") == (("今", "han"), ("天", "han"))


def test_latin_words_stay_together() -> None:
    assert mixed_token_sequence("use AI") == (("use", "latin"), ("AI", "latin"))


def test_a_mixed_sentence_keeps_every_script_run_in_order() -> None:
    assert mixed_token_sequence("今天use AI好吧") == (
        ("今", "han"),
        ("天", "han"),
        ("use", "latin"),
        ("AI", "latin"),
        ("好", "han"),
        ("吧", "han"),
    )


def test_punctuation_and_spaces_are_dropped() -> None:
    assert mixed_token_sequence("你好，世界！") == (
        ("你", "han"),
        ("好", "han"),
        ("世", "han"),
        ("界", "han"),
    )
    assert mixed_token_sequence("a, b.") == (("a", "latin"), ("b", "latin"))


def test_a_digit_run_is_one_token() -> None:
    assert mixed_token_sequence("room 42 ok") == (
        ("room", "latin"),
        ("42", "digit"),
        ("ok", "latin"),
    )


def test_digits_inside_han_stay_their_own_run() -> None:
    assert mixed_token_sequence("第42层") == (
        ("第", "han"),
        ("42", "digit"),
        ("层", "han"),
    )


def test_an_empty_string_has_no_tokens() -> None:
    assert mixed_token_sequence("") == ()
    assert mixed_token_sequence("   ") == ()
    assert mixed_token_sequence("，。！") == ()


def test_a_pure_script_has_no_switches() -> None:
    assert count_code_switches("the cat sat") == 0
    assert count_code_switches("今天天气") == 0


def test_each_script_transition_counts_once() -> None:
    assert count_code_switches("今天use AI好吧") == 2


def test_digits_count_as_a_script_of_their_own() -> None:
    assert count_code_switches("第42层") == 2


def test_punctuation_does_not_interrupt_a_run() -> None:
    assert count_code_switches("hello, 世界!") == 1


def test_an_empty_string_has_no_switches() -> None:
    assert count_code_switches("") == 0


PAIRS = [
    ("今天use AI好吧", "今天use AG好吧"),
    ("今天use AI好吧", "今天use AI好巴"),
    ("hello world", "hello 世界 world"),
    ("打开the document", "打开document"),
    ("第42层", "第43层楼"),
    ("pure english only", "pure english"),
    ("纯中文句子", "纯中文的句"),
]


@pytest.mark.parametrize("reference,hypothesis", PAIRS)
def test_the_parts_sum_to_the_whole(reference: str, hypothesis: str) -> None:
    report = score_mixed(reference, hypothesis)
    assert sum(score.substitutions for score in report.per_script) == (report.overall.substitutions)
    assert sum(score.deletions for score in report.per_script) == report.overall.deletions
    assert sum(score.insertions for score in report.per_script) == report.overall.insertions
    assert sum(score.errors for score in report.per_script) == report.overall.errors
    assert sum(score.reference_tokens for score in report.per_script) == (
        report.overall.reference_length
    )


@pytest.mark.parametrize("reference,hypothesis", PAIRS)
def test_each_script_rate_is_consistent_with_its_counts(reference: str, hypothesis: str) -> None:
    report = score_mixed(reference, hypothesis)
    for score in report.per_script:
        if score.error_rate is None:
            assert score.reference_tokens == 0
        else:
            assert score.error_rate == round(score.errors / score.reference_tokens, 6)


def test_scripts_appear_in_sorted_order_for_a_stable_report() -> None:
    report = score_mixed("今天use 42好吧", "今天use 43好吧")
    assert [score.script for score in report.per_script] == ["digit", "han", "latin"]


@pytest.mark.parametrize("seed", range(5))
def test_random_mixed_pairs_keep_the_sum_invariant(seed: int) -> None:
    import random

    pieces = ["hello", "world", "今天", "天气", "use", "AI", "好吧", "42"]
    rng = random.Random(seed)
    for _ in range(20):
        reference = " ".join(rng.choice(pieces) for _ in range(rng.randint(1, 6)))
        hypothesis = " ".join(rng.choice(pieces) for _ in range(rng.randint(0, 6)))
        report = score_mixed(reference, hypothesis)
        assert sum(score.errors for score in report.per_script) == report.overall.errors
        assert sum(score.reference_tokens for score in report.per_script) == (
            report.overall.reference_length
        )


def test_a_pure_english_pair_matches_plain_wer() -> None:
    assert score_mixed("the cat sat", "the cat").overall == wer("the cat sat", "the cat")


def test_a_pure_chinese_pair_matches_plain_cer() -> None:
    assert score_mixed("今天天气不错", "今天气不错").overall == cer("今天天气不错", "今天气不错")


def test_a_pure_utterance_has_one_script_score() -> None:
    report = score_mixed("the cat sat", "the cat")
    assert len(report.per_script) == 1
    assert report.per_script[0].script == "latin"
    assert report.per_script[0].error_rate == report.overall.error_rate


def test_a_pure_utterance_reports_no_switches() -> None:
    assert score_mixed("今天天气不错", "今天气不错").code_switches == 0


def _script(report, name: str):
    return next(score for score in report.per_script if score.script == name)


def test_a_latin_substitution_is_charged_to_the_latin_script() -> None:
    report = score_mixed("今天use AI好吧", "今天use AG好吧")
    assert (_script(report, "latin").substitutions, _script(report, "latin").reference_tokens) == (
        1,
        2,
    )
    assert _script(report, "han").substitutions == 0
    assert _script(report, "han").reference_tokens == 4
    assert report.overall.errors == 1
    assert report.code_switches == 2


def test_a_han_substitution_is_charged_to_the_han_script() -> None:
    report = score_mixed("今天use AI好吧", "今天use AI好八")
    assert _script(report, "han").substitutions == 1
    assert _script(report, "latin").substitutions == 0


def test_an_inserted_script_absent_from_the_reference_has_no_rate() -> None:
    report = score_mixed("hello world", "hello 世界 world")
    han = _script(report, "han")
    assert han.reference_tokens == 0
    assert han.insertions == 2
    assert han.error_rate is None
    assert report.overall.insertions == 2


def test_a_deletion_is_charged_to_the_script_it_removed() -> None:
    report = score_mixed("打开the document", "打开document")
    assert _script(report, "latin").deletions == 1
    assert _script(report, "han").errors == 0


def test_the_golden_mixed_score() -> None:
    report = score_mixed("今天use AI好吧", "今天use AG好巴")
    assert format_code_switch(report) == "CS 33.33% | han 1/4 latin 1/2 | switches 2"


def test_a_perfect_mixed_hypothesis_reads_as_zero() -> None:
    report = score_mixed("今天use AI好吧", "今天use AI好吧")
    assert format_code_switch(report) == "CS 0.00% | han 0/4 latin 0/2 | switches 2"


def test_the_empty_pair_has_no_script_parts() -> None:
    assert format_code_switch(score_mixed("", "")) == "CS 0.00% | switches 0"


def test_a_script_only_in_the_hypothesis_still_appears_in_the_summary() -> None:
    report = score_mixed("hello world", "hello 世界 world")
    assert format_code_switch(report) == "CS 100.00% | han 2/0 latin 0/2 | switches 0"


def test_an_empty_reference_with_content_is_refused() -> None:
    import pytest

    from echoscribe.errors import EvalError

    with pytest.raises(EvalError, match="the reference is empty"):
        score_mixed("", "hello 世界")


def test_punctuation_only_text_scores_as_an_empty_pair() -> None:
    report = score_mixed("，。", "！")
    assert report.overall.errors == 0
    assert report.overall.error_rate == 0.0


def test_the_eval_subpackage_re_exports_the_code_switch_api() -> None:
    from echoscribe import eval as evaluation

    assert evaluation.score_mixed is score_mixed
    assert evaluation.count_code_switches is count_code_switches
    for name in ("CodeSwitchReport", "ScriptScore", "format_code_switch", "mixed_token_sequence"):
        assert name in evaluation.__all__
