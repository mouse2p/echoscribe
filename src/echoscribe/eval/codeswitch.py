"""Hybrid scoring for utterances that mix English and Chinese.

A code-switched utterance has no single natural token: English is scored in
words, Chinese in characters, and a sentence can contain both.  The convention
here is one *mixed token stream* -- han characters become individual tokens,
latin and digit runs become whitespace-separated words -- aligned once as a
whole, with every error attributed to the script it landed in.

Aligning the mixed stream once, rather than aligning each script segment
separately, is a deliberate choice.  Segment-wise alignment needs the two
sides to agree on where the segments are, and a recognizer that inserts an
English word into a Chinese sentence changes exactly that.  A single global
alignment always exists, and it keeps the parts consistent with the whole:
the per-script error counts sum to the overall count by construction, which
segment-wise scores cannot promise.
"""

from __future__ import annotations

from dataclasses import dataclass

from echoscribe.errors import EvalError
from echoscribe.eval.wer import ERROR_RATE_DECIMALS, ScoreReport, score_tokens
from echoscribe.text.script import script_runs

UNSCORED_SCRIPTS = ("space", "punctuation")


def mixed_token_sequence(text: str) -> tuple[tuple[str, str], ...]:
    """Split text into ``(token, script)`` pairs for scoring.

    Han runs contribute one token per character; every other scored run
    contributes whitespace-separated words, so a digit run like ``42`` is a
    single token.  Spaces and punctuation carry no scoreable content and are
    dropped, which matches the segmentation convention elsewhere in the
    toolkit.
    """
    tokens: list[tuple[str, str]] = []
    for run in script_runs(text, merge_space=True):
        if run.script in UNSCORED_SCRIPTS:
            continue
        if run.script == "han":
            for character in run.text:
                if not character.isspace():
                    tokens.append((character, "han"))
        else:
            for word in run.text.split():
                tokens.append((word, run.script))
    return tuple(tokens)


def count_code_switches(text: str) -> int:
    """How many times the script changes between consecutive scored tokens.

    Digits count as a script of their own here, so ``第42层`` has two switches.
    The count describes the *reference*, because that is the side the score is
    normalised against; a hypothesis that switches in different places is
    already paying for it in insertions and deletions.
    """
    scripts = [script for _, script in mixed_token_sequence(text)]
    return sum(1 for before, after in zip(scripts, scripts[1:], strict=False) if before != after)


@dataclass(frozen=True)
class ScriptScore:
    """How one script fared inside a mixed utterance."""

    script: str
    reference_tokens: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        """Errors attributed to this script."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float | None:
        """This script's rate, or ``None`` when it has no reference tokens.

        A script the reference never used has no denominator.  Reporting 0.0
        would claim perfection that was never tested, and reporting 1.0 would
        claim a failure that was never possible, so the honest answer is that
        the rate does not exist.
        """
        if self.reference_tokens == 0:
            return None
        return round(self.errors / self.reference_tokens, ERROR_RATE_DECIMALS)


@dataclass(frozen=True)
class CodeSwitchReport:
    """A mixed utterance scored as a whole and per script."""

    overall: ScoreReport
    per_script: tuple[ScriptScore, ...]
    code_switches: int


def score_mixed(reference: str, hypothesis: str) -> CodeSwitchReport:
    """Score a possibly code-switched pair over one global alignment.

    Substitutions and deletions are attributed to the script of the reference
    token they consumed; insertions are attributed to the script of the
    inserted token itself.  Every error lands in exactly one script, so the
    per-script counts always sum back to the overall breakdown.
    """
    ref_pairs = mixed_token_sequence(reference)
    hyp_pairs = mixed_token_sequence(hypothesis)
    overall = score_tokens([token for token, _ in ref_pairs], [token for token, _ in hyp_pairs])
    scripts = sorted({script for _, script in ref_pairs} | {script for _, script in hyp_pairs})
    refs = dict.fromkeys(scripts, 0)
    subs = dict.fromkeys(scripts, 0)
    dels = dict.fromkeys(scripts, 0)
    ins = dict.fromkeys(scripts, 0)
    for _, script in ref_pairs:
        refs[script] += 1
    for op in overall.alignment.operations:
        if op.kind == "correct":
            continue
        if op.kind == "insertion":
            if op.hyp_index is None:
                raise EvalError("an insertion operation carries no hypothesis index")
            ins[hyp_pairs[op.hyp_index][1]] += 1
            continue
        if op.ref_index is None:
            raise EvalError(f"a {op.kind} operation carries no reference index")
        if op.kind == "deletion":
            dels[ref_pairs[op.ref_index][1]] += 1
        else:
            subs[ref_pairs[op.ref_index][1]] += 1
    per_script = tuple(
        ScriptScore(script, refs[script], subs[script], dels[script], ins[script])
        for script in scripts
    )
    return CodeSwitchReport(
        overall=overall,
        per_script=per_script,
        code_switches=count_code_switches(reference),
    )
