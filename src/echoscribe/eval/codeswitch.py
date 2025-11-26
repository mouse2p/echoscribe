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
