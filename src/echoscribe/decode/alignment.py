"""Lattice-free alignment: decoded tokens grouped into words with time spans.

A decoder emits modelling units -- phones for an English word, toned syllable units
for a Chinese character.  Everything a person reads a transcript for needs *words*,
and the only way to get from one to the other without a search lattice is to carry
the unit counts alongside the hypothesis: the tokenizer already knows that
``hello`` is four units, so the alignment is a prefix sum rather than an inference.

That is a deliberate limitation.  A lattice-free alignment cannot recover from a
decoder that emitted the wrong *number* of units, and it does not try: a mismatch
between the unit counts and the decoded tokens is an error, not something to paper
over with a heuristic.  When the counts do match, every word gets the exact frame
range its units occupied, which is what a word-level timestamp, a per-word
confidence and a latency measurement all need.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from echoscribe.decode.confidence import geometric_mean
from echoscribe.decode.tokens import DecodedToken
from echoscribe.errors import DecodeError, require


@dataclass(frozen=True)
class WordSpan:
    """One word: the frames it occupied, the tokens behind it, and a confidence."""

    word: str
    start: int
    end: int
    tokens: tuple[DecodedToken, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        require(
            0 <= self.start <= self.end,
            f"word {self.word!r} has an invalid frame span ({self.start}, {self.end})",
            DecodeError,
        )
        require(
            len(self.tokens) == 0 or self.tokens[0].start >= self.start,
            f"word {self.word!r} starts before its first token",
            DecodeError,
        )
        require(
            0.0 <= self.confidence <= 1.0,
            f"a word confidence must lie in [0, 1], got {self.confidence!r}",
            DecodeError,
        )

    @property
    def num_tokens(self) -> int:
        """How many modelling units this word was decoded from."""
        return len(self.tokens)

    @property
    def num_frames(self) -> int:
        """Frames the word spans."""
        return self.end - self.start

    def times(self, frame_shift_ms: float = 10.0) -> tuple[float, float]:
        """Start and end time in seconds."""
        shift = frame_shift_ms / 1000.0
        return self.start * shift, self.end * shift


def align_words(
    words: Sequence[str],
    token_counts: Sequence[int],
    tokens: Sequence[DecodedToken],
    confidences: Sequence[float] = (),
) -> tuple[WordSpan, ...]:
    """Group consecutive decoded tokens into words.

    ``token_counts[i]`` is how many modelling units word ``i`` is spelled with, and
    the counts have to add up to ``len(tokens)`` exactly.  A word with a count of
    zero -- one the decoder produced no units for -- gets a zero-width span at the
    cursor rather than being dropped, so the word list and the span list stay the
    same length and a scorer can still count it as a deletion.
    """
    require(
        len(words) == len(token_counts),
        f"{len(words)} words cannot carry {len(token_counts)} unit counts",
        DecodeError,
    )
    require(
        all(count >= 0 for count in token_counts),
        f"a word cannot be spelled with a negative number of units: {tuple(token_counts)}",
        DecodeError,
    )
    require(
        sum(token_counts) == len(tokens),
        f"the unit counts add to {sum(token_counts)} but {len(tokens)} tokens were decoded",
        DecodeError,
    )
    require(
        len(confidences) in (0, len(tokens)),
        f"{len(confidences)} confidences cannot describe {len(tokens)} tokens",
        DecodeError,
    )
    spans: list[WordSpan] = []
    cursor = 0
    boundary = 0
    for word, count in zip(words, token_counts, strict=True):
        group = tuple(tokens[cursor : cursor + count])
        scores = tuple(confidences[cursor : cursor + count]) if confidences else ()
        start = group[0].start if group else boundary
        end = group[-1].end if group else start
        spans.append(
            WordSpan(
                word=word,
                start=start,
                end=end,
                tokens=group,
                confidence=geometric_mean(scores) if scores else 0.0 if not group else 1.0,
            )
        )
        cursor += count
        boundary = end
    return tuple(spans)
