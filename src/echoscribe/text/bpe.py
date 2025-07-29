"""Byte-pair encoding, trained from scratch on the synthetic corpus.

BPE starts from single characters and repeatedly merges the most frequent adjacent
pair, recording each merge.  Encoding a word then means applying the learned merges
in rank order.  It is the simplest subword scheme that has the property an ASR
output layer actually needs: a finite alphabet that can still spell any word,
including one the training corpus never contained.

Two decisions here are worth calling out because they are where implementations
usually differ:

*words are spelled with an explicit end marker.*  A word is turned into its
characters plus ``"</w>"``, so a merge can only ever join a character to what
follows it inside the same word.  Without the marker, "the" and "then" would share
a prefix merge and the model could not tell a finished word from an unfinished one.

*the trainer takes no seed.*  Frequency ties are broken by the lexicographic order
of the pair, which makes the merge list a pure function of the word counts.  A
seeded tie-break would produce a different model per run for no benefit and make
"same corpus, same tokenizer" untestable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from echoscribe.errors import ConfigError, require, require_positive

#: Marks the end of a word inside a symbol sequence.
WORD_END = "</w>"


def word_symbols(word: str) -> tuple[str, ...]:
    """Spell ``word`` as its characters followed by :data:`WORD_END`."""
    require(
        isinstance(word, str), f"a word must be a string, got {type(word).__name__}", ConfigError
    )
    require(bool(word), "cannot spell an empty word", ConfigError)
    return tuple(word) + (WORD_END,)


def count_pairs(words: Iterable[str]) -> Counter:
    """Count adjacent symbol pairs over a (possibly repeated) word list.

    Repeats matter: a word that occurs ten times contributes ten of each of its
    pairs, which is exactly the frequency signal the merge search needs.
    """
    counts: Counter = Counter()
    for word in words:
        symbols = word_symbols(word)
        counts.update(zip(symbols, symbols[1:], strict=False))
    return counts


def merge_pair(symbols: Sequence[str], pair: tuple[str, str]) -> tuple[str, ...]:
    """Apply one merge left to right, without overlapping matches."""
    result: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            result.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            result.append(symbols[index])
            index += 1
    return tuple(result)


def learn_bpe(words: Iterable[str], num_merges: int) -> tuple[tuple[str, str], ...]:
    """Learn at most ``num_merges`` merges from ``words``.

    Each round counts adjacent pairs weighted by word frequency, takes the most
    frequent one, and breaks a tie by choosing the lexicographically smaller pair.
    The tie-break is what makes the result reproducible: without it two runs over
    the same corpus can produce different merge orders whenever two pairs are
    equally common, which is often.

    Fewer merges are returned than requested when the corpus runs out of pairs to
    merge; that is a normal end state, not an error.
    """
    require_positive("num_merges", num_merges, ConfigError)
    frequencies = Counter(words)
    splits = {word: word_symbols(word) for word in frequencies}
    merges: list[tuple[str, str]] = []
    for _ in range(int(num_merges)):
        pairs: Counter = Counter()
        for word, frequency in frequencies.items():
            symbols = splits[word]
            for pair in zip(symbols, symbols[1:], strict=False):
                pairs[pair] += frequency
        if not pairs:
            break
        best = min(pairs.items(), key=lambda item: (-item[1], item[0]))[0]
        merges.append(best)
        splits = {word: merge_pair(symbols, best) for word, symbols in splits.items()}
    return tuple(merges)
