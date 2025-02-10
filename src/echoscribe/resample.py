"""Sample-rate conversion by polyphase filtering and by linear interpolation.

Two methods are offered because they have genuinely different trade-offs, and
the synthetic laboratory needs both:

``polyphase``
    A windowed-sinc FIR applied through the polyphase decomposition of the exact
    rational ratio ``target / source``.  Band-limited, so a tone keeps its
    frequency and no aliasing is introduced, at the cost of designing a filter
    and accepting pass-band ripple that depends on the window.

``linear``
    Interpolation between the two nearest input samples.  Cheap, endpoint
    preserving and free of ringing, but it applies no anti-alias filter, so
    energy above the target Nyquist folds back into the result.  It exists so a
    caller can measure what is lost by not filtering.

Both are pure NumPy and both are deterministic for a given input.
"""

from __future__ import annotations

from math import gcd

from echoscribe.errors import ConfigError, require


def rational_factors(source_rate: int, target_rate: int) -> tuple[int, int]:
    """Reduce ``target / source`` to the coprime pair ``(up, down)``.

    ``44100 -> 16000`` gives ``(160, 441)``; ``16000 -> 8000`` gives ``(1, 2)``.
    The polyphase decomposition indexes by this pair, so it has to be exact
    rather than a float ratio.
    """
    for name, rate in (("source_rate", source_rate), ("target_rate", target_rate)):
        require(
            isinstance(rate, int) and not isinstance(rate, bool) and rate > 0,
            f"{name} must be a positive integer, got {rate!r}",
            ConfigError,
        )
    divisor = gcd(int(source_rate), int(target_rate))
    return int(target_rate) // divisor, int(source_rate) // divisor
