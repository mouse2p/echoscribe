"""Exception types shared by every echoscribe layer.

The hierarchy is deliberately shallow.  :class:`EchoscribeError` derives from
``Exception`` so a caller can catch the whole toolkit in one clause, while each
subclass also derives from the builtin that matches its semantics: ``ValueError``
for configuration or data that is wrong on arrival, ``RuntimeError`` for a state
that became illegal while running.  Generic handlers therefore keep working
without knowing anything about echoscribe.
"""

from __future__ import annotations

from collections.abc import Sequence


class EchoscribeError(Exception):
    """Base class for every error raised by echoscribe."""


class ConfigError(EchoscribeError, ValueError):
    """A configuration value is missing, malformed or self-contradictory."""


class AudioError(EchoscribeError, ValueError):
    """A waveform, sample rate or PCM payload cannot be used as given."""


class FeatureError(EchoscribeError, ValueError):
    """Feature extraction was given audio or settings it cannot honour."""


class DecodeError(EchoscribeError, RuntimeError):
    """A decoder was asked to consume posteriors it cannot decode."""


class StreamError(EchoscribeError, RuntimeError):
    """The streaming runtime saw an illegal chunk or state transition."""


class EvalError(EchoscribeError, ValueError):
    """Scoring received references or hypotheses it cannot align."""


class SchemaError(EchoscribeError, ValueError):
    """A serialized document does not match the schema version it claims."""


class CorpusError(EchoscribeError, ValueError):
    """A synthetic corpus manifest or rendered asset is inconsistent."""


class TokenizerError(EchoscribeError, ValueError):
    """Text could not be mapped onto the configured modelling units."""


class ModelError(EchoscribeError, RuntimeError):
    """A torch model bundle is missing, unsupported or structurally invalid."""


__all__ = [
    "AudioError",
    "ConfigError",
    "CorpusError",
    "DecodeError",
    "EchoscribeError",
    "EvalError",
    "FeatureError",
    "ModelError",
    "SchemaError",
    "StreamError",
    "TokenizerError",
]


ErrorType = type[EchoscribeError]


def require(condition: bool, message: str, error: ErrorType = ConfigError) -> None:
    """Raise ``error(message)`` unless ``condition`` holds.

    This is the toolkit-wide replacement for bare ``assert``: assertions are
    stripped under ``python -O``, but input validation must survive that.
    """
    if not condition:
        raise error(message)


def require_positive(name: str, value: float, error: ErrorType = ConfigError) -> float:
    """Return ``value`` when it is finite and strictly greater than zero."""
    require(value > 0, f"{name} must be positive, got {value!r}", error)
    return value


def require_non_negative(name: str, value: float, error: ErrorType = ConfigError) -> float:
    """Return ``value`` when it is finite and not below zero."""
    require(value >= 0, f"{name} must not be negative, got {value!r}", error)
    return value


def require_in(
    name: str,
    value: object,
    allowed: Sequence[object],
    error: ErrorType = ConfigError,
) -> object:
    """Return ``value`` when it is one of ``allowed``, naming them in the message."""
    require(
        value in allowed,
        f"{name} must be one of {tuple(allowed)!r}, got {value!r}",
        error,
    )
    return value


def require_range(
    name: str,
    value: float,
    low: float,
    high: float,
    error: ErrorType = ConfigError,
) -> float:
    """Return ``value`` when ``low <= value <= high`` on a closed interval."""
    require(low <= value <= high, f"{name} must lie in [{low}, {high}], got {value!r}", error)
    return value
