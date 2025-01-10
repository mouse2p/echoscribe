"""Exception types shared by every echoscribe layer.

The hierarchy is deliberately shallow.  :class:`EchoscribeError` derives from
``Exception`` so a caller can catch the whole toolkit in one clause, while each
subclass also derives from the builtin that matches its semantics: ``ValueError``
for configuration or data that is wrong on arrival, ``RuntimeError`` for a state
that became illegal while running.  Generic handlers therefore keep working
without knowing anything about echoscribe.
"""

from __future__ import annotations


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
