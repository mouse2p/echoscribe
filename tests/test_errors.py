"""The exception hierarchy keeps builtin semantics usable for generic callers."""

from __future__ import annotations

import pytest

from echoscribe import errors


def test_every_error_derives_from_the_toolkit_base() -> None:
    for name in errors.__all__:
        if name == "EchoscribeError":
            continue
        assert issubclass(getattr(errors, name), errors.EchoscribeError), name


@pytest.mark.parametrize(
    ("name", "builtin"),
    [
        ("ConfigError", ValueError),
        ("AudioError", ValueError),
        ("FeatureError", ValueError),
        ("EvalError", ValueError),
        ("DecodeError", RuntimeError),
        ("StreamError", RuntimeError),
    ],
)
def test_each_error_keeps_a_matching_builtin_base(name: str, builtin: type) -> None:
    assert issubclass(getattr(errors, name), builtin)


def test_base_error_is_not_a_value_error() -> None:
    # Catching ValueError must not silently swallow pure runtime failures.
    assert not issubclass(errors.EchoscribeError, ValueError)
    assert not issubclass(errors.DecodeError, ValueError)


def test_message_is_passed_through_unchanged() -> None:
    with pytest.raises(errors.ConfigError, match="chunk size must be positive"):
        raise errors.ConfigError("chunk size must be positive")


def test_value_error_handler_catches_configuration_failures() -> None:
    with pytest.raises(ValueError):
        raise errors.AudioError("sample rate 0 is not supported")


def test_all_lists_every_public_error_exactly_once() -> None:
    assert len(errors.__all__) == len(set(errors.__all__))
    declared = set(errors.__all__)
    public = {
        name
        for name in vars(errors)
        if isinstance(vars(errors)[name], type)
        and issubclass(vars(errors)[name], Exception)
        and name not in {"Exception", "ValueError", "RuntimeError"}
    }
    assert declared == public


def test_errors_can_be_chained_without_losing_context() -> None:
    try:
        try:
            raise KeyError("num_mel_bins")
        except KeyError as missing:
            raise errors.ConfigError("mel bins are not configured") from missing
    except errors.ConfigError as wrapped:
        assert isinstance(wrapped.__cause__, KeyError)
        assert str(wrapped) == "mel bins are not configured"
