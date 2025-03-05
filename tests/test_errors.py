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


@pytest.mark.parametrize(
    ("name", "builtin"),
    [
        ("SchemaError", ValueError),
        ("CorpusError", ValueError),
        ("TokenizerError", ValueError),
        ("ModelError", RuntimeError),
    ],
)
def test_later_error_types_follow_the_same_convention(name: str, builtin: type) -> None:
    error = getattr(errors, name)
    assert issubclass(error, errors.EchoscribeError)
    assert issubclass(error, builtin)


def test_require_is_silent_when_the_condition_holds() -> None:
    assert errors.require(True, "unused") is None


def test_require_raises_the_requested_error_type() -> None:
    with pytest.raises(errors.AudioError, match="sample rate must be positive"):
        errors.require(False, "sample rate must be positive", errors.AudioError)


def test_require_positive_accepts_the_boundary_of_zero_exclusive() -> None:
    assert errors.require_positive("chunk_ms", 0.5) == 0.5
    with pytest.raises(errors.ConfigError, match="chunk_ms must be positive"):
        errors.require_positive("chunk_ms", 0.0)
    with pytest.raises(errors.ConfigError):
        errors.require_positive("chunk_ms", -1.0)


def test_require_non_negative_accepts_zero_but_not_below() -> None:
    assert errors.require_non_negative("hangover_ms", 0) == 0
    with pytest.raises(errors.ConfigError, match="hangover_ms must not be negative"):
        errors.require_non_negative("hangover_ms", -0.001)


def test_require_in_names_the_allowed_values() -> None:
    assert errors.require_in("window", "povey", ["hamming", "povey"]) == "povey"
    with pytest.raises(errors.ConfigError) as info:
        errors.require_in("window", "blackman", ["hamming", "povey"])
    assert "'hamming'" in str(info.value)
    assert "'blackman'" in str(info.value)


def test_require_range_is_closed_on_both_ends() -> None:
    assert errors.require_range("beam", 4, 1, 8) == 4
    assert errors.require_range("beam", 1, 1, 8) == 1
    assert errors.require_range("beam", 8, 1, 8) == 8
    for bad in (0, 9, -1):
        with pytest.raises(errors.ConfigError):
            errors.require_range("beam", bad, 1, 8)


@pytest.mark.parametrize(
    "helper",
    [
        lambda: errors.require_positive("x", -1.0, errors.FeatureError),
        lambda: errors.require_non_negative("x", -1.0, errors.FeatureError),
        lambda: errors.require_range("x", 99.0, 0.0, 1.0, errors.FeatureError),
        lambda: errors.require_in("x", 3, [1, 2], errors.FeatureError),
        lambda: errors.require(False, "nope", errors.FeatureError),
    ],
)
def test_helper_error_type_is_honoured(helper) -> None:
    with pytest.raises(errors.FeatureError):
        helper()


def test_require_positive_accepts_infinity_and_rejects_nan() -> None:
    # NaN compares false against everything, so it must never pass a bound.
    assert errors.require_positive("gain", float("inf")) == float("inf")
    with pytest.raises(errors.ConfigError):
        errors.require_positive("gain", float("nan"))
    with pytest.raises(errors.ConfigError):
        errors.require_non_negative("gain", float("nan"))
    with pytest.raises(errors.ConfigError):
        errors.require_range("gain", float("nan"), 0.0, 1.0)


def test_require_non_negative_returns_the_value_it_accepted() -> None:
    assert errors.require_non_negative("frames", 0) == 0
    assert errors.require_non_negative("frames", 12) == 12
    assert errors.require_non_negative("frames", 0.25) == 0.25
