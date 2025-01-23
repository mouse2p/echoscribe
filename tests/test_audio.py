"""Codec behaviour is pinned at the extremes, where off-by-ones live."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.audio import INT16_SCALE, float32_to_pcm16, pcm16_to_float32


def test_int16_scale_is_two_to_the_fifteen() -> None:
    assert INT16_SCALE == 32768.0


def test_decoding_the_most_negative_code_gives_minus_one() -> None:
    payload = (-32768).to_bytes(2, "little", signed=True)
    assert float(pcm16_to_float32(payload)[0]) == -1.0


def test_decoding_the_most_positive_code_stays_below_one() -> None:
    payload = (32767).to_bytes(2, "little", signed=True)
    value = float(pcm16_to_float32(payload)[0])
    assert value == pytest.approx(32767 / 32768, rel=1e-7)
    assert value < 1.0


def test_decoding_zero_gives_zero() -> None:
    assert float(pcm16_to_float32(b"\x00\x00")[0]) == 0.0


def test_decoding_is_little_endian() -> None:
    # 0x0102 little-endian is 258, big-endian would be 258 only by coincidence.
    assert float(pcm16_to_float32(b"\x02\x01")[0]) == pytest.approx(258 / 32768)


def test_decoding_an_empty_payload_gives_an_empty_array() -> None:
    assert pcm16_to_float32(b"").shape == (0,)


def test_decoding_rejects_an_odd_byte_count() -> None:
    with pytest.raises(ValueError, match="whole samples"):
        pcm16_to_float32(b"\x01\x02\x03")


def test_encoding_plus_one_clips_to_the_largest_code() -> None:
    payload = float32_to_pcm16(np.array([1.0], dtype=np.float32))
    assert int.from_bytes(payload, "little", signed=True) == 32767


def test_encoding_minus_one_gives_the_most_negative_code() -> None:
    payload = float32_to_pcm16(np.array([-1.0], dtype=np.float32))
    assert int.from_bytes(payload, "little", signed=True) == -32768


def test_encoding_clips_rather_than_wrapping() -> None:
    payload = float32_to_pcm16(np.array([4.0, -4.0], dtype=np.float32))
    codes = np.frombuffer(payload, dtype="<i2")
    assert list(codes) == [32767, -32768]


def test_int16_round_trip_is_exact_for_every_code() -> None:
    codes = np.arange(-32768, 32768, dtype=np.int64)
    payload = codes.astype("<i2").tobytes()
    recovered = np.frombuffer(float32_to_pcm16(pcm16_to_float32(payload)), dtype="<i2")
    assert np.array_equal(recovered.astype(np.int64), codes)
