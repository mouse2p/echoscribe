"""Codec behaviour is pinned at the extremes, where off-by-ones live."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.audio import (
    INT16_SCALE,
    float32_to_pcm16,
    pcm8_to_float32,
    pcm16_to_float32,
    pcm24_to_float32,
    pcm32_to_float32,
)


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


def test_encoding_rounds_half_way_values_away_from_zero() -> None:
    # np.rint uses banker's rounding, so pin what actually happens rather than
    # assume: 0.5 codes -> 0, 1.5 codes -> 2.
    quantum = 1.0 / INT16_SCALE
    values = np.array([0.5 * quantum, 1.5 * quantum], dtype=np.float64)
    codes = np.frombuffer(float32_to_pcm16(values), "<i2")
    assert list(codes) == [0, 2]


def test_encoding_a_sine_stays_within_one_quantisation_step() -> None:
    t = np.arange(1000, dtype=np.float64) / 16000.0
    wave = np.sin(2.0 * np.pi * 440.0 * t)
    recovered = pcm16_to_float32(float32_to_pcm16(wave.astype(np.float32)))
    assert np.max(np.abs(recovered - wave)) <= 1.0 / INT16_SCALE + 1e-9


def test_encoding_accepts_a_python_list() -> None:
    assert len(float32_to_pcm16([0.0, 0.5])) == 4


def test_decoding_then_encoding_is_stable_on_repeated_passes() -> None:
    payload = np.arange(-1000, 1000, 7, dtype=np.int64).astype("<i2").tobytes()
    once = float32_to_pcm16(pcm16_to_float32(payload))
    twice = float32_to_pcm16(pcm16_to_float32(once))
    assert once == twice


def test_8bit_zero_code_maps_to_minus_one() -> None:
    assert float(pcm8_to_float32(b"\x00")[0]) == -1.0


def test_8bit_centre_code_maps_to_zero() -> None:
    assert float(pcm8_to_float32(bytes([128]))[0]) == 0.0


def test_8bit_top_code_stays_below_one() -> None:
    value = float(pcm8_to_float32(bytes([255]))[0])
    assert value == pytest.approx(127 / 128, rel=1e-7)
    assert value < 1.0


def test_8bit_decoding_accepts_an_empty_payload() -> None:
    assert pcm8_to_float32(b"").shape == (0,)


def test_32bit_extremes() -> None:
    assert float(pcm32_to_float32((-2147483648).to_bytes(4, "little", signed=True))[0]) == -1.0
    # 2147483647 / 2147483648 sits one float32 ulp below 1.0, so the cast rounds
    # it up: the top 32-bit code is indistinguishable from full scale once the
    # waveform is held as float32.  Pinned because it is a real precision limit,
    # not a codec bug.
    top = float(pcm32_to_float32((2147483647).to_bytes(4, "little", signed=True))[0])
    assert top == 1.0


def test_32bit_decoding_rejects_a_truncated_payload() -> None:
    with pytest.raises(ValueError, match="whole samples"):
        pcm32_to_float32(b"\x01\x02\x03")


def test_24bit_extremes() -> None:
    assert float(pcm24_to_float32(b"\x00\x00\x80")[0]) == -1.0
    top = float(pcm24_to_float32(b"\xff\xff\x7f")[0])
    assert top < 1.0
    assert top == pytest.approx(1.0, abs=1e-6)


def test_24bit_zero_code_maps_to_zero() -> None:
    assert float(pcm24_to_float32(b"\x00\x00\x00")[0]) == 0.0


def test_24bit_decoding_is_little_endian() -> None:
    # 0x000001 in the low byte is code 1, not code 65536.
    assert float(pcm24_to_float32(b"\x01\x00\x00")[0]) == pytest.approx(1.0 / (1 << 23))


def test_24bit_decoding_rejects_a_truncated_payload() -> None:
    with pytest.raises(ValueError, match="whole samples"):
        pcm24_to_float32(b"\x01\x02")


def test_24bit_decoding_handles_multiple_samples() -> None:
    payload = b"\x00\x00\x00" + b"\x00\x00\x80" + b"\xff\xff\x7f"
    result = pcm24_to_float32(payload)
    assert result.shape == (3,)
    assert result[0] == 0.0
    assert result[1] == -1.0
