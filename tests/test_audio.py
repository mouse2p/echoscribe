"""Codec behaviour is pinned at the extremes, where off-by-ones live."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from echoscribe.audio import (
    INT16_SCALE,
    SUPPORTED_WIDTHS,
    decode_pcm,
    encode_pcm,
    float32_to_pcm8,
    float32_to_pcm16,
    float32_to_pcm24,
    float32_to_pcm32,
    inspect_wav,
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


def test_decode_pcm_routes_to_the_matching_codec() -> None:
    assert float(decode_pcm(b"\x80", 1)[0]) == 0.0
    assert float(decode_pcm(b"\x00\x00", 2)[0]) == 0.0
    assert float(decode_pcm(b"\x00\x00\x00", 3)[0]) == 0.0
    assert float(decode_pcm(b"\x00\x00\x00\x00", 4)[0]) == 0.0


@pytest.mark.parametrize("width", [0, 5, -2, 6])
def test_decode_pcm_rejects_an_unsupported_width(width: int) -> None:
    with pytest.raises(ValueError, match="unsupported PCM sample width"):
        decode_pcm(b"\x00" * 8, width)


def test_supported_widths_cover_every_registered_codec() -> None:
    assert tuple(sorted(SUPPORTED_WIDTHS)) == (1, 2, 3, 4)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_every_width_round_trips_within_one_quantisation_step(width: int) -> None:
    rng = np.random.default_rng(5)
    # Feed the codec a float32 waveform, which is how Audio stores one.  Feeding
    # float64 instead makes 32-bit PCM look lossy: the extra mantissa bits land
    # within one float32 ulp of a rounding midpoint and the cast can flip a code.
    wave = rng.uniform(-1.0, 1.0, 256).astype(np.float32)
    quantum = 2.0 / (1 << (width * 8 - 1))
    recovered = decode_pcm(encode_pcm(wave, width), width)
    error = np.abs(recovered.astype(np.float64) - wave.astype(np.float64))
    assert np.max(error) <= quantum


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_every_width_clips_out_of_range_input(width: int) -> None:
    payload = encode_pcm(np.array([9.0, -9.0]), width)
    recovered = decode_pcm(payload, width)
    assert recovered[0] > 0.9
    assert recovered[1] == -1.0


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_every_width_encodes_silence_to_its_zero_code(width: int) -> None:
    payload = encode_pcm(np.zeros(4), width)
    assert len(payload) == 4 * width
    assert np.all(decode_pcm(payload, width) == 0.0)


def test_encode_pcm_rejects_an_unsupported_width() -> None:
    with pytest.raises(ValueError, match="unsupported PCM sample width"):
        encode_pcm(np.zeros(2), 5)


def test_8bit_encoder_centres_silence_on_128() -> None:
    assert list(float32_to_pcm8(np.zeros(3))) == [128, 128, 128]


def test_24bit_encoder_packs_three_bytes_per_sample() -> None:
    assert len(float32_to_pcm24(np.array([0.0, 0.5, -0.5]))) == 9


def test_32bit_encoder_holds_the_full_range() -> None:
    payload = float32_to_pcm32(np.array([1.0, -1.0]))
    codes = np.frombuffer(payload, dtype="<i4")
    assert codes[0] == 2147483647
    assert codes[1] == -2147483648


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_encoding_is_monotonic_in_the_input(width: int) -> None:
    ramp = np.linspace(-1.0, 1.0, 512)
    recovered = decode_pcm(encode_pcm(ramp, width), width).astype(np.float64)
    assert np.all(np.diff(recovered) >= 0.0)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_encode_decode_is_idempotent_on_reconstructed_audio(width: int) -> None:
    rng = np.random.default_rng(9)
    wave = rng.uniform(-0.8, 0.8, 128).astype(np.float32)
    first = encode_pcm(wave, width)
    second = encode_pcm(decode_pcm(first, width), width)
    assert first == second


def test_decoders_all_return_float32() -> None:
    for width in SUPPORTED_WIDTHS:
        assert decode_pcm(b"\x00" * (width * 3), width).dtype == np.float32


def _write_raw_wav(path, payload: bytes, width: int, channels: int, rate: int) -> None:
    """Write a wav with an explicit PCM payload, bypassing the encoders."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(payload)


def test_inspect_wav_reports_the_header(tmp_path) -> None:
    target = tmp_path / "tone.wav"
    _write_raw_wav(target, b"\x00\x00" * 100, 2, 1, 16000)
    info = inspect_wav(target)
    assert info.channels == 1
    assert info.sample_rate == 16000
    assert info.sample_width == 2
    assert info.num_frames == 100
    assert info.compression == "NONE"
    assert info.is_pcm
    assert info.duration == pytest.approx(100 / 16000)
    assert info.path.endswith("tone.wav")


def test_inspect_wav_rejects_a_missing_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="wav file not found"):
        inspect_wav(tmp_path / "absent.wav")


def test_inspect_wav_rejects_a_non_wav_file(tmp_path) -> None:
    target = tmp_path / "broken.wav"
    target.write_bytes(b"not a riff container at all")
    with pytest.raises(ValueError, match="not a readable wav file"):
        inspect_wav(target)
