"""PCM codecs and WAV file I/O built on the standard library ``wave`` module.

echoscribe deliberately does not depend on ``soundfile`` or ``librosa``: the WAV
support needed for a synthetic laboratory is small, and keeping it on ``wave``
plus NumPy means the codec behaviour is fully inspectable and testable.

The convention throughout is:

* float samples live in ``[-1.0, 1.0]`` as ``float32``;
* integer PCM is little-endian, which is what ``wave`` writes on every platform;
* decoding divides by ``2 ** (bits - 1)`` so ``-1.0`` maps to the most negative
  code and ``+1.0`` maps to one step *below* the most positive code, keeping the
  mapping symmetric and the round trip exact.

The asymmetry is inherent to two's complement: for ``n`` bits there are
``2 ** (n - 1)`` negative codes but only ``2 ** (n - 1) - 1`` positive ones.
Dividing by ``2 ** (n - 1)`` on the way out therefore can never produce exactly
``+1.0``, and clipping on the way in keeps ``+1.0`` at the largest positive code
rather than letting it wrap to the most negative one.
"""

from __future__ import annotations

import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echoscribe.errors import AudioError, require
from echoscribe.types import SAMPLE_DTYPE, SampleArray, require_rank

INT16_SCALE = float(1 << 15)


def pcm16_to_float32(payload: bytes) -> SampleArray:
    """Decode little-endian signed 16-bit PCM into float32 samples."""
    require(
        len(payload) % 2 == 0,
        f"16-bit PCM payload must hold whole samples, got {len(payload)} bytes",
        AudioError,
    )
    raw = np.frombuffer(payload, dtype="<i2")
    return (raw.astype(np.float32) / np.float32(INT16_SCALE)).astype(SAMPLE_DTYPE)


def float32_to_pcm16(samples: SampleArray) -> bytes:
    """Encode float32 samples as little-endian signed 16-bit PCM, clipping."""
    array = np.asarray(samples, dtype=np.float64)
    scaled = np.rint(np.clip(array, -1.0, 1.0) * INT16_SCALE)
    return np.clip(scaled, -INT16_SCALE, INT16_SCALE - 1).astype("<i2").tobytes()


INT8_OFFSET = 128.0
INT8_SCALE = float(1 << 7)
INT32_SCALE = float(1 << 31)


def pcm8_to_float32(payload: bytes) -> SampleArray:
    """Decode unsigned 8-bit PCM, which is centred on 128 rather than on zero."""
    raw = np.frombuffer(payload, dtype=np.uint8)
    shifted = raw.astype(np.float32) - np.float32(INT8_OFFSET)
    return (shifted / np.float32(INT8_SCALE)).astype(SAMPLE_DTYPE)


def pcm32_to_float32(payload: bytes) -> SampleArray:
    """Decode little-endian signed 32-bit PCM into float32 samples."""
    require(
        len(payload) % 4 == 0,
        f"32-bit PCM payload must hold whole samples, got {len(payload)} bytes",
        AudioError,
    )
    raw = np.frombuffer(payload, dtype="<i4")
    return (raw.astype(np.float64) / INT32_SCALE).astype(SAMPLE_DTYPE)


INT24_SCALE = float(1 << 23)


def pcm24_to_float32(payload: bytes) -> SampleArray:
    """Decode little-endian signed 24-bit PCM, which ``wave`` stores unpacked.

    There is no ``<i3`` dtype, so the three bytes are reassembled into ``int32``
    and sign-extended manually.
    """
    require(
        len(payload) % 3 == 0,
        f"24-bit PCM payload must hold whole samples, got {len(payload)} bytes",
        AudioError,
    )
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    codes = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    codes = np.where(codes >= 1 << 23, codes - (1 << 24), codes)
    return (codes.astype(np.float64) / INT24_SCALE).astype(SAMPLE_DTYPE)


#: PCM sample widths in bytes that echoscribe can decode.
SUPPORTED_WIDTHS = (1, 2, 3, 4)

_DECODERS = {
    1: pcm8_to_float32,
    2: pcm16_to_float32,
    3: pcm24_to_float32,
    4: pcm32_to_float32,
}


def decode_pcm(payload: bytes, sample_width: int) -> SampleArray:
    """Decode ``payload`` according to ``sample_width`` in bytes."""
    require(
        sample_width in SUPPORTED_WIDTHS,
        f"unsupported PCM sample width {sample_width}; echoscribe reads {SUPPORTED_WIDTHS}",
        AudioError,
    )
    return _DECODERS[sample_width](payload)


def float32_to_pcm8(samples: SampleArray) -> bytes:
    """Encode float32 samples as unsigned 8-bit PCM centred on 128."""
    array = np.asarray(samples, dtype=np.float64)
    scaled = np.rint(np.clip(array, -1.0, 1.0) * INT8_SCALE + INT8_OFFSET)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8).tobytes()


def float32_to_pcm24(samples: SampleArray) -> bytes:
    """Encode float32 samples as little-endian signed 24-bit PCM."""
    array = np.asarray(samples, dtype=np.float64)
    scaled = np.rint(np.clip(array, -1.0, 1.0) * INT24_SCALE)
    codes = np.clip(scaled, -INT24_SCALE, INT24_SCALE - 1).astype(np.int32)
    raw = (codes & 0xFFFFFF).astype(np.uint32)
    packed = np.empty((raw.size, 3), dtype=np.uint8)
    packed[:, 0] = raw & 0xFF
    packed[:, 1] = (raw >> 8) & 0xFF
    packed[:, 2] = (raw >> 16) & 0xFF
    return packed.tobytes()


def float32_to_pcm32(samples: SampleArray) -> bytes:
    """Encode float32 samples as little-endian signed 32-bit PCM."""
    array = np.asarray(samples, dtype=np.float64)
    scaled = np.rint(np.clip(array, -1.0, 1.0) * INT32_SCALE)
    return np.clip(scaled, -INT32_SCALE, INT32_SCALE - 1).astype("<i4").tobytes()


_ENCODERS = {
    1: float32_to_pcm8,
    2: float32_to_pcm16,
    3: float32_to_pcm24,
    4: float32_to_pcm32,
}


def encode_pcm(samples: SampleArray, sample_width: int) -> bytes:
    """Encode ``samples`` at the requested ``sample_width`` in bytes."""
    require(
        sample_width in SUPPORTED_WIDTHS,
        f"unsupported PCM sample width {sample_width}; echoscribe writes {SUPPORTED_WIDTHS}",
        AudioError,
    )
    return _ENCODERS[sample_width](samples)


@dataclass(frozen=True)
class WavInfo:
    """What a WAV header claims, read without decoding any samples."""

    path: str
    channels: int
    sample_rate: int
    sample_width: int
    num_frames: int
    compression: str

    @property
    def duration(self) -> float:
        """Length in seconds implied by the frame count."""
        return self.num_frames / self.sample_rate

    @property
    def is_pcm(self) -> bool:
        """True for the uncompressed encoding that ``wave`` labels ``NONE``."""
        return self.compression == "NONE"


def inspect_wav(path: str | Path) -> WavInfo:
    """Read only the header of ``path``.

    Useful before committing to a decode: a caller can reject a floating-point or
    compressed file from its metadata alone.
    """
    resolved = Path(path)
    require(resolved.is_file(), f"wav file not found: {resolved}", AudioError)
    try:
        with wave.open(str(resolved), "rb") as handle:
            return WavInfo(
                path=str(resolved),
                channels=handle.getnchannels(),
                sample_rate=handle.getframerate(),
                sample_width=handle.getsampwidth(),
                num_frames=handle.getnframes(),
                compression=handle.getcomptype(),
            )
    except wave.Error as error:
        raise AudioError(f"{resolved}: not a readable wav file ({error})") from error


def read_wav_channels(path: str | Path) -> tuple[np.ndarray, int]:
    """Read ``path`` without mixing, returning ``(frames, channels)`` and the rate.

    Only uncompressed PCM in one of :data:`SUPPORTED_WIDTHS` is accepted.
    Compressed and floating-point WAV variants raise :class:`AudioError` rather
    than being silently mis-decoded.
    """
    info = inspect_wav(path)
    require(
        info.is_pcm,
        f"{info.path}: only uncompressed PCM wav is supported, got {info.compression!r}",
        AudioError,
    )
    require(
        info.sample_width in SUPPORTED_WIDTHS,
        f"{info.path}: unsupported sample width {info.sample_width}",
        AudioError,
    )
    require(info.channels > 0, f"{info.path}: header claims {info.channels} channels", AudioError)
    try:
        with wave.open(info.path, "rb") as handle:
            payload = handle.readframes(info.num_frames)
    except wave.Error as error:
        raise AudioError(f"{info.path}: cannot read frames ({error})") from error
    flat = decode_pcm(payload, info.sample_width)
    require(
        flat.size == info.num_frames * info.channels,
        f"{info.path}: header claims {info.num_frames} frames but the payload holds "
        f"{flat.size // max(info.channels, 1)}",
        AudioError,
    )
    return flat.reshape(info.num_frames, info.channels), info.sample_rate


def to_mono(channels: np.ndarray, weights: Sequence[float] | None = None) -> SampleArray:
    """Mix a ``(frames, channels)`` matrix down to one channel.

    Without ``weights`` the channels are averaged, which is what a mono ASR front
    end wants.  With ``weights`` the mix is divided by their sum, so passing
    ``[0.0, 1.0]`` *selects* the right channel instead of halving it.
    """
    matrix = require_rank("channel matrix", np.asarray(channels, dtype=np.float64), 2)
    if weights is None:
        return matrix.mean(axis=1).astype(SAMPLE_DTYPE)
    gains = np.asarray(weights, dtype=np.float64)
    require(
        gains.shape == (matrix.shape[1],),
        f"expected {matrix.shape[1]} channel weights, got shape {gains.shape}",
        AudioError,
    )
    total = float(gains.sum())
    require(abs(total) > 1e-12, "channel weights must not sum to zero", AudioError)
    return ((matrix @ gains) / total).astype(SAMPLE_DTYPE)
