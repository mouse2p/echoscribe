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
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import AudioError, require
from echoscribe.types import SAMPLE_DTYPE, SampleArray

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
