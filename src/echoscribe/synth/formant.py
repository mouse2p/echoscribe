"""Two-pole resonators and the phone renderer built on them.

A formant is modelled as a complex-conjugate pole pair.  For a centre frequency
``fc`` and a 3 dB bandwidth ``bw`` the poles sit at radius ``exp(-pi * bw / fs)``
and angle ``2 * pi * fc / fs``, which gives the recursion

    ``y[n] = a0 * x[n] + b1 * y[n-1] + b2 * y[n-2]``

with ``b1 = 2 r cos(theta)``, ``b2 = -r**2`` and ``a0 = 1 - b1 - b2``.  Choosing
``a0`` that way makes the resonator unity-gain at DC, so cascading three of them
does not silently change a phone's loudness.

The recursion is a plain Python loop rather than a blocked or FFT-based filter.
That is deliberate: it has no block-boundary artefacts, it starts from an exactly
zero state, and it is what a streaming synthesizer would run anyway.  It is also
the slowest part of the laboratory, which is why the corpora here are short.
"""

from __future__ import annotations

import math

import numpy as np

from echoscribe.errors import ConfigError, require, require_positive
from echoscribe.synth.phones import Phone
from echoscribe.synth.source import (
    DEFAULT_BASE_F0,
    glottal_source,
    noise_source,
    phone_envelope,
)
from echoscribe.types import FEATURE_DTYPE, FloatArray


def resonator_coefficients(
    centre_hz: float, bandwidth_hz: float, sample_rate: int
) -> tuple[float, float, float]:
    """Return ``(a0, b1, b2)`` of the two-pole resonator for one formant."""
    require_positive("centre_hz", centre_hz, ConfigError)
    require_positive("bandwidth_hz", bandwidth_hz, ConfigError)
    require_positive("sample_rate", sample_rate, ConfigError)
    require(
        centre_hz < sample_rate / 2.0,
        f"formant at {centre_hz} Hz is at or above Nyquist ({sample_rate / 2.0} Hz)",
        ConfigError,
    )
    radius = math.exp(-math.pi * bandwidth_hz / sample_rate)
    theta = 2.0 * math.pi * centre_hz / sample_rate
    b1 = 2.0 * radius * math.cos(theta)
    b2 = -(radius * radius)
    return 1.0 - b1 - b2, b1, b2


def apply_resonator(
    signal: np.ndarray, centre_hz: float, bandwidth_hz: float, sample_rate: int
) -> FloatArray:
    """Filter ``signal`` through one resonator starting from a zero state."""
    a0, b1, b2 = resonator_coefficients(centre_hz, bandwidth_hz, sample_rate)
    x = np.asarray(signal, dtype=FEATURE_DTYPE)
    require(x.ndim == 1, f"resonator input must be 1-D, got shape {x.shape}", ConfigError)
    # The recursion is inherently sequential; the only speed available without a
    # compiled filter is to stay in plain Python floats and append to a list.
    output = []
    append = output.append
    y1 = 0.0
    y2 = 0.0
    for value in x.tolist():
        y = a0 * value + b1 * y1 + b2 * y2
        append(y)
        y2 = y1
        y1 = y
    return np.asarray(output, dtype=FEATURE_DTYPE)


def render_phone(
    phone: Phone,
    sample_rate: int,
    f0: float | np.ndarray = DEFAULT_BASE_F0,
    rng: np.random.Generator | None = None,
    envelope_attack: int = 24,
    envelope_release: int = 48,
) -> FloatArray:
    """Render one phone to a waveform of its declared duration.

    The source is the glottal pulse train for a voiced phone, turbulence for a
    voiceless one, and the sum of both when a phone is voiced *and* turbulent --
    which is how a ``z`` or a ``v`` behaves.  The source passes through one
    resonator per formant and is then scaled by the phone's gain and envelope.

    Silence short-circuits to zeros: rendering it through the resonators would
    only add rounding noise to what should be exactly quiet.
    """
    require_positive("sample_rate", sample_rate, ConfigError)
    num_samples = samples_for_duration(phone.duration_ms, sample_rate)
    if num_samples <= 0 or phone.is_silent:
        return np.zeros(max(num_samples, 0), dtype=FEATURE_DTYPE)
    if np.ndim(f0) == 0:
        track: np.ndarray = np.full(num_samples, float(f0))
    else:
        track = np.asarray(f0, dtype=FEATURE_DTYPE)
    require(
        track.shape == (num_samples,),
        f"f0 track has {track.shape[0]} samples but the phone needs {num_samples}",
        ConfigError,
    )
    source = np.zeros(num_samples, dtype=FEATURE_DTYPE)
    if phone.voiced:
        source = source + glottal_source(num_samples, sample_rate, track)
    if phone.turbulent:
        require(
            rng is not None,
            f"{phone.symbol} is turbulent and needs a seeded generator",
            ConfigError,
        )
        source = source + noise_source(
            num_samples, rng, tilt_hz=_tilt_for(phone), sample_rate=sample_rate
        )
    for centre, bandwidth in phone.formants:
        source = apply_resonator(source, centre, bandwidth, sample_rate)
    return source * phone_envelope(num_samples, envelope_attack, envelope_release) * phone.gain


def samples_for_duration(duration_ms: float, sample_rate: int) -> int:
    """Whole samples covered by ``duration_ms``, rounded half up."""
    require_positive("sample_rate", sample_rate, ConfigError)
    return int(math.floor(duration_ms * sample_rate / 1000.0 + 0.5))


def _tilt_for(phone: Phone) -> float:
    """Tie the noise colour to the phone's own lowest resonance.

    A fricative's spectrum is dominated by where its constriction sits, so reusing
    the first formant as the tilt corner keeps ``s`` bright and ``sh`` dark without
    maintaining a second table.
    """
    return float(phone.formants[0][0])
