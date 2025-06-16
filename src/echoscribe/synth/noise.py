"""Background noise, SNR control and the mixing that puts them together.

The laboratory needs noise for two reasons: to check that a feature pipeline
degrades the way the theory says it should, and to give an endpointer something
to reject between utterances.  Both need noise whose *statistics* are known, which
is why the generators here are specified by their spectrum rather than by a
recording.

Three kinds are available:

``white``
    flat power spectral density.  The reference case.
``pink``
    power falling at 3 dB per octave (amplitude at ``1 / sqrt(f)``), which is much
    closer to ambient room and microphone noise than white is.
``modulated``
    white noise under a slow random envelope, standing in for a busy background.
    This is *not* babble: there is no speech in it, only amplitude fluctuation,
    and calling it babble would overstate what an endpointer is being tested
    against.

Every generator is peak-normalised to one and driven by a caller-supplied
generator, so a noise condition is reproducible from a seed.
"""

from __future__ import annotations

import numpy as np

from echoscribe.errors import ConfigError, require, require_in, require_positive
from echoscribe.types import FEATURE_DTYPE, FloatArray
from echoscribe.waveform import Audio

#: Sample rate the generators assume when none is given.
RATE_DEFAULT = 16000

#: Background noise kinds the laboratory can generate.
NOISE_KINDS = ("white", "pink", "modulated")


def white_noise(num_samples: int, rng: np.random.Generator) -> FloatArray:
    """Peak-normalised Gaussian white noise."""
    require_non_negative_count(num_samples)
    if num_samples == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    return _peak_normalise(rng.standard_normal(num_samples))


def pink_noise(num_samples: int, rng: np.random.Generator) -> FloatArray:
    """Peak-normalised noise with a ``1 / sqrt(f)`` amplitude spectrum.

    Shaped in the frequency domain rather than by cascading filters, because the
    target spectrum is then exact and the result does not depend on a warm-up
    transient at the start of the buffer.
    """
    require_non_negative_count(num_samples)
    if num_samples == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    bins = num_samples // 2 + 1
    shaped = rng.standard_normal(bins) + 1j * rng.standard_normal(bins)
    index = np.arange(bins, dtype=FEATURE_DTYPE)
    shaped = shaped / np.sqrt(np.maximum(index, 1.0))
    shaped[0] = 0.0  # no DC: a pink floor should not offset the waveform
    return _peak_normalise(np.fft.irfft(shaped, n=num_samples))


def require_non_negative_count(num_samples: int) -> None:
    """Shared guard for every generator's length argument."""
    require(
        isinstance(num_samples, int) and not isinstance(num_samples, bool) and num_samples >= 0,
        f"num_samples must be a non-negative integer, got {num_samples!r}",
        ConfigError,
    )


def _peak_normalise(values: np.ndarray) -> FloatArray:
    """Scale so the largest absolute sample is exactly one."""
    peak = float(np.max(np.abs(values)))
    return values / peak if peak > 1e-12 else np.asarray(values, dtype=FEATURE_DTYPE)


def signal_to_noise_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measured SNR in dB between a clean signal and its noisy version.

    The noise is taken as the *difference*, not as a separately recorded track, so
    this also measures whatever a denoiser left behind.  A difference of exactly
    zero returns ``inf`` rather than dividing by zero.
    """
    reference = np.asarray(clean, dtype=FEATURE_DTYPE)
    measured = np.asarray(noisy, dtype=FEATURE_DTYPE)
    require(
        reference.shape == measured.shape,
        f"cannot compare shapes {reference.shape} and {measured.shape}",
        ConfigError,
    )
    require(reference.size > 0, "SNR of an empty signal is undefined", ConfigError)
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean((measured - reference) ** 2))
    if noise_power <= 1e-24:
        return float("inf")
    if signal_power <= 1e-24:
        return float("-inf")
    return 10.0 * float(np.log10(signal_power / noise_power))


def mix_at_snr(
    clean: Audio, noise: np.ndarray, snr_db: float, rng: np.random.Generator | None = None
) -> Audio:
    """Return ``clean`` with ``noise`` added at the requested SNR in dB.

    The noise is rescaled, never the signal, so the clean part keeps its original
    level and a measured SNR on the result matches the request.  Noise shorter than
    the signal is tiled; longer noise is truncated.  ``rng`` is unused by the mix
    itself and exists so a caller can thread one generator through a whole corpus.
    """
    require(clean.num_frames > 0, "cannot add noise to an empty signal", ConfigError)
    track = np.asarray(noise, dtype=FEATURE_DTYPE)
    require(track.size > 0, "cannot mix with an empty noise track", ConfigError)
    repeats = -(-clean.num_frames // track.size)
    tiled = np.tile(track, repeats)[: clean.num_frames]
    signal_power = float(np.mean(clean.samples.astype(FEATURE_DTYPE) ** 2))
    noise_power = float(np.mean(tiled**2))
    require(noise_power > 1e-24, "the noise track is silent, so any SNR is infinite", ConfigError)
    target = signal_power / (noise_power * 10.0 ** (snr_db / 10.0))
    mixed = clean.samples.astype(FEATURE_DTYPE) + tiled * float(np.sqrt(target))
    return Audio(mixed.astype(np.float32), clean.sample_rate)


def moving_average(values: np.ndarray, width: int) -> FloatArray:
    """A causal-free centred moving average, edge-truncated to the input length.

    Used to build the slow envelope of :func:`modulated_noise`; ``cumsum``
    differences make it linear in the signal length rather than quadratic in the
    window.
    """
    require_positive("width", width, ConfigError)
    data = np.asarray(values, dtype=FEATURE_DTYPE)
    require(data.ndim == 1, f"moving_average needs 1-D input, got shape {data.shape}", ConfigError)
    if data.size == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    span = min(int(width), data.size)
    # Edge padding rather than zeros, so a constant input stays constant all the
    # way to its first and last sample instead of sagging at the boundaries.
    padded = np.pad(data, (span // 2, span - 1 - span // 2), mode="edge")
    running = np.concatenate([[0.0], np.cumsum(padded)])
    return (running[span:] - running[:-span]) / span


def modulated_noise(
    num_samples: int,
    rng: np.random.Generator,
    modulation_hz: float = 3.5,
    sample_rate: int = RATE_DEFAULT,
) -> FloatArray:
    """White noise under a slow random envelope.

    The envelope is a smoothed rectified noise, normalised to a mean of one, so the
    carrier's spectrum is unchanged while its energy comes and goes a few times a
    second.  That is the property an energy-based endpointer is most likely to be
    fooled by, which is why this generator exists.
    """
    require_non_negative_count(num_samples)
    require_positive("modulation_hz", modulation_hz, ConfigError)
    require_positive("sample_rate", sample_rate, ConfigError)
    if num_samples == 0:
        return np.zeros(0, dtype=FEATURE_DTYPE)
    carrier = rng.standard_normal(num_samples)
    spacing = sample_rate / modulation_hz
    nodes = max(3, int(num_samples / spacing) + 2)
    levels = rng.uniform(0.25, 1.75, size=nodes)
    positions = np.linspace(0.0, num_samples - 1, nodes)
    envelope = np.interp(np.arange(num_samples, dtype=FEATURE_DTYPE), positions, levels)
    # A light pass removes the interpolation kinks without flattening the envelope:
    # the window is a quarter of the node spacing, so the fluctuation survives.
    envelope = moving_average(envelope, max(2, int(spacing) // 4))
    mean = float(envelope.mean())
    if mean <= 1e-12:
        return _peak_normalise(carrier)
    return _peak_normalise(carrier * (envelope / mean))


def generate_noise(
    kind: str, num_samples: int, rng: np.random.Generator, sample_rate: int = RATE_DEFAULT
) -> FloatArray:
    """Dispatch on a noise kind so a config can name one string."""
    require_in("kind", kind, NOISE_KINDS, ConfigError)
    if kind == "white":
        return white_noise(num_samples, rng)
    if kind == "pink":
        return pink_noise(num_samples, rng)
    return modulated_noise(num_samples, rng, sample_rate=sample_rate)
