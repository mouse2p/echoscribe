"""echoscribe: an offline workbench for streaming speech-recognition mechanics.

The package implements and verifies the *mechanics* of streaming automatic
speech recognition -- feature pipelines, tokenizers, CTC and transducer
decoding, latency-budgeted chunk scheduling, hotword biasing, endpointing and
code-switch aware scoring.  Every measurable number it reports comes from the
bundled synthetic acoustic laboratory, which renders waveforms from a seeded
formant synthesizer and carries its own planted transcripts as ground truth.
Nothing here evaluates a real acoustic system.
"""

from echoscribe.audio import read_wav, write_wav
from echoscribe.cmvn import CmvnStatistics, apply_cmvn, sliding_window_cmvn
from echoscribe.delta import DeltaConfig, stack_features
from echoscribe.features import FeatureConfig, extract_log_mel, extract_mfcc
from echoscribe.framing import FrameConfig, frame_signal
from echoscribe.resample import ResampleConfig

# Exported under a distinct name: binding `resample` here would shadow the
# `echoscribe.resample` submodule for anyone doing `from echoscribe import
# resample`, which silently changes what that expression means.
from echoscribe.resample import resample as resample_audio
from echoscribe.spectrum import log_power_spectrum, power_spectrum
from echoscribe.waveform import Audio, concatenate, silence

__version__ = "0.1.1"

__all__ = [
    "Audio",
    "CmvnStatistics",
    "DeltaConfig",
    "FeatureConfig",
    "FrameConfig",
    "ResampleConfig",
    "__version__",
    "apply_cmvn",
    "concatenate",
    "extract_log_mel",
    "extract_mfcc",
    "frame_signal",
    "log_power_spectrum",
    "power_spectrum",
    "read_wav",
    "resample_audio",
    "silence",
    "sliding_window_cmvn",
    "stack_features",
    "write_wav",
]
