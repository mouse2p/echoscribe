"""echoscribe: an offline workbench for streaming speech-recognition mechanics.

The package implements and verifies the *mechanics* of streaming automatic
speech recognition -- feature pipelines, tokenizers, CTC and transducer
decoding, latency-budgeted chunk scheduling, hotword biasing, endpointing and
code-switch aware scoring.  Every measurable number it reports comes from the
bundled synthetic acoustic laboratory, which renders waveforms from a seeded
formant synthesizer and carries its own planted transcripts as ground truth.
Nothing here evaluates a real acoustic system.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
