# Usage

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The optional torch extra (CPU wheels are sufficient; nothing here uses a GPU)
powers the trainable demos:

```bash
.venv/bin/pip install -e ".[dev,torch]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

NumPy is the only required runtime dependency.

## The command line

One work directory carries a run end to end; every stage reads what the
previous stage wrote there.

```bash
echoscribe synth  --workdir run --utterances 24 --seed 2025   # wav/ + manifest.jsonl
echoscribe feats  --workdir run --num-mel-bins 40             # features.npz
echoscribe decode --workdir run --method greedy               # decode.jsonl
echoscribe stream --workdir run --frames-per-chunk 16         # session.jsonl
echoscribe bias   --workdir run --hotword "<unit>"            # bias.jsonl
echoscribe eval   --workdir run --level text --metric hybrid  # scores.json
echoscribe bench  --workdir run --repeats 3                   # bench.json
echoscribe report --workdir run                               # report.svg
```

`echoscribe train --workdir run --method ctc` trains a tiny model on a freshly
rendered laboratory (torch extra), and `echoscribe demo --workdir run` executes
the whole loop above in one call and prints the numbers it measured.

Exit codes: `0` success, `1` the job failed on valid usage (one line on
stderr), `2` the usage itself was wrong.

Note what the default pipeline decodes: **scripted fixture posteriors** built
from the planted transcripts. It exercises features, decoders, streaming,
scoring and reporting against known answers; it does not run an acoustic
model, and a zero error rate from it certifies the mechanics, not recognition.

## The library

Features from a waveform:

```python
from echoscribe import FeatureConfig, extract_log_mel, read_wav

audio = read_wav("utterance.wav")
log_mel = extract_log_mel(audio, FeatureConfig(num_mel_bins=40))  # (frames, 40)
```

Decoding a posterior matrix:

```python
from echoscribe.decode import ctc_greedy
from echoscribe.decode.posteriors import scripted_posteriors

posteriors = scripted_posteriors([3, 0, 1, 1, 0, 2], num_symbols=5)
greedy = ctc_greedy(posteriors)
greedy.symbols          # (3, 1, 2): blanks dropped, repeats collapsed
```

Scoring a hypothesis against a reference:

```python
from echoscribe.eval import format_summary, score_mixed, wer

format_summary(wer("the cat sat", "the cat"), label="WER")
# 'WER 33.33% | S0 D1 I0 | N3'
score_mixed("打开the document", "打开document")  # hybrid, per-script attribution
```

Streaming with a latency budget:

```python
from echoscribe.stream import AppendableJoint, ChunkSchedule, StreamConfig, StreamDecoder

joint = AppendableJoint(num_symbols=5)
decoder = StreamDecoder(
    joint, StreamConfig(schedule=ChunkSchedule(frames_per_chunk=16))
)
events = decoder.push_logits(logits_block)   # partial hypotheses
events += decoder.finish()                   # the final
decoder.hypothesis.symbols
```

## Make targets

`make build`, `make test` (fast suite), `make test-all` (adds `slow` and
`model` tests), `make format`, `make lint`, `make typecheck`, `make release`
and `make demo` mirror exactly what CI runs.

## File formats

Every serialized artifact is JSON or JSONL with a `schema` version and a
round-trip test: corpus manifests (`manifest.jsonl`, checksummed wavs),
vocabularies, stream sessions (`session.jsonl`), stream recordings (base64
float64, digest-checked), bench results and model bundles (torch payloads
with config, state and history).
