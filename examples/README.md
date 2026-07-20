# Examples

Five offline examples, each a directory with a runnable `run.py` and its own
README. Every one of them runs without network access, without pretrained
weights and without real speech: the audio is rendered formants and the
transcripts are planted, so every number an example prints is a measurement
of these mechanics against known answers.

| example | what it shows | needs torch |
| --- | --- | --- |
| `corpus_and_features` | seeded corpus rendering, manifest digests, log-mel + CMVN + deltas | no |
| `decoding_comparison` | CTC greedy vs prefix beam vs transducer on one scripted posterior | no |
| `streaming_session` | chunked streaming, latency budget, endpointing, deterministic replay | no |
| `codeswitch_report` | hybrid English/Chinese scoring and a deterministic SVG report | no |
| `training_demo` | a tiny CTC model whose lattice loss really falls on synthetic speech | yes |

Run any of them with the project virtualenv:

```bash
python examples/<name>/run.py --workdir /tmp/echoscribe-<name>
```

`tests/test_examples.py` executes each example as a subprocess, so an example
that stops running is a failing test rather than a stale document.
