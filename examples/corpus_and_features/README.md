# Corpus and features

Generates a seeded synthetic corpus, renders it to WAV files with a manifest,
and walks the feature pipeline over the result: log-mel filterbank, per-utterance
CMVN, and delta stacking.

```bash
python examples/corpus_and_features/run.py --workdir /tmp/echoscribe-ex1
```

What it demonstrates:

* `generate_corpus` + `render_corpus`: text and audio from one seed, with a
  manifest whose checksums describe the bytes actually written;
* the feature chain `extract_log_mel` -> `apply_cmvn` -> `stack_features`;
* determinism: re-rendering the corpus reproduces the manifest digest exactly.

What it is not: the transcripts are **planted ground truth** produced by the
synthetic lexicon, and the audio is rendered formants. Nothing here involves
real speech, a real recognizer, or a downloaded model.
