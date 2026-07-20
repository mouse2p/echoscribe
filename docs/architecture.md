# Architecture

echoscribe is layered so that every stage can be tested against the stage below it
with known answers. Dependencies point downward only: the decoding math never
imports the CLI, the NumPy core never imports torch, and nothing outside
`echoscribe.model` knows torch exists.

```
audio front end        waveform, resample, framing, windows, spectrum, mel,
                       features, cmvn, delta
        |
synthetic laboratory   synth/: phones, lexicon, tone, source, formant, render,
                       noise, room, perturb, corpus, manifest
        |
text and units         text/: normalize, script, segment, bpe, unigram, vocab, units
        |
decoding math          decode/: posteriors, tokens, ctc_greedy, ctc_prefix,
                       transducer, chunked, cache, bias, confidence, alignment
        |
streaming runtime      stream/: chunks, buffer, joint, protocol, runtime,
                       endpoint, replay, session
        |
evaluation             eval/: edit, wer, codeswitch, bench, report
        |
noise utilities        noise/: stft, gains, denoise, snr
        |
trainable scaffolding  model/: schema, stack, encoder, ctc, transducer, train
                       (optional torch extra)
        |
command line           cli/: one module per subcommand, plus the scripted
                       fixtures they share
```

## The synthetic laboratory is the load-bearing wall

Every measured number in this toolkit comes from `echoscribe.synth`: a seeded
formant synthesizer renders an invented English-style and tone-language lexicon
into waveforms, and the transcripts that drove the render are kept as planted
ground truth. This is what makes offline testing honest -- a feature pipeline,
decoder, endpointer or scorer can be checked against an answer that is known
exactly, with no real speech, no pretrained weights and no network involved.

## Design rules the layers follow

* **Determinism is an API.** Same seed and configuration in, same bytes out:
  corpora carry manifest digests, sessions replay to identical event traces,
  reports are pinned by golden digests, and float values that reach a file or a
  summary are rounded where they are produced.
* **Schemas are versioned.** Manifests, vocabularies, sessions, recordings, run
  metadata, bench results and model bundles all carry a schema version and
  refuse foreign ones, so a file written by one release cannot be silently
  misread by another.
* **Errors are typed.** One shallow hierarchy in `echoscribe.errors`, each class
  also deriving from the matching builtin (`ValueError` for bad inputs,
  `RuntimeError` for bad states), so generic handlers keep working.
* **Validation survives `-O`.** Public constructors check their fields with
  `require(...)` rather than `assert`, which Python strips when optimized.
* **torch is quarantined.** `echoscribe.model` is the only package that imports
  it; the package root probes availability without importing, and every test
  that needs it is marked `model` and skips cleanly without it.

## Where the honest seams are

The offline decoder and the trainable model meet at exactly one place: a
posterior/logit matrix of shape `(frames, symbols)` with the blank at index 0.
The NumPy decoders consume such a matrix from anywhere -- the scripted fixtures
of `decode.posteriors`, a torch head in `model.ctc`, or a file a user produced
elsewhere -- and the streaming runtime consumes it chunk by chunk through
`stream.joint.AppendableJoint`. Keeping that seam narrow is why the decoding
math can be exact-tested without a model, and why a trained model can be
decoded without touching the training code.
