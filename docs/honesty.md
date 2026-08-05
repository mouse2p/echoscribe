# Honesty and limitations

This page is the contract every number in this repository is held to.

## What echoscribe verifies

The *mechanics* of streaming speech recognition, each against answers that are
known exactly:

* feature pipelines: golden round-trips on seeded waveforms, determinism of
  the same wav + config producing byte-identical features;
* tokenizers and vocabularies: token/id round-trips, golden vocabularies,
  deterministic subword training;
* CTC and transducer decoding: greedy equals the argmax path, prefix-beam
  n-best stays score-ordered, hand-computable lattice losses, transducer
  alignments never revisit frames, brute-force cross-checks of edit distance;
* streaming: a streamed decode equals the batch decode at any chunk size,
  recorded chunk sequences replay to byte-identical event traces, endpointers
  never fire inside a planted utterance at high SNR;
* scoring: alignment operations reconstruct the hypothesis from the
  reference, WER >= 0, per-script counts sum to the overall breakdown;
* noise utilities: measured SNR improvement on planted clean/noisy pairs,
  including the cases where the denoiser loses;
* training: real loss decrease on synthetic data with deterministic seeds --
  evidence that forward, loss and backward agree.

## What echoscribe does not claim

* **No accuracy on real speech.** No pretrained weights, no real datasets, no
  benchmark results. Any WER/CER/RTF printed anywhere in this repository was
  measured on the synthetic laboratory against planted transcripts, and is
  labeled as such at the point of use.
* **No recognizer.** The trainable models are a few tens of thousands of
  parameters trained for seconds on rendered formants. A decreasing loss
  demonstrates the plumbing, nothing more.
* **Zero scores from scripted fixtures certify mechanics.** The default CLI
  pipeline decodes planted fixture posteriors; scoring those decodes zero
  proves the join, alignment and arithmetic work, not that anything was
  recognized.

## Known limitations, stated plainly

* The encoder is bidirectional: it belongs to the offline half of the toolkit.
  Streaming here runs on precomputed logits, not on a chunked acoustic model.
* The transducer training loss is the exact lattice criterion, but the demo
  models are tiny and the corpus is seconds long; overfitting the laboratory
  is expected and is not evidence of learning speech.
* The denoiser assumes the recording starts with noise before speech. A
  stationary tone from sample zero is subtracted along with the noise, and a
  test asserts exactly that failure mode rather than hiding it.
* Spectral gains cannot recover speech masked in the same time-frequency bin
  by noise; measured SNR improvement comes with distortion the SNR number
  does not show.
* The hybrid code-switch score attributes a cross-script substitution to the
  reference side's script -- a convention, documented where it is computed,
  chosen so per-script counts always sum to the total.
* RTF numbers describe one process on one machine; they are comparable with
  each other and with nothing else.

## Labeling rules

Synthetic fixtures say so in their docstrings; example outputs end with a
note naming the planted ground truth; the demo prints where every number
came from before printing the numbers. If a sentence in this repository
could be read as a real-world accuracy claim, that sentence is a bug.
