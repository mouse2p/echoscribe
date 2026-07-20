# Training demo

Renders a dozen utterances from the synthetic laboratory, builds a ~40k-parameter
encoder plus CTC head, and trains for a few epochs -- printing every epoch's mean
lattice loss and refusing to call a non-decreasing curve a success.

```bash
pip install "echoscribe[dev,torch]" --extra-index-url https://download.pytorch.org/whl/cpu
python examples/training_demo/run.py --workdir /tmp/echoscribe-ex5
```

What it demonstrates:

* `laboratory_examples`: the bridge from rendered audio to normalized feature
  tensors and vocabulary target ids, with planted transcripts as labels;
* `train_ctc`: a seeded loop over the exact CTC lattice loss, one recorded mean
  per epoch;
* honest failure: if the loss does not fall, the demo exits nonzero and says so.

What it is not: a recipe. Seconds of synthetic formant audio, a tiny lexicon
and a few epochs cannot produce a recognizer, and no number here should be
read as accuracy. Without the torch extra the demo prints an installation
note and exits 0 without claiming any result.
