# References

The mechanisms implemented here are standard; the code, schemas and tests are
original. Nothing was copied from any of the sources below -- they are listed
so a reader can check the math against its literature.

* A. Graves, "Sequence Transduction with Recurrent Neural Networks", 2012.
  The transducer formulation: the two-dimensional lattice, the joint over
  acoustic and prediction states, blank-advances-time semantics
  (`echoscribe.model.transducer`, `echoscribe.decode.transducer`).
* S. Sak, M. Beaufays, K. Engel, "Label-Synchronous RNN Transducer for
  Streaming End-to-End Speech Recognition" (2018) -- label-synchronous
  monotonic decoding (`echoscribe.decode.chunked`).
* Y. He et al., "Streaming End-to-End Speech Recognition for Mobile
  Devices" (arXiv:1811.11052, 2018) -- monotonic chunked attention and
  latency-bounded chunk scheduling (`echoscribe.stream.chunks`).
* Script-aware mixed tokenization for code-switched text follows the
  convention-level practice common in the code-switch ASR literature of the
  early 2010s (tokenization discussions around Ranzato et al., 2012); the
  attribution rules themselves are this project's own and are documented
  where they are computed (`echoscribe.text.script`,
  `echoscribe.eval.codeswitch`).
* W. Chan et al., "Listen, Attend and Spell", ICASSP 2016 -- background on
  end-to-end ASR objectives the CTC/transducer work sits beside.
* Y. Ephraim, D. Malah, "Speech Enhancement Using a Minimum Mean-Square
  Error Short-Time Spectral Amplitude Estimator", 1984/1985 -- the
  decision-directed a-priori estimate behind `echoscribe.noise.gains.wiener_gain`.
* J. S. Garofolo et al., NIST `sclite` documentation -- WER/CER alignment
  semantics: reference-length denominators, substitution/insertion/deletion
  breakdowns, and why rates may exceed 100% (`echoscribe.eval.wer`).
* Kaldi documentation on fbank/CMVN recipes -- mechanism reference for
  pre-emphasis, povey windowing, power normalization and sliding-window CMVN
  (`echoscribe.framing`, `echoscribe.features`, `echoscribe.cmvn`).
* W3C, "Scalable Vector Graphics (SVG) 1.1 Specification" -- the subset the
  report renderer targets (https://www.w3.org/TR/SVG11/).

Citation policy: these papers describe *what to compute*. The implementations,
conventions (blank at index 0, tie-break orders, schema versions) and every
test in this repository are echoscribe's own.
