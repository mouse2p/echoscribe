# Decoding comparison

Decodes one scripted posterior matrix three ways -- CTC greedy, CTC prefix-beam
search, and a transducer over a history-indexed joint table -- and checks that
all three replay the planted symbol sequence.

```bash
python examples/decoding_comparison/run.py --workdir /tmp/echoscribe-ex2
```

What it demonstrates:

* `scripted_posteriors`: a log-posterior fixture whose per-frame argmax is a
  chosen path, the standard way this toolkit tests decoders against known
  answers;
* CTC greedy collapse (repeats fold, blanks drop) next to prefix-beam n-best
  output ordered by score;
* the transducer's different emission rule -- one symbol per frame under a
  per-frame cap, no collapsing -- over a `TableJoint`.

What it is not: no audio, no torch, no model. The agreement between decoders
is guaranteed by the fixture and validates their arithmetic; it says nothing
about recognition accuracy.
