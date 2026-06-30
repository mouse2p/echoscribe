# Streaming session

Streams one scripted utterance through the chunked runtime twice: once plain,
recording the chunk sequence and replaying it through a fresh decoder, and once
with a blank-ratio endpointer that closes the utterance inside the trailing
silence.

```bash
python examples/streaming_session/run.py --workdir /tmp/echoscribe-ex3
```

What it demonstrates:

* the latency budget in concrete numbers: 16-frame chunks at a 10 ms shift
  bound the look-ahead at 160 ms;
* partial-vs-final hypothesis events over a growing joint;
* deterministic replay: the recorded chunk sequence, fed to a fresh decoder,
  reproduces a byte-identical event trace (compared by digest);
* blank-ratio endpointing with a hangover window, firing after the planted
  units and before the audio runs out;
* a session serialized to JSONL in the library's own format.

What it is not: the logits are scripted fixtures with a planted transcript.
The example exercises scheduling, endpointing and replay mechanics; it does
not measure recognition.
