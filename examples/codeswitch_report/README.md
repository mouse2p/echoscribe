# Code-switch report

Scores four hand-written English/Chinese/mixed pairs with the hybrid metric and
renders one deterministic SVG report: a score table plus the mixed-token
alignment grid of the code-switched utterance.

```bash
python examples/codeswitch_report/run.py --workdir /tmp/echoscribe-ex4
```

What it demonstrates:

* `score_mixed`: one global alignment over a mixed token stream, with every
  error attributed to the script it landed in;
* `format_code_switch` / `format_summary`: the fixed one-line score formats;
* `render_report` + `write_report`: sections stacked into one SVG stamped with
  run metadata, byte-identical for identical inputs;
* plain `wer` and `cer` next to the hybrid, so the granularity choice is visible.

What it is not: the hypothesis pairs are hand-written demonstrations with
planted errors. No recognizer produced them, and the scores validate the
scoring arithmetic, not any system's accuracy.
