---
name: Bug report
about: Something in echoscribe behaves contrary to its documentation or tests
labels: bug
---

## What happened

<!-- One paragraph, with the exact command or snippet and its output. -->

## What the documentation or tests promise instead

<!-- Quote the docstring, docs page or test that the behaviour contradicts. -->

## Reproduction

```bash
# A minimal, offline reproduction. If it needs audio, generate it with
# `echoscribe synth` and include the seed -- everything in this project is
# reproducible from seeds, and a bug report should be too.
```

## Environment

- echoscribe version:
- Python version:
- OS:
- NumPy version:
- torch version (or "not installed"):

## Checklist

- [ ] The reproduction runs fully offline (no datasets, no model downloads).
- [ ] I checked whether the numbers involved come from the synthetic
      laboratory, and said so if they do.
