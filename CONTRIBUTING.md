# Contributing to echoscribe

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
# optional, CPU wheels only:
.venv/bin/pip install -e ".[dev,torch]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

`make test` is the fast loop (no `slow`-marked tests, ~15 seconds);
`make test-all`, `make format-check`, `make lint` and `make typecheck` are
what CI runs, so they are what a pull request has to pass.

## Commit conventions

* Conventional-commit subjects (`feat`, `fix`, `test`, `docs`, `refactor`,
  `perf`, `chore`) with a short lowercase summary; scope in parentheses when
  it helps (`feat(eval): ...`).
* One coherent change per commit: a mechanism and its tests grow together,
  not one giant commit and not one commit per field.
* Version bumps are patch-only until 1.0 (0.1.0 -> 0.1.1 -> ...) and always
  touch `pyproject.toml`, `src/echoscribe/__init__.py` and `CHANGELOG.md`
  together -- `tests/test_version.py` enforces the agreement.

## Testing conventions

* **Fixtures plant the answer.** Decoders, scorers and stream mechanics are
  tested against scripted posteriors and planted transcripts, so a test can
  assert exact equality instead of a tolerance nobody can justify.
* **Invariants over mirrored logic.** A test that re-implements the code it
  tests proves nothing; the suites prefer properties (n-best stays
  score-ordered, alignments reconstruct the hypothesis, per-script counts sum
  to the total, replay is byte-identical).
* **Golden files are deliberate.** Report digests and serialized shapes are
  pinned; when bytes must move, recompute the golden on purpose and say why
  in the commit message.
* **Determinism rules.** Round floats where they are produced, not where they
  are serialized; seed every random source; never read a clock inside a
  golden path.
* **Markers.** Long training or end-to-end integration goes under
  `@pytest.mark.slow`; anything importing torch also goes under
  `@pytest.mark.model` and must skip cleanly when the extra is absent.

## Honesty rules for docs and outputs

* Every number says where it was measured. Synthetic-laboratory results are
  labeled as such at the point of use.
* Nothing may read as a real-world accuracy claim: no benchmark names, no
  implied human-label evaluation, no pretrained-model comparisons.
* New public modules go into `docs/api.md` in the same change --
  `tests/test_docs.py` fails otherwise.

## Serialized formats

Anything that reaches a file carries a `schema` version and a round-trip
test. A breaking format change bumps the version and keeps the reader
refusing foreign versions loudly rather than guessing.
