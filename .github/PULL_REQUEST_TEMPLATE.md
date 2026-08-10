# What and why

<!-- One paragraph on the change and the mechanism behind it. Link the issue. -->

# How it is tested

<!-- Name the tests. If the change is a mechanism, say what the fixtures plant
     and which invariant the tests pin. -->

# Checklist

- [ ] `make test-all` is green locally (slow and model tests included).
- [ ] `make format-check`, `make lint` and `make typecheck` are green.
- [ ] New public surface is in `docs/api.md` (the docs test enforces it).
- [ ] User-visible change has a `CHANGELOG.md` entry.
- [ ] Any number this PR prints or documents says where it was measured, and
      nothing reads as a real-world accuracy claim.
- [ ] Serialized formats that changed carry a schema-version bump.
