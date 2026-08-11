# Scripts

Developer scripts that are part of the release path rather than the package.

## check_package.py

Smoke-tests the **built wheel** the way a user installs it: a throwaway
virtualenv, `pip install` of `dist/echoscribe-*.whl`, then an import, one
scripted decode through the installed code, and a `--version` call through
the installed console script. It never touches the source tree, so it catches
packaging mistakes -- missing modules, broken entry points, files the build
backend did not include -- that in-place tests cannot see.

```bash
make build
.venv/bin/python scripts/check_package.py
```

The release workflow runs exactly these two commands before attaching
artifacts to the GitHub release, and `tests/test_scripts.py` runs the check
as a slow test whenever a wheel is present in `dist/`.
