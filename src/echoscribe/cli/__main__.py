"""``python -m echoscribe.cli``: the console script's other front door.

The Makefile's demo target and any user who prefers module invocation reach
the CLI through here; both doors call the same :func:`echoscribe.cli.main`,
so there is one behavior to test and one set of exit codes.
"""

from echoscribe.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
