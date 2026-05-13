"""Shared plumbing for every subcommand: exit codes, error text, JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: The job succeeded.
EXIT_OK = 0
#: The usage was fine but the job failed; one line of explanation went to stderr.
EXIT_FAILURE = 1
#: The usage itself was wrong; argparse or this module said why.
EXIT_USAGE = 2


def fail(message: str) -> int:
    """Print one error line and hand back the failure exit code.

    Subcommands return ``fail(...)`` instead of raising for expected problems
    -- a missing manifest, an empty corpus -- because the message a user
    needs is the specific one, not whatever an exception's str() happens to
    render to.
    """
    print(f"echoscribe: error: {message}", file=sys.stderr)
    return EXIT_FAILURE


def write_json(path: Path | str, payload: object) -> Path:
    """Write JSON with sorted keys and LF newlines, so files stay diffable."""
    target = Path(path)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def read_json(path: Path | str) -> object:
    """Read one JSON document back."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
