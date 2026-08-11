#!/usr/bin/env python3
"""Smoke-test the built distribution the way a user installs it.

Creates a throwaway virtualenv, installs the built wheel into it -- never the
source tree -- then imports the package, decodes one scripted fixture through
the installed code and runs the console script.  Any failure exits nonzero
naming the first thing that broke.

Run it from the repository root after ``make build``::

    .venv/bin/python scripts/check_package.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SMOKE_SOURCE = """
import echoscribe
from echoscribe.decode.ctc_greedy import ctc_greedy
from echoscribe.decode.posteriors import scripted_posteriors

assert echoscribe.__version__, "the installed package reports no version"
result = ctc_greedy(scripted_posteriors([2, 0, 3, 3, 0, 1], num_symbols=5))
assert result.symbols == (2, 3, 1), result.symbols
print(f"echoscribe {echoscribe.__version__} imports and decodes from the wheel")
"""


def run(command: list[str]) -> str:
    """Run one command, echoing it, and stop at the first failure."""
    print("+", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        raise SystemExit(f"failed with exit code {result.returncode}: {command[0]}")
    return result.stdout


def find_wheel(dist: Path) -> Path:
    """The newest built wheel, or an instruction to build one."""
    wheels = sorted(dist.glob("echoscribe-*.whl"))
    if not wheels:
        raise SystemExit(f"no wheel found in {dist}; run `make build` first")
    return wheels[-1]


def main() -> int:
    """Install into a throwaway venv and exercise the installed package."""
    parser = argparse.ArgumentParser(description="built-wheel install smoke test")
    parser.add_argument("--dist", default=str(ROOT / "dist"), help="directory holding the wheel")
    args = parser.parse_args()
    wheel = find_wheel(Path(args.dist))
    temporary = Path(tempfile.mkdtemp(prefix="echoscribe-check-"))
    try:
        venv = temporary / "venv"
        run([sys.executable, "-m", "venv", str(venv)])
        python = venv / "bin" / "python"
        run([str(python), "-m", "pip", "install", "--quiet", str(wheel)])
        run([str(python), "-c", SMOKE_SOURCE])
        script = venv / "bin" / "echoscribe"
        if not script.exists():
            raise SystemExit("the wheel installed no `echoscribe` console script")
        output = run([str(script), "--version"])
        assert "echoscribe" in output, output
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print(f"OK: {wheel.name} installs, imports, decodes and exposes its CLI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
