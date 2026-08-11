"""The packaging smoke check is run for real, when a wheel exists."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_the_built_wheel_installs_and_decodes() -> None:
    if not sorted((ROOT / "dist").glob("echoscribe-*.whl")):
        pytest.skip("no built wheel in dist/; run `make build` first")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_package.py")],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "installs, imports, decodes" in result.stdout


def test_the_smoke_source_pins_a_planted_decode() -> None:
    # The snippet the throwaway venv runs must keep matching the library's
    # collapse semantics: [2, 0, 3, 3, 0, 1] -> (2, 3, 1).
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_package
        from echoscribe.decode.ctc_greedy import ctc_greedy
        from echoscribe.decode.posteriors import scripted_posteriors

        assert ctc_greedy(scripted_posteriors([2, 0, 3, 3, 0, 1], num_symbols=5)).symbols == (
            2,
            3,
            1,
        )
        assert "assert result.symbols == (2, 3, 1)" in check_package.SMOKE_SOURCE
    finally:
        sys.path.pop(0)
