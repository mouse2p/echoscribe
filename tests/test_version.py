"""The declared version must agree with the installed distribution metadata."""

from importlib import metadata

import echoscribe


def test_module_version_matches_distribution_metadata() -> None:
    assert echoscribe.__version__ == metadata.version("echoscribe")


def test_version_is_a_three_part_release() -> None:
    parts = echoscribe.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
