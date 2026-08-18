"""The declared version must agree with the installed distribution metadata."""

from importlib import metadata

import echoscribe


def test_module_version_matches_distribution_metadata() -> None:
    assert echoscribe.__version__ == metadata.version("echoscribe")


def test_version_is_a_three_part_release() -> None:
    parts = echoscribe.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_the_public_namespace_is_importable_from_the_package_root() -> None:
    for name in echoscribe.__all__:
        assert hasattr(echoscribe, name), name


def test_the_exported_names_cover_the_front_end() -> None:
    for name in (
        "Audio",
        "FeatureConfig",
        "FrameConfig",
        "extract_log_mel",
        "frame_signal",
        "read_wav",
        "write_wav",
    ):
        assert name in echoscribe.__all__


def test_the_package_declares_itself_typed() -> None:
    # py.typed is what tells a downstream mypy run to check these sources at all.
    from pathlib import Path

    import echoscribe

    marker = Path(echoscribe.__file__).parent / "py.typed"
    assert marker.is_file()


def test_the_version_agrees_with_pyproject_and_the_changelog() -> None:
    from pathlib import Path

    root = Path(echoscribe.__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{echoscribe.__version__}"' in pyproject
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{echoscribe.__version__}]" in changelog
    latest = changelog.split("## [", 1)[1].split("]", 1)[0]
    assert latest == echoscribe.__version__, "the changelog must lead with the release"
