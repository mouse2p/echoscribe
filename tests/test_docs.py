"""What the docs promise, the package must actually have."""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def documented_modules() -> list[str]:
    """Every ``echoscribe.*`` dotted name the API reference mentions."""
    text = (DOCS / "api.md").read_text(encoding="utf-8")
    names = re.findall(r"echoscribe(?:\.[a-z_][a-z0-9_]*)+", text)
    return sorted(set(names))


def test_the_api_reference_names_the_whole_toolkit() -> None:
    modules = documented_modules()
    assert len(modules) >= 50, f"only {len(modules)} modules documented"
    for package in ("synth", "text", "decode", "stream", "eval", "noise", "model", "cli"):
        assert any(name.startswith(f"echoscribe.{package}.") for name in modules), package


def test_every_documented_module_exists() -> None:
    for name in documented_modules():
        if ".model" in name:
            # find_spec resolves the module without importing torch
            assert importlib.util.find_spec(name) is not None, name
        else:
            importlib.import_module(name)


def test_every_source_module_is_documented() -> None:
    documented = set(documented_modules())
    root = Path(__file__).resolve().parents[1] / "src" / "echoscribe"
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        # root.parent is src/, so the relative parts already start with "echoscribe"
        module = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        assert module in documented, f"{module} is missing from docs/api.md"


def test_the_docs_index_links_to_real_files() -> None:
    text = (DOCS / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", text)
    assert links, "the index should link somewhere"
    for link in links:
        assert (DOCS / link).exists(), link


def test_the_guides_agree_on_what_is_not_claimed() -> None:
    honesty = (DOCS / "honesty.md").read_text(encoding="utf-8")
    usage = (DOCS / "usage.md").read_text(encoding="utf-8")
    assert "does not claim" in honesty.lower() or "not claim" in honesty.lower()
    assert "scripted fixture" in usage


def test_the_readme_lists_every_command_and_example() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for command in (
        "synth",
        "feats",
        "decode",
        "stream",
        "bias",
        "eval",
        "bench",
        "report",
        "train",
        "demo",
    ):
        assert f"echoscribe {command}" in readme or f"`{command}`" in readme, command
    for example in (
        "corpus_and_features",
        "decoding_comparison",
        "streaming_session",
        "codeswitch_report",
        "training_demo",
    ):
        assert example in readme, example


def test_the_readme_links_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([^)#]+)\)", readme):
        if link.startswith("http"):
            continue
        assert (root / link).exists(), link
