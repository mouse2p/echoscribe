"""The train command is a slow integration: it really renders and trains."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.cli import main
from echoscribe.cli.common import read_json
from echoscribe.model.schema import load_bundle

pytestmark = pytest.mark.model


def train_arguments(tmp_path, **overrides) -> list[str]:
    arguments = [
        "train",
        "--workdir",
        str(tmp_path),
        "--epochs",
        "2",
        "--utterances",
        "6",
        "--hidden",
        "16",
        "--batch-size",
        "3",
    ]
    for name, value in overrides.items():
        arguments.extend([f"--{name.replace('_', '-')}", str(value)])
    return arguments


@pytest.mark.slow
def test_ctc_training_writes_a_bundle_and_a_history(tmp_path, capsys) -> None:
    code = main(train_arguments(tmp_path))
    assert code == 0
    out = capsys.readouterr().out
    assert "echoscribe train (ctc)" in out
    assert "planted transcripts" in out
    history = read_json(tmp_path / "train-history.json")
    assert len(history["losses"]) == 2
    payload = load_bundle(tmp_path / "model.pt", kind="ctc")
    assert payload["history"]["seed"] == history["seed"]
    assert all(isinstance(value, torch.Tensor) for value in payload["state"].values())
    assert any(key.startswith("encoder.") for key in payload["state"])
    assert any(key.startswith("head.") for key in payload["state"])
