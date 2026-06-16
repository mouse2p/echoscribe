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


@pytest.mark.slow
def test_two_runs_with_one_seed_agree(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert main(train_arguments(first, model_seed=11)) == 0
    assert main(train_arguments(second, model_seed=11)) == 0
    assert read_json(first / "train-history.json") == read_json(second / "train-history.json")


@pytest.mark.slow
def test_transducer_training_writes_encoder_predictor_and_joint(tmp_path) -> None:
    code = main(train_arguments(tmp_path, method="transducer"))
    assert code == 0
    payload = load_bundle(tmp_path / "model.pt", kind="transducer")
    prefixes = {key.split(".")[0] for key in payload["state"]}
    assert prefixes == {"encoder", "predictor", "joint"}
    history = read_json(tmp_path / "train-history.json")
    assert history["epochs_completed"] == 2
