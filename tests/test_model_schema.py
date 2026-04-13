"""Bundle files are the persistence contract for every trainable piece."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("the torch extra is not installed", allow_module_level=True)

from echoscribe.model.schema import (
    MODEL_KINDS,
    MODEL_SCHEMA_VERSION,
    load_bundle,
    save_bundle,
)

pytestmark = pytest.mark.model


def test_a_bundle_round_trips_its_tensors_and_config(tmp_path) -> None:
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    target = save_bundle(tmp_path / "m.pt", "encoder", {"hidden": 8}, state)
    payload = load_bundle(target)
    assert payload["schema"] == MODEL_SCHEMA_VERSION
    assert payload["kind"] == "encoder"
    assert payload["config"] == {"hidden": 8}
    assert torch.equal(payload["state"]["weight"], state["weight"])


def test_the_kinds_are_the_three_heads() -> None:
    assert MODEL_KINDS == ("encoder", "ctc", "transducer")


def test_history_rides_along_when_given(tmp_path) -> None:
    target = save_bundle(tmp_path / "m.pt", "ctc", {}, {}, history={"losses": [1.0, 0.5]})
    assert load_bundle(target)["history"] == {"losses": [1.0, 0.5]}


def test_a_bundle_without_history_has_no_history_key(tmp_path) -> None:
    target = save_bundle(tmp_path / "m.pt", "ctc", {}, {})
    assert "history" not in load_bundle(target)


def test_saved_tensors_are_detached(tmp_path) -> None:
    source = torch.zeros(2, requires_grad=True)
    target = save_bundle(tmp_path / "m.pt", "encoder", {}, {"w": source})
    assert not load_bundle(target)["state"]["w"].requires_grad


def test_a_foreign_schema_version_is_refused(tmp_path) -> None:
    from echoscribe.errors import ModelError

    target = tmp_path / "m.pt"
    torch.save({"schema": 99, "kind": "encoder", "config": {}, "state": {}}, target)
    with pytest.raises(ModelError, match="not supported"):
        load_bundle(target)


def test_a_missing_key_is_refused(tmp_path) -> None:
    from echoscribe.errors import ModelError

    target = tmp_path / "m.pt"
    torch.save({"schema": MODEL_SCHEMA_VERSION, "kind": "encoder", "config": {}}, target)
    with pytest.raises(ModelError, match="missing 'state'"):
        load_bundle(target)


def test_the_kind_is_checked_when_requested(tmp_path) -> None:
    from echoscribe.errors import ModelError

    target = save_bundle(tmp_path / "m.pt", "encoder", {}, {})
    with pytest.raises(ModelError, match="not the requested"):
        load_bundle(target, kind="transducer")


def test_an_unknown_kind_cannot_be_saved(tmp_path) -> None:
    from echoscribe.errors import ModelError

    with pytest.raises(ModelError, match="kind"):
        save_bundle(tmp_path / "m.pt", "magic", {}, {})


def test_a_config_json_cannot_hold_is_refused(tmp_path) -> None:
    from echoscribe.errors import ModelError

    with pytest.raises(ModelError, match="JSON"):
        save_bundle(tmp_path / "m.pt", "encoder", {"dims": {3, 4}}, {})


def test_a_non_tensor_state_entry_is_refused(tmp_path) -> None:
    from echoscribe.errors import ModelError

    with pytest.raises(ModelError, match="must be a tensor"):
        save_bundle(tmp_path / "m.pt", "encoder", {}, {"w": [1.0]})


def test_a_missing_file_is_refused(tmp_path) -> None:
    from echoscribe.errors import ModelError

    with pytest.raises(ModelError, match="does not exist"):
        load_bundle(tmp_path / "absent.pt")
