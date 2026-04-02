"""Versioned bundle files for everything trainable.

A bundle is one ``torch.save`` payload with four required keys -- schema
version, kind, configuration and state dict -- plus an optional training
history.  The version is checked on load because a state dict without its
architecture is unreadable: tensors alone do not say which shape of network
they came from, and guessing is how weights get loaded silently into the
wrong graph.  The kind is checked too, so an encoder bundle can never be
opened as a transducer by a caller who mixed up two paths.

Configurations must survive ``json.dumps``, which keeps them to primitives a
future version can still read, and lets a bundle's config be diffed as text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from echoscribe.errors import ModelError, require, require_in

MODEL_SCHEMA_VERSION = 1
MODEL_KINDS = ("encoder", "ctc", "transducer")
BUNDLE_KEYS = ("schema", "kind", "config", "state")


def _json_ready(payload: Mapping[str, object], what: str) -> dict:
    """The mapping as a plain dict, refused unless JSON can hold it."""
    plain = dict(payload)
    try:
        json.dumps(plain)
    except (TypeError, ValueError) as error:
        raise ModelError(f"the bundle {what} must serialize to JSON: {error}") from error
    return plain


def save_bundle(
    path: Path | str,
    kind: str,
    config: Mapping[str, object],
    state: Mapping[str, torch.Tensor],
    history: Mapping[str, object] | None = None,
) -> Path:
    """Write one bundle file and return its path."""
    require_in("kind", kind, MODEL_KINDS, ModelError)
    checked_config = _json_ready(config, "config")
    checked_history = None if history is None else _json_ready(history, "history")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        require(
            isinstance(value, torch.Tensor),
            f"state entry {name!r} must be a tensor, got {type(value).__name__}",
            ModelError,
        )
        tensors[name] = value.detach()
    payload: dict = {
        "schema": MODEL_SCHEMA_VERSION,
        "kind": kind,
        "config": checked_config,
        "state": tensors,
    }
    if checked_history is not None:
        payload["history"] = checked_history
    target = Path(path)
    torch.save(payload, target)
    return target


def load_bundle(path: Path | str, kind: str | None = None) -> dict:
    """Read one bundle file, refusing foreign schemas and the wrong kind."""
    target = Path(path)
    require(target.exists(), f"the bundle {target} does not exist", ModelError)
    payload = torch.load(target, weights_only=True)
    require(isinstance(payload, dict), f"the bundle {target} is not a mapping", ModelError)
    version = payload.get("schema")
    require(
        version == MODEL_SCHEMA_VERSION,
        f"bundle schema {version!r} is not supported; this build reads {MODEL_SCHEMA_VERSION}",
        ModelError,
    )
    for key in BUNDLE_KEYS:
        require(key in payload, f"the bundle is missing {key!r}", ModelError)
    if kind is not None:
        require(
            payload["kind"] == kind,
            f"the bundle holds a {payload['kind']!r}, not the requested {kind!r}",
            ModelError,
        )
    return payload
