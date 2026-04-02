"""Trainable acoustic-model scaffolding, gated on the optional torch extra.

Everything here is deliberately tiny -- encoders and joints of a few tens of
thousands of parameters, trained on seconds of synthetic audio -- and exists
to exercise the *mechanics* around a trainable recognizer: frame stacking,
CTC and transducer heads, lattice losses, seeded reproducibility and versioned
bundle files.  It is not a recipe for a competitive acoustic model, it trains
only on the bundled synthetic laboratory, and it runs only on CPU.

Importing this package never requires torch.  Availability is probed with
:data:`TORCH_AVAILABLE`, and :func:`require_torch` turns a missing extra into
a clean :class:`~echoscribe.errors.ModelError` instead of an ``ImportError``
from somewhere deep.  The submodules below *do* import torch at their top, so
callers check the gate first; the test suite enforces the same contract with
``pytest.mark.model`` and a module-level skip.
"""

from __future__ import annotations

import importlib.util

from echoscribe.errors import ModelError

TORCH_AVAILABLE: bool = importlib.util.find_spec("torch") is not None


def require_torch() -> None:
    """Raise :class:`ModelError` unless the torch extra is installed.

    Meant for entry points -- the CLI above all -- that want to fail with an
    installation instruction rather than a traceback through ``import torch``.
    """
    if not TORCH_AVAILABLE:
        raise ModelError(
            "this feature needs the torch extra; install it with "
            'pip install "echoscribe[torch]" -- CPU wheels are sufficient'
        )


__all__ = ["TORCH_AVAILABLE", "require_torch"]
