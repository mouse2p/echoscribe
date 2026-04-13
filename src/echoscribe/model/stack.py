"""Frame stacking: cheap local context for a compact encoder.

Stacking concatenates each frame with its predecessors, which trades feature
width for temporal resolution: an encoder over stacked frames sees local
dynamics without spending recurrent depth on them.  It is the standard trick
behind small streaming recognizers, and here it is exactly what keeps the
demo encoder tiny while still seeing more than one 10 ms slice at a time.
"""

from __future__ import annotations

import torch

from echoscribe.errors import ModelError, require


def stack_frames(features: torch.Tensor, stack: int) -> torch.Tensor:
    """Concatenate each frame with its ``stack - 1`` predecessors.

    The left edge is padded by replicating the first frame, so the output has
    as many time steps as the input and frame zero sees itself ``stack``
    times -- a small lie at the boundary, but a boundary is exactly where a
    recognizer has no left context anyway.  Within each window the oldest
    frame comes first, which is the order a recurrent encoder reads time in.
    """
    require(
        isinstance(features, torch.Tensor),
        f"features must be a tensor, got {type(features).__name__}",
        ModelError,
    )
    require(
        features.dim() == 3,
        f"features must be (batch, time, dim), got {tuple(features.shape)}",
        ModelError,
    )
    require(
        isinstance(stack, int) and not isinstance(stack, bool) and stack >= 1,
        f"stack must be an integer >= 1, got {stack!r}",
        ModelError,
    )
    if stack == 1:
        return features
    batch, time, dim = features.shape
    padded = torch.cat([features[:, :1].expand(batch, stack - 1, dim), features], dim=1)
    windows = padded.unfold(1, stack, 1)  # (batch, time, dim, stack)
    return windows.permute(0, 1, 3, 2).reshape(batch, time, stack * dim)
