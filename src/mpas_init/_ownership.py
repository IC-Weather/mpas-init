"""Exact distance ordering for independent, fixed-geometry pixel blocks."""

import torch

from mpas_init.static import _distance2


def nearest_block_owners(points, centers, ids):
    """Select lexicographically ordered candidates without changing scalar distances."""
    distance2 = _distance2(points[:, :, None, :] - centers[:, None, :, :])
    return torch.gather(ids, 1, torch.argmin(distance2, dim=-1))
