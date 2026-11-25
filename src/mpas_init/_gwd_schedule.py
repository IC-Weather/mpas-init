"""Bounded scheduling of independent, fixed-shape geography footprints."""

from collections import defaultdict

import torch


def footprint_batches(shapes, latitude, longitude, *, max_pixels=262144, max_cells=32):
    """Group exact shapes, order by source locality, and retain native output indices."""
    if len(shapes) != len(latitude) or latitude.shape != longitude.shape or latitude.ndim != 1:
        raise ValueError('footprint scheduling requires one shape and coordinate pair per cell')
    if min(max_pixels, max_cells) < 1 or any(min(shape) < 1 for shape in shapes):
        raise ValueError('footprint scheduling requires positive shapes and budgets')
    rows = torch.floor((latitude + .5 * torch.pi) / (torch.pi / 18)).long().clamp(0, 17)
    columns = torch.floor((longitude + torch.pi).remainder(2 * torch.pi) / (torch.pi / 18)).long().clamp(0, 35)
    locality = (36 * rows + columns).tolist()
    grouped = defaultdict(list)
    for cell, shape in enumerate(shapes):
        grouped[(locality[cell], tuple(shape))].append(cell)
    result = []
    for (_, (nx, ny)), cells in grouped.items():
        batch = max(1, min(max_cells, max_pixels // (nx * ny)))
        for first in range(0, len(cells), batch):
            result.append(((nx, ny), cells[first:first + batch]))
    result.sort(key=lambda group: (locality[group[1][0]], group[1][0]))
    return result
