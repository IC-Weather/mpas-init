"""Native land-only integer climatology reductions and explicitly field-specific means."""

import torch
from torch import Tensor


def land_accumulation(pixels: Tensor, cells: Tensor, landmask: Tensor, *, missing: int,
                       fill: int | None) -> tuple[Tensor, Tensor]:
    """Reduce native integer pixels in layer-by-pixel order.

    Monthly fields count missing samples after substitution. Snow albedo instead
    excludes missing samples. Inputs already passed the native C_FLOAT conversion.
    """
    if pixels.ndim != 2 or min(pixels.shape) < 1 or pixels.dtype != torch.int64:
        raise ValueError("climatology pixels must be a nonempty int64 layer-by-pixel tensor")
    if cells.ndim != 1 or cells.dtype != torch.int64 or cells.shape[0] != pixels.shape[1]:
        raise ValueError("climatology cell IDs must be matching int64 pixel IDs")
    if landmask.ndim != 1 or landmask.dtype not in (torch.int32, torch.int64):
        raise ValueError("landmask must be an integer cell vector")
    if fill is None and pixels.shape[0] != 1:
        raise ValueError("missing-value exclusion applies only to one-layer snow albedo")
    retained = landmask[cells] == 1
    valid = pixels != missing
    if fill is None:
        retained = retained & valid[0]
        values = pixels
    else:
        values = torch.where(valid, pixels, fill)
    values = torch.where(retained[None], values, 0)
    sums = pixels.new_zeros((pixels.shape[0], landmask.shape[0])).scatter_add(1, cells[None].expand_as(pixels), values)
    counts = pixels.new_zeros(landmask.shape[0]).scatter_add(0, cells, retained.to(torch.int64))
    return sums, counts


def _mean(total: Tensor, count: Tensor, scale: float, fill: float) -> Tensor:
    value = (total.to(torch.float64) / count.clamp_min(1).to(torch.float64)) * scale
    return torch.where(count[None] > 0, value, fill)


def snow_albedo(total: Tensor, count: Tensor, scale: float) -> Tensor:
    """Native annual maximum snow albedo, as a fraction, with zero uncovered/water fill."""
    return 0.01 * _mean(total, count, scale, 0.0).squeeze(0)


def vegetation_fraction(total: Tensor, count: Tensor) -> dict[str, Tensor]:
    """Native monthly vegetation percent. The geotile scale factor is not applied."""
    values = _mean(total, count, 1.0, 0.0)
    return {"greenfrac": values, "shdmin": values.amin(dim=0), "shdmax": values.amax(dim=0)}


def monthly_albedo(total: Tensor, count: Tensor, scale: float, landuse: Tensor, ice_category: int) -> Tensor:
    """Native percent albedo, including unscaled uncovered fill and the 70-percent ice override."""
    values = _mean(total, count, scale, 8.0)
    return torch.where(landuse[None] == ice_category, 70.0, values)


def leaf_area_index(total: Tensor, count: Tensor, scale: float) -> Tensor:
    """Native monthly LAI with zero uncovered/water fill."""
    return _mean(total, count, scale, 0.0)
