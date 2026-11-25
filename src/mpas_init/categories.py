"""Native MPAS category counts, fractions, and ordered surface corrections."""

import torch
from torch import Tensor


def category_counts(pixels: Tensor, cells: Tensor, n_cells: int,
                    category_min: int, category_max: int) -> Tensor:
    """Count valid category IDs; the original mapper ignores out-of-range pixels."""
    if pixels.dtype != torch.int64 or cells.dtype != torch.int64 or pixels.shape != cells.shape or pixels.ndim != 1:
        raise ValueError("category pixels and cell IDs must be matching int64 vectors")
    if n_cells < 1 or category_max < category_min:
        raise ValueError("invalid category range or cell count")
    n_categories = category_max - category_min + 1
    valid = (pixels >= category_min) & (pixels <= category_max)
    categories = (pixels - category_min).clamp(0, n_categories - 1)
    flat = categories * n_cells + cells
    return pixels.new_zeros(n_categories * n_cells).scatter_add(0, flat, valid.to(torch.int64)).reshape(n_categories, n_cells)


def category_mode_fractions(counts: Tensor, category_min: int) -> tuple[Tensor, Tensor]:
    """Lowest ID wins ties. Uncovered cells have NaN fractions, not false coverage."""
    if counts.ndim != 2 or min(counts.shape) < 1 or counts.dtype != torch.int64:
        raise ValueError("category counts must be a nonempty int64 category-by-cell tensor")
    total = counts.sum(dim=0)
    fractions = counts.to(torch.float64) / total.clamp_min(1).to(torch.float64)
    fractions = torch.where(total[None] > 0, fractions, torch.full_like(fractions, torch.nan))
    return counts.argmax(dim=0) + category_min, fractions


def surface_categories(land_counts: Tensor, soil_counts: Tensor, *, land_min: int,
                       soil_min: int, land_water: int, land_ice: int,
                       soil_water: int) -> dict[str, Tensor]:
    """Apply the original ice-then-water consistency rules, without changing fractions."""
    land, land_fraction = category_mode_fractions(land_counts, land_min)
    soil, soil_fraction = category_mode_fractions(soil_counts, soil_min)
    if land.shape != soil.shape:
        raise ValueError("land-use and soil counts must contain the same cells")
    soil = torch.where(land == land_ice, 16, soil)
    water = (land == land_water) | (soil == soil_water)
    land = torch.where(water, land_water, land)
    soil = torch.where(water, soil_water, soil)
    return {"lu_index": land, "soilcat_top": soil, "landmask": (land != land_water).to(torch.int64),
            "landusef": land_fraction, "soilf": soil_fraction}
