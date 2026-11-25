"""Finite-volume dry-air and represented-water inventories, without mass adjustment."""

import torch
from torch import Tensor


WATER_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")


def atmosphere_inventory(area: Tensor, dry_density: Tensor, height_interfaces: Tensor,
                         water_mixing_ratios: dict[str, Tensor]) -> dict[str, Tensor]:
    """Integrate supplied shallow-atmosphere finite volumes in SI units.

    Density and dry-air mixing ratios are layer values on [cell, layer]. Areas
    are [cell]; height interfaces are [cell, layer+1], increasing upward. The
    caller must supply finite-volume geometry and layer values, not infer them
    from point samples. qv/qc/qr/qi/qs/qg are required; hail qh is optional and
    contributes when present. No unrepresented species or deep-shell factors
    are invented. All returned values retain ordinary forward and reverse AD.
    """
    if (dry_density.ndim != 2 or min(dry_density.shape) < 1
            or area.shape != dry_density.shape[:1]
            or height_interfaces.shape != (dry_density.shape[0], dry_density.shape[1] + 1)):
        raise ValueError("inventory requires cell areas, cell/layer density, and complete height interfaces")
    if not set(WATER_SPECIES[:-1]) <= set(water_mixing_ratios) or set(water_mixing_ratios) - set(WATER_SPECIES):
        raise ValueError("inventory requires qv/qc/qr/qi/qs/qg dry-air ratios, with optional qh only")
    if dry_density.dtype not in (torch.float32, torch.float64):
        raise ValueError("inventory requires float32 or float64 tensors")
    inputs = {"area": area, "dry_density": dry_density, "height_interfaces": height_interfaces,
              **water_mixing_ratios}
    for name, value in inputs.items():
        if value.dtype != dry_density.dtype or value.device != dry_density.device:
            raise ValueError("inventory dtype/device differs: " + name)
        if name in WATER_SPECIES and value.shape != dry_density.shape:
            raise ValueError("inventory water shape differs: " + name)
        torch._assert_async(torch.isfinite(value).all(), "nonfinite inventory input: " + name)
    torch._assert_async((area > 0).all(), "inventory areas must be positive")
    torch._assert_async((dry_density > 0).all(), "inventory dry density must be positive")
    thickness = height_interfaces[:, 1:] - height_interfaces[:, :-1]
    torch._assert_async((thickness > 0).all(), "inventory heights must increase upward")
    layer_dry_mass = dry_density * thickness
    dry_column = layer_dry_mass.sum(-1)
    condensate_column = torch.zeros_like(dry_column)
    result = {"dry_air_column_kg_m2": dry_column, "dry_air_mass_kg": (area * dry_column).sum()}
    for name in WATER_SPECIES:
        if name not in water_mixing_ratios:
            continue
        ratio = water_mixing_ratios[name]
        torch._assert_async((ratio >= 0).all(), "negative inventory water ratio: " + name)
        column = (layer_dry_mass * ratio).sum(-1)
        result[name + "_column_kg_m2"] = column
        result[name + "_mass_kg"] = (area * column).sum()
        if name != "qv":
            condensate_column = condensate_column + column
    water_column = result["qv_column_kg_m2"] + condensate_column
    result.update({"condensate_column_kg_m2": condensate_column,
                   "condensate_mass_kg": (area * condensate_column).sum(),
                   "represented_water_column_kg_m2": water_column,
                   "represented_water_mass_kg": (area * water_column).sum(),
                   "represented_air_mass_kg": (area * (dry_column + water_column)).sum(),
                   "area_m2": area.sum(), "volume_m3": (area * thickness.sum(-1)).sum(),
                   "bottom_min_m": height_interfaces[:, 0].amin(), "bottom_max_m": height_interfaces[:, 0].amax(),
                   "top_min_m": height_interfaces[:, -1].amin(), "top_max_m": height_interfaces[:, -1].amax()})
    for name, value in result.items():
        torch._assert_async(torch.isfinite(value).all(), "nonfinite inventory result: " + name)
    return result
