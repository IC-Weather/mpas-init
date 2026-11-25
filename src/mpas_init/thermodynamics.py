"""Atmospheric full levels to native MPAS cell thermodynamics."""

import torch
from torch import Tensor

from mpas_init.arithmetic import _exp
from mpas_init.balance import hydrostatic_state
from mpas_init.horizontal import MISSING, HorizontalGrid, global_latlon_indices, remap_atmosphere, sixteen_point
from mpas_init.vertical import interpolate_height


def wps_ice_blend_to_liquid(temperature: Tensor, humidity: Tensor) -> Tensor:
    """Original WPS source-grid percent-RH conversion, without clipping or smoothing.

    The declared WPS blend uses ice at/below 253.15 K and liquid above
    273.15 K. Preserve missing RH; valid RH requires collocated valid T.
    Branch thresholds and mask ownership are discrete, not differentiable.
    """
    if temperature.shape != humidity.shape:
        raise ValueError("RH phase conversion requires collocated temperature and humidity")
    valid = humidity != MISSING
    torch._assert_async((~valid | (torch.isfinite(temperature) & (temperature > 0))).all(),
                        "RH phase conversion requires valid positive collocated temperature")
    torch._assert_async(torch.isfinite(humidity).all(), "nonfinite source relative humidity")
    cold = valid & (temperature <= 273.15)
    t = torch.where(cold, temperature, 273.15)
    # Scalar / tensor uses reciprocal multiplication in eager Torch. WPS divides.
    eis = 0.01 * torch.exp(9.550426 - (torch.full_like(t, 5723.265) / t)
                           + (3.53068 * torch.log(t)) - (0.00728332 * t))
    ews = 6.112 * torch.exp(17.67 * (t - 273.15) / ((t - 273.15) + 243.5))
    weight = (273.15 - t) / 20.0
    saturation = torch.where(t > 253.15, weight * eis + (1.0 - weight) * ews, eis)
    result = torch.where(cold, humidity * (saturation / ews), humidity)
    torch._assert_async(torch.isfinite(result).all(), "nonfinite converted source relative humidity")
    return result


def validate_atmospheric_support(fields: dict[str, Tensor], pressure: Tensor,
                                  field_surface_pressure: dict[str, Tensor] | None = None) -> set[str]:
    """Permit declared missing pixels only below collocated isobaric surfaces.

    The masks and SEARCH ownership are discrete. Surface pressure classifies
    source support; it does not replace native height-based target pressure.
    """
    surfaces = {} if field_surface_pressure is None else field_surface_pressure
    if set(surfaces) - set(fields):
        raise ValueError("surface-pressure declarations name an unknown atmospheric field")
    if surfaces and (pressure.ndim != 1 or pressure.numel() < 2):
        raise ValueError("below-ground masks require a complete isobaric pressure axis")
    torch._assert_async((torch.isfinite(pressure) & (pressure > 0)).all(), "invalid source pressure")
    for name, value in fields.items():
        torch._assert_async(torch.isfinite(value).all(), "nonfinite atmospheric source field: " + name)
        if name not in surfaces:
            torch._assert_async((value != MISSING).all(), "undeclared missing atmospheric source field: " + name)
            continue
        surface = surfaces[name]
        if value.ndim != 3 or surface.shape != value.shape[1:] or value.shape[0] != pressure.numel():
            raise ValueError("below-ground surface pressure must be collocated with its field: " + name)
        if surface.dtype != value.dtype or surface.device != value.device:
            raise ValueError("below-ground surface pressure must match its field precision/device")
        torch._assert_async((torch.isfinite(surface) & (surface > 0)).all(), "invalid source surface pressure")
        above = pressure[:, None, None] <= surface
        torch._assert_async((above.sum(0) >= 2).all(), "insufficient above-ground source levels")
        torch._assert_async(((value != MISSING) | ~above).all(), "missing above-ground atmospheric source field: " + name)
    # Dynamo 2.8 loses entries in dynamically constructed frozensets.
    return set(surfaces)


def log_pressure_columns(pressure: Tensor, x: Tensor, y: Tensor, n_levels: int, *, row_counts: Tensor | None = None) -> Tensor:
    """Remap pressure before taking log, as the original PRES/GFS path does."""
    if pressure.ndim == 1 and pressure.shape[0] == n_levels:
        return torch.log(pressure).expand(y.numel(), -1)
    if pressure.ndim != (2 if row_counts is not None else 3) or pressure.shape[0] != n_levels:
        raise ValueError("pressure must be a level vector or collocated [level, latitude, longitude] grid")
    torch._assert_async((torch.isfinite(pressure) & (pressure > 0)).all(), "invalid source pressure")
    columns = sixteen_point(pressure, x, y, row_counts=row_counts).transpose(-1, -2)
    torch._assert_async((torch.isfinite(columns) & (columns > 0)).all(), "invalid remapped pressure")
    return torch.log(columns)


def pressure_level_state(fields: dict[str, Tensor], pressure_levels: Tensor,
                          latitude: Tensor, longitude: Tensor, lat_cell: Tensor,
                          lon_cell: Tensor, grid: dict[str, Tensor], *,
                          grid_type: str = "regular_latlon",
                          field_grids: dict[str, HorizontalGrid] | None = None,
                          row_counts: Tensor | None = None,
                          field_surface_pressure: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
    """Interpolate complete earth-relative fields and apply MPAS discrete balance.

    Pressure uses either a level vector or a collocated 3-D full-level grid.
    Source coordinates use degrees. Target coordinates use radians. Geography,
    winds, surface state, and serialization are not produced by this function.
    """
    missing_fields = validate_atmospheric_support(fields, pressure_levels, field_surface_pressure)
    x, y = global_latlon_indices(latitude, longitude, lat_cell, lon_cell, grid_type=grid_type, row_counts=row_counts)
    remapped = remap_atmosphere(fields, ("geopotential_height", "air_temperature", "specific_humidity"),
                                latitude, longitude, lat_cell, lon_cell, grid_type=grid_type, field_grids=field_grids,
                                row_counts=row_counts, missing_fields=missing_fields)
    height, order = torch.sort(remapped["geopotential_height"], dim=-1)
    midpoint = 0.5 * (grid["zgrid"][..., 1:] + grid["zgrid"][..., :-1])
    temperature = interpolate_height(height, torch.gather(remapped["air_temperature"], -1, order), midpoint, "lapse-rate")
    specific = interpolate_height(height, torch.gather(remapped["specific_humidity"].clamp_min(0.0), -1, order), midpoint, "constant")
    qv = specific / (1.0 - specific)
    if pressure_levels.ndim > 1 and pressure_levels.shape != fields["geopotential_height"].shape:
        raise ValueError("3-D pressure and geopotential height must be collocated")
    logp = log_pressure_columns(pressure_levels, x, y, height.shape[-1], row_counts=row_counts)
    pressure = _exp(interpolate_height(height, torch.gather(logp, -1, order), midpoint, "linear"))
    state = hydrostatic_state(temperature, pressure, qv,
                               *(grid[name] for name in ("zgrid", "zz", "fzm", "fzp", "dzu", "rdzw")))
    return {**state, "qv": qv}
