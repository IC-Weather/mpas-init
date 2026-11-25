"""Native MPAS humidity and near-surface initialization diagnostics."""

import torch
from torch import Tensor

from mpas_init.horizontal import global_latlon_indices, remap_atmosphere, sixteen_point
from mpas_init.netcdf import NearSurfaceSource, PressureLevelSource, same_forecast_identity
from mpas_init.surface import MISSING
from mpas_init.thermodynamics import log_pressure_columns, validate_atmospheric_support
from mpas_init.vertical import interpolate_height


DIAGNOSTIC_FIELDS = frozenset(("relhum", "t2m", "rh2", "q2", "u10", "v10"))
NEAR_SURFACE_FIELDS = frozenset(("air_temperature", "relative_humidity", "eastward_wind", "northward_wind"))


def relative_humidity_wrt_ice(temperature: Tensor, liquid_rh: Tensor) -> Tensor:
    """Original percent RH conversion, including its mixed-phase region and cold-only clipping."""
    cold = temperature <= 273.15
    t = torch.where(cold, temperature, 273.15)
    eis = 0.01 * torch.exp(9.550426 - (5723.265 / t) + (3.53068 * torch.log(t)) - (0.00728332 * t))
    ews = 6.112 * torch.exp(17.67 * (t - 273.15) / ((t - 273.15) + 243.5))
    weight = (273.15 - t) / 20.0
    saturation = torch.where(t > 253.15, weight * eis + (1.0 - weight) * ews, eis).clamp_min(1.0e-12)
    converted = (ews.clamp_min(0.0) / saturation * liquid_rh).clamp(0.0, 100.0)
    return torch.where(cold, converted, liquid_rh)


def near_surface_mixing_ratio(temperature: Tensor, liquid_rh: Tensor, pressure: Tensor) -> Tensor:
    """Native q2 formula, using pre-balance interpolated surface pressure, not final surface_pressure."""
    es = 6.112 * torch.exp((17.27 * (temperature - 273.16)) / (temperature - 35.86))
    rs = 0.622 * es * 100.0 / (pressure - es * 100.0)
    return 0.01 * rs * liquid_rh


def initialize_diagnostics(source: PressureLevelSource, near: NearSurfaceSource,
                             dycore: dict[str, Tensor]) -> dict[str, Tensor]:
    """Interpolate supplied liquid-water RH and explicit 2m/10m source fields.

    This prefix does not invent missing humidity diagnostics. Near-surface winds
    are earth-relative point samples, not projected model-edge velocities.
    RH phase conversion uses temperature BEFORE hydrostatic adjustment.
    """
    if "relative_humidity" not in source.fields or set(near.fields) != NEAR_SURFACE_FIELDS:
        raise ValueError("humidity diagnostics require atmospheric liquid RH and complete 2m/10m fields")
    if source.pressure.ndim > 1 and source.pressure.shape != source.fields["geopotential_height"].shape:
        raise ValueError("3-D pressure and geopotential height must be collocated")
    if not same_forecast_identity(near, source):
        raise ValueError("near-surface and atmospheric forecast identities differ")
    field_grids = {} if near.field_grids is None else near.field_grids
    if set(field_grids) - NEAR_SURFACE_FIELDS:
        raise ValueError("near-surface field grids name an unknown field")
    for name, value in near.fields.items():
        grid = field_grids.get(name)
        latitude, longitude = (near.latitude, near.longitude) if grid is None else (grid.latitude, grid.longitude)
        if grid is not None and grid.row_counts is not None:
            raise ValueError("near-surface fields require rectangular grids")
        if latitude.ndim != 1 or longitude.ndim != 1 or value.shape != (latitude.numel(), longitude.numel()):
            raise ValueError("near-surface field shape differs from its grid: " + name)
    missing_fields = validate_atmospheric_support(source.fields, source.pressure, source.field_surface_pressure)
    for name, value in near.fields.items():
        torch._assert_async((torch.isfinite(value) & (value != MISSING)).all(), f"invalid diagnostic source field: near_{name}")
    x, y = global_latlon_indices(source.latitude, source.longitude, dycore["latCell"], dycore["lonCell"],
                                 grid_type=source.grid_type, row_counts=source.row_counts)
    columns = remap_atmosphere(source.fields, ("air_temperature", "geopotential_height", "relative_humidity"),
                               source.latitude, source.longitude, dycore["latCell"], dycore["lonCell"],
                               grid_type=source.grid_type, field_grids=source.field_grids, row_counts=source.row_counts,
                               missing_fields=missing_fields)
    height, order = torch.sort(columns["geopotential_height"], dim=-1)
    midpoint = 0.5 * (dycore["zgrid"][..., 1:] + dycore["zgrid"][..., :-1])
    temperature = interpolate_height(height, torch.gather(columns["air_temperature"], -1, order), midpoint, "lapse-rate")
    humidity = interpolate_height(height, torch.gather(columns["relative_humidity"], -1, order), midpoint, "constant")
    logp = log_pressure_columns(source.pressure, x, y, height.shape[-1], row_counts=source.row_counts)
    psfc = torch.exp(interpolate_height(height, torch.gather(logp, -1, order), dycore["zgrid"][:, :1], "linear"))[:, 0]
    x, y = global_latlon_indices(near.latitude, near.longitude, dycore["latCell"], dycore["lonCell"],
                                 grid_type=near.grid_type)
    remapped = {}
    for name, value in near.fields.items():
        grid = field_grids.get(name)
        xx, yy = ((x, y) if grid is None else global_latlon_indices(
            grid.latitude, grid.longitude, dycore["latCell"], dycore["lonCell"], grid_type=grid.grid_type))
        remapped[name] = sixteen_point(value, xx, yy)
    result = {"relhum": relative_humidity_wrt_ice(temperature, humidity), "t2m": remapped["air_temperature"],
              "rh2": remapped["relative_humidity"], "u10": remapped["eastward_wind"], "v10": remapped["northward_wind"]}
    result["q2"] = near_surface_mixing_ratio(result["t2m"], result["rh2"], psfc)
    for name, value in result.items():
        torch._assert_async(torch.isfinite(value).all(), f"invalid initialized diagnostic: {name}")
    torch._assert_async(((result["t2m"] > 0) & (result["q2"] >= 0)).all(), "invalid near-surface temperature or moisture")
    return result
