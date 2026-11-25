"""Analyzed UFS surface state through the shared periodic point remapper."""

import torch
from torch import Tensor

from mpas_init.netcdf import PressureLevelSource, UfsSurfaceSource, UFS_SURFACE_VARIABLES, same_forecast_identity
from mpas_init.tracers import periodic_bilinear


UFS_SURFACE_SOURCES = {
    "ufs_sfc_znt": "aerodynamic_roughness_length", "ufs_sfc_z0": "aerodynamic_roughness_length",
    "ufs_sfc_alvsf": "visible_strong_cosine_albedo", "ufs_sfc_alnsf": "near_ir_strong_cosine_albedo",
    "ufs_sfc_alvwf": "visible_weak_cosine_albedo", "ufs_sfc_alnwf": "near_ir_weak_cosine_albedo",
    "ufs_sfc_canwat": "canopy_water_amount", "ufs_sfc_f10m": "wind_speed_ratio_10m_to_lowest_model_level",
    "ufs_sfc_ffhh": "surface_heat_transfer_function", "ufs_sfc_ffmm": "surface_momentum_transfer_function",
    "ufs_sfc_ust": "friction_velocity", "ufs_sfc_hice": "sea_ice_thickness",
    "ufs_sfc_q2": "specific_humidity_2m", "ufs_sfc_t2m": "air_temperature_2m",
    "ufs_sfc_tg3": "deep_soil_temperature", "ufs_sfc_tisfc": "sea_ice_surface_temperature",
}
UFS_SURFACE_UNITS = {name: UFS_SURFACE_VARIABLES[source] for name, source in UFS_SURFACE_SOURCES.items()}


def _validate_grid(latitude: Tensor, longitude: Tensor) -> Tensor:
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("UFS surface requires one-dimensional rectilinear axes")
    if latitude.shape[0] < 3 or longitude.shape[0] < 2:
        raise ValueError("incomplete analyzed UFS surface grid")
    for value in (latitude, longitude):
        torch._assert_async(torch.isfinite(value).all(), "nonfinite analyzed UFS surface")
    lat, lon = torch.sort(latitude).values, torch.sort(longitude).values
    torch._assert_async(((lat[1:] > lat[:-1]).all() & (lat >= -90).all() & (lat <= 90).all()), "invalid UFS surface latitude")
    torch._assert_async(((lat[0] + 90 <= lat[1] - lat[0] + 1.e-8)
                         & (90 - lat[-1] <= lat[-1] - lat[-2] + 1.e-8)), "UFS surface latitude lacks global polar coverage")
    delta = lon[1:] - lon[:-1]
    torch._assert_async(((delta > 0) & ((delta - delta[0]).abs() <= 1.e-8)).all(), "UFS surface longitude must be regular and unique")
    torch._assert_async(((delta[0] * lon.shape[0] - 360).abs() <= 1.e-8), "UFS surface longitude must cover one periodic circuit")
    return lat


def initialize_ufs_surface(source: UfsSurfaceSource, atmosphere: PressureLevelSource,
                           dycore: dict[str, Tensor], native: dict[str, Tensor],
                           surface: dict[str, Tensor], *, polar_extension: str = "reject") -> dict[str, Tensor]:
    """Remap all analyzed fields, then apply original canopy ownership.

    Coordinates use degrees and physical values use UFS_SURFACE_VARIABLES units.
    Nonuniform rectilinear latitude is supported. Explicit ``clamp`` extends only
    the polar caps, each no wider than its adjacent source-latitude interval.
    Specific humidity stays a moist-air fraction; native q2 uses a different basis.
    AD applies within stencils and active masks, before float32 serialization.
    """
    if polar_extension not in ("reject", "clamp"):
        raise ValueError("UFS surface polar extension must be reject or clamp")
    if not same_forecast_identity(source, atmosphere):
        raise ValueError("UFS surface and atmospheric forecast identities differ")
    if set(source.fields) != set(UFS_SURFACE_VARIABLES):
        raise ValueError("incomplete analyzed UFS surface inventory")
    field_grids = {} if source.field_grids is None else source.field_grids
    if set(field_grids) - set(source.fields):
        raise ValueError("analyzed UFS field grids name an unknown quantity")
    primary_lat = _validate_grid(source.latitude, source.longitude)
    for value in source.fields.values():
        torch._assert_async(torch.isfinite(value).all(), "nonfinite analyzed UFS surface")
    for name, value in source.fields.items():
        if name == "aerodynamic_roughness_length" or name.endswith("temperature") or name == "air_temperature_2m":
            torch._assert_async((value > 0).all(), "nonpositive analyzed UFS surface length or temperature")
        elif name.endswith("albedo"):
            torch._assert_async(((value >= 0) & (value <= 1)).all(), "analyzed UFS albedo must be a fraction")
        elif name == "specific_humidity_2m":
            torch._assert_async(((value >= 0) & (value < 1)).all(), "invalid analyzed 2m specific humidity")
        elif name != "canopy_water_amount":
            torch._assert_async((value >= 0).all(), "negative analyzed UFS surface state")
    target_lat, target_lon = dycore["latCell"] * (180.0 / torch.pi), dycore["lonCell"] * (180.0 / torch.pi)
    if (target_lat.ndim != 1 or target_lon.shape != target_lat.shape
            or native["landmask"].shape != target_lat.shape or surface["ivgtyp"].shape != target_lat.shape
            or native["isice_lu"].ndim != 0):
        raise ValueError("UFS surface target coordinate/category shapes differ")
    # Native static files can store pi/2 in float32 before the codec promotes it.
    native_pole = target_lat.new_tensor(torch.pi / 2., dtype=torch.float32).to(target_lat.dtype)
    torch._assert_async((torch.isfinite(target_lat) & torch.isfinite(target_lon)
                         & (dycore["latCell"].abs() <= native_pole)).all(), "invalid UFS surface target coordinates")
    target_lat = target_lat.clamp(-90., 90.)
    torch._assert_async(((native["landmask"] == 0) | (native["landmask"] == 1)).all(), "UFS surface landmask must be binary")
    remapped = {}
    for name, value in source.fields.items():
        grid = field_grids.get(name)
        if grid is not None and (grid.grid_type not in ("regular_latlon", "rectilinear_latlon") or grid.row_counts is not None):
            raise ValueError("analyzed UFS surface fields require rectangular grids")
        latitude, longitude = (source.latitude, source.longitude) if grid is None else (grid.latitude, grid.longitude)
        lat = primary_lat if grid is None else _validate_grid(latitude, longitude)
        if value.shape != (latitude.shape[0], longitude.shape[0]):
            raise ValueError("analyzed UFS surface field shape differs from its grid: " + name)
        if polar_extension == "reject":
            torch._assert_async(((target_lat >= lat[0]) & (target_lat <= lat[-1])).all(), "UFS surface requires explicit polar extension")
        remapped[name] = periodic_bilinear(value, latitude, longitude, target_lat, target_lon)
    eligible = (native["landmask"] == 1) & (surface["ivgtyp"] != native["isice_lu"])
    remapped["canopy_water_amount"] = torch.where(eligible, remapped["canopy_water_amount"].clamp_min(0), 0.0)
    return {name: remapped[quantity] for name, quantity in UFS_SURFACE_SOURCES.items()}
