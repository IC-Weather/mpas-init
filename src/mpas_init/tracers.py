"""Analyzed ozone and declared UFS R4 clear-sky cold-start tracers."""

import torch
from torch import Tensor

from mpas_init.netcdf import OzoneSource, PressureLevelSource, same_forecast_identity


UFS_TRACER_FIELDS = frozenset(("o3mr", "ni", "nr", "sgs_tke"))


def mpas_layer_pressure(rho: Tensor, theta: Tensor, qv: Tensor) -> Tensor:
    """Original UFS preparation equation of state, using native dry density."""
    if rho.shape != theta.shape or rho.shape != qv.shape:
        raise ValueError("MPAS thermodynamic shapes differ")
    torch._assert_async((torch.isfinite(rho) & (rho > 0)).all(), "invalid native dry density")
    torch._assert_async((torch.isfinite(theta) & (theta > 0)).all(), "invalid native potential temperature")
    torch._assert_async((torch.isfinite(qv) & (qv >= 0)).all(), "invalid native water-vapor mixing ratio")
    rd, rv, cp, p0 = 287.0, 461.6, 3.5 * 287.0, 1.0e5
    theta_m = theta * (1.0 + (rv / rd) * qv)
    exner = (rho * rd * theta_m / p0) ** (rd / (cp - rd))
    pressure = rho * rd * theta_m * exner
    torch._assert_async((torch.isfinite(pressure) & (pressure > 0)).all(), "invalid diagnosed pressure")
    return pressure


def periodic_bilinear(field: Tensor, latitude: Tensor, longitude: Tensor,
                       target_latitude: Tensor, target_longitude: Tensor, *, row_counts: Tensor | None = None) -> Tensor:
    """Reference periodic bilinear point remapping; all coordinates use degrees."""
    dtype = torch.promote_types(field.dtype, torch.promote_types(latitude.dtype, longitude.dtype))
    lat, lat_order = torch.sort(latitude.to(dtype))
    longitude = longitude.to(dtype)
    target_lat = torch.clamp(target_latitude, lat[0], lat[-1])
    lat_hi = torch.searchsorted(lat, target_lat.contiguous(), right=True).clamp(1, lat.shape[0] - 1)
    lat_lo = lat_hi - 1
    lat_weight = (target_lat - lat[lat_lo]) / (lat[lat_hi] - lat[lat_lo])
    if row_counts is not None:
        offsets = torch.cumsum(row_counts, 0) - row_counts
        longitude = torch.remainder(longitude, 360.0)

        def row_values(rows):
            row = lat_order[rows]
            start, count = offsets[row], row_counts[row]
            origin = longitude[start]
            target = torch.remainder(target_longitude - origin, 360.0) + origin
            lo = torch.floor((target - origin) * count / 360.0).to(torch.int64).clamp_min(0)
            lo = torch.minimum(lo, count - 1)

            def bounds(index):
                high = (index + 1).remainder(count)
                left = longitude[start + index]
                right = torch.where(high == 0, origin + 360.0, longitude[start + high])
                return left, right, high

            left, right, _ = bounds(lo)
            # Stored angular rounding can put the nominal regular-grid estimate one node away.
            lo = torch.where(target < left, lo - 1, torch.where(target >= right, lo + 1, lo))
            lo = torch.minimum(lo.clamp_min(0), count - 1)
            left, right, high = bounds(lo)
            torch._assert_async(((target >= left) & (target <= right)).all(), "invalid reduced bilinear bracket")
            weight = (target - left) / (right - left)
            return (1.0 - weight) * field[..., start + lo] + weight * field[..., start + high]

        lower, upper = row_values(lat_lo), row_values(lat_hi)
        return (1.0 - lat_weight) * lower + lat_weight * upper
    # Match the reference coordinate preparation before periodic query arithmetic.
    lon, lon_order = torch.sort(torch.remainder(longitude, 360.0))
    data = field.index_select(-2, lat_order).index_select(-1, lon_order)
    target_lon = torch.remainder(target_longitude - lon[0], 360.0) + lon[0]
    extended = torch.cat((lon, lon[:1] + 360.0))
    lon_hi = torch.searchsorted(extended, target_lon.contiguous(), right=True).clamp(1, lon.shape[0])
    lon_lo = lon_hi - 1
    lon_weight = (target_lon - extended[lon_lo]) / (extended[lon_hi] - extended[lon_lo])
    wrapped_hi = lon_hi.remainder(lon.shape[0])
    lower = (1.0 - lon_weight) * data[..., lat_lo, lon_lo] + lon_weight * data[..., lat_lo, wrapped_hi]
    upper = (1.0 - lon_weight) * data[..., lat_hi, lon_lo] + lon_weight * data[..., lat_hi, wrapped_hi]
    return (1.0 - lat_weight) * lower + lat_weight * upper


def vertical_log_pressure(pressure: Tensor, profiles: Tensor, target: Tensor, *, extension: str = "reject") -> Tensor:
    """Reference log-pressure interpolation, with a shared axis or column pressures."""
    if extension not in ("reject", "constant"):
        raise ValueError("ozone extension must be reject or constant")
    if (pressure.ndim not in (1, 2) or pressure.shape[0] < 2 or profiles.ndim != 2 or target.ndim != 2
            or profiles.shape != (pressure.shape[0], target.shape[0])
            or pressure.ndim == 2 and pressure.shape != profiles.shape):
        raise ValueError("ozone pressure/profile/cell shape mismatch")
    levels, order = torch.sort(pressure, dim=0)
    torch._assert_async((torch.isfinite(levels) & (levels > 0)).all(), "invalid ozone pressure levels")
    torch._assert_async((levels[1:] > levels[:-1]).all(), "ozone pressure levels are not unique")
    torch._assert_async((torch.isfinite(target) & (target > 0)).all(), "invalid target pressure")
    lower = levels[0] if pressure.ndim == 1 else levels[0, :, None]
    upper = levels[-1] if pressure.ndim == 1 else levels[-1, :, None]
    if extension == "reject":
        torch._assert_async(((target >= lower) & (target <= upper)).all(), "insufficient ozone pressure coverage")
    clipped = torch.clamp(target, lower, upper)
    log_levels, log_target = torch.log(levels), torch.log(clipped)
    source = (profiles.index_select(0, order) if pressure.ndim == 1 else torch.gather(profiles, 0, order)).transpose(0, 1)
    log_levels = (log_levels[None, :].expand(target.shape[0], -1) if pressure.ndim == 1 else log_levels.T).contiguous()
    hi = torch.searchsorted(log_levels, log_target.contiguous(), right=True).clamp(1, levels.shape[0] - 1)
    lo = hi - 1
    log_lower, log_upper = torch.gather(log_levels, 1, lo), torch.gather(log_levels, 1, hi)
    weight = (log_target - log_lower) / (log_upper - log_lower)
    return (1.0 - weight) * torch.gather(source, 1, lo) + weight * torch.gather(source, 1, hi)


def initialize_ufs_tracers(source: OzoneSource, atmosphere: PressureLevelSource,
                            dycore: dict[str, Tensor], *, extension: str = "reject", polar_extension: str = "reject",
                            negative_source: str = "reject") -> dict[str, Tensor]:
    """Add native dry-air ozone, zero ni/nr, and the original float32 MYNN seed.

    This is the existing R4 no-analyzed-cloud cold-start contract, not a complete
    UFS physics profile. Convert moist ozone at source points before remapping.
    Discrete stencil choices and endpoint clipping have no cross-boundary AD claim.
    Explicit legacy source policy preserves signed source samples, as the original
    adapter does. Both policies require finite, nonnegative initialized ozone.
    """
    if negative_source not in ("reject", "legacy"):
        raise ValueError("ozone negative source policy must be reject or legacy")
    if polar_extension not in ("reject", "clamp"):
        raise ValueError("ozone polar extension must be reject or clamp")
    if (source.grid_type == "reduced_gaussian") != (source.row_counts is not None) or source.grid_type not in (
            "regular_latlon", "rectilinear_latlon", "reduced_gaussian"):
        raise ValueError("ozone requires an explicit rectangular or reduced global grid")
    if not same_forecast_identity(source, atmosphere):
        raise ValueError("ozone and atmospheric forecast identities differ")
    if any(axis.ndim != 1 for axis in (source.latitude, source.longitude)) or source.pressure.ndim not in (1, 2, 3):
        raise ValueError("ozone requires horizontal axes and shared or collocated full-level pressure")
    shape = ((source.pressure.shape[0], source.latitude.shape[0], source.longitude.shape[0]) if source.row_counts is None
             else (source.pressure.shape[0], source.longitude.shape[0]))
    if source.ozone.shape != shape or min(shape) < 2:
        raise ValueError("incomplete ozone source shape")
    if source.pressure.ndim != 1 and source.pressure.shape != shape:
        raise ValueError("ozone full-level pressure must be collocated with source values")
    for value in (source.pressure, source.latitude, source.longitude, source.ozone):
        torch._assert_async(torch.isfinite(value).all(), "nonfinite ozone source")
    torch._assert_async((source.pressure > 0).all(), "invalid ozone pressure levels")
    if source.pressure.ndim > 1:
        dp = torch.diff(source.pressure, dim=0)
        torch._assert_async((dp > 0).all() | (dp < 0).all(), "ozone full-level pressure requires one strictly monotonic scan")
    dtype = torch.promote_types(source.ozone.dtype, torch.promote_types(source.latitude.dtype, source.longitude.dtype))
    lat = torch.sort(source.latitude.to(dtype)).values
    if lat.numel() < (2 if source.row_counts is None else 3):
        raise ValueError("ozone latitude requires complete global rows")
    torch._assert_async(((lat[1:] > lat[:-1]).all() & (lat >= -90).all() & (lat <= 90).all()
                         & (lat[0] + 90 <= lat[1] - lat[0] + 1.e-8)
                         & (90 - lat[-1] <= lat[-1] - lat[-2] + 1.e-8)), "ozone latitude lacks unique global coverage")
    if source.row_counts is None:
        lon = torch.sort(source.longitude.to(dtype)).values
        delta = lon[1:] - lon[:-1]
        torch._assert_async(((delta > 0) & ((delta - delta[0]).abs() <= 1.e-8)).all(), "ozone longitude must be regular and unique")
        torch._assert_async(((delta[0] * lon.shape[0] - 360).abs() <= 1.e-8), "ozone longitude must cover one periodic circuit")
    else:
        counts = source.row_counts
        if counts.shape != source.latitude.shape or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("ozone reduced row counts must be an integer vector matching latitude")
        torch._assert_async((counts >= 4).all() & (counts.sum() == source.longitude.numel()), "incomplete ozone reduced rows")
        rows = torch.repeat_interleave(torch.arange(lat.numel(), device=counts.device), counts, output_size=source.longitude.numel())
        starts = torch.cumsum(counts, 0) - counts
        point = torch.arange(source.longitude.numel(), device=counts.device)
        next_point = torch.where(point + 1 < starts[rows] + counts[rows], point + 1, starts[rows])
        lon = source.longitude.to(dtype).remainder(360.)
        gap = lon[next_point] - lon + torch.where(next_point == starts[rows], 360., 0.)
        epsilon = torch.finfo(source.longitude.dtype).eps if source.longitude.is_floating_point() else 0.
        tolerance = max(1.e-8, 720. * epsilon)
        torch._assert_async(((gap > 0) & ((gap - 360. / counts[rows]).abs() <= tolerance)).all(),
                            "ozone reduced rows must scan eastward over complete regular periodic circuits")
    if negative_source == "reject":
        torch._assert_async((source.ozone >= 0).all(), "negative ozone source")
    values = source.ozone
    if source.mass_basis == "moist_air":
        q = source.specific_humidity
        if q is None or q.shape != shape:
            raise ValueError("moist ozone requires collocated specific humidity")
        torch._assert_async((torch.isfinite(q) & (q >= 0) & (q < 1)).all(), "invalid ozone-source specific humidity")
        values = values / (1.0 - q)
    elif source.mass_basis != "dry_air" or source.specific_humidity is not None:
        raise ValueError("ozone requires an explicit dry_air or moist_air mass basis")
    for name in ("qc", "qr", "qi", "qs", "qg"):
        torch._assert_async((dycore[name] == 0).all(), "R4 cold-start tracer prefix requires the no-analyzed-cloud profile")
    pressure = mpas_layer_pressure(dycore["rho"], dycore["theta"], dycore["qv"])
    if (dycore["latCell"].ndim != 1 or dycore["lonCell"].shape != dycore["latCell"].shape
            or pressure.ndim != 2 or pressure.shape[0] != dycore["latCell"].numel()):
        raise ValueError("ozone target coordinates must match MPAS columns")
    degrees = 180.0 / torch.pi
    target_lat = dycore["latCell"] * degrees
    target_lon = dycore["lonCell"] * degrees
    pole = target_lat.new_tensor(torch.pi / 2., dtype=torch.float32).to(target_lat.dtype)
    torch._assert_async((torch.isfinite(target_lat) & torch.isfinite(target_lon) & (dycore["latCell"].abs() <= pole)).all(),
                        "invalid ozone target coordinates")
    target_lat = target_lat.clamp(-90., 90.)
    if polar_extension == "reject":
        torch._assert_async(((target_lat >= lat[0]) & (target_lat <= lat[-1])).all(), "ozone requires explicit polar extension")
    profiles = periodic_bilinear(values, source.latitude, source.longitude,
                                 target_lat, target_lon.remainder(360.0), row_counts=source.row_counts)
    source_pressure = source.pressure if source.pressure.ndim == 1 else periodic_bilinear(
        source.pressure, source.latitude, source.longitude, target_lat, target_lon.remainder(360.0), row_counts=source.row_counts)
    ozone = vertical_log_pressure(source_pressure, profiles, pressure, extension=extension)
    torch._assert_async((torch.isfinite(ozone) & (ozone >= 0)).all(), "invalid initialized ozone")
    seed = pressure.new_tensor(1.0e-4, dtype=torch.float32).to(pressure.dtype).expand_as(pressure)
    return {"o3mr": ozone, "ni": torch.zeros_like(pressure), "nr": torch.zeros_like(pressure), "sgs_tke": seed}
