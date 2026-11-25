"""First-order MPAS interpolation in physical height."""

import torch
from torch import Tensor


def geopotential_to_height(geopotential: Tensor) -> Tensor:
    """CF geopotential m2/s2 to geopotential metres, relative to the geoid.

    Standard gravity is 9.80665 m/s2, not the MPAS dynamical gravity constant.
    This does not convert geometric altitude or height above the local surface.
    """
    return geopotential / 9.80665


def hybrid_pressure(a: Tensor, b: Tensor, surface_pressure: Tensor,
                    reference_pressure: Tensor | None = None, *, interfaces: bool = False) -> Tensor:
    """CF a*p0+b*ps, or ap+b*ps when p0 is absent; pressures in Pa.

    Coefficients describe full levels by default. Explicit ``interfaces=True``
    permits zero pressure for a separately selected interface convention.
    Surface pressure has [latitude, longitude] axes. No level convention is inferred.
    """
    if a.ndim != 1 or a.shape != b.shape or a.numel() < 2 or surface_pressure.ndim not in (1, 2):
        raise ValueError("hybrid pressure requires full-level coefficient vectors and a surface pressure grid")
    if reference_pressure is not None and reference_pressure.ndim != 0:
        raise ValueError("hybrid reference pressure must be scalar")
    torch._assert_async((torch.isfinite(a) & torch.isfinite(b)).all(), "nonfinite hybrid coefficients")
    torch._assert_async((torch.isfinite(surface_pressure) & (surface_pressure > 0)).all(), "invalid hybrid surface pressure")
    if reference_pressure is not None:
        torch._assert_async(torch.isfinite(reference_pressure) & (reference_pressure > 0), "invalid hybrid reference pressure")
        a = a * reference_pressure
    dimensions = (a.shape[0],) + (1,) * surface_pressure.ndim
    pressure = a.reshape(dimensions) + b.reshape(dimensions) * surface_pressure
    torch._assert_async((torch.isfinite(pressure) & ((pressure >= 0) if interfaces else (pressure > 0))).all(),
                        "invalid hybrid interface pressure" if interfaces else "invalid hybrid full-level pressure")
    return pressure


def ifs_full_level_pressure(interface_pressure: Tensor) -> Tensor:
    """Explicit IFS arithmetic mean; complete interfaces scan from zero to surface."""
    if interface_pressure.ndim < 1 or interface_pressure.shape[0] < 3 or not interface_pressure.is_floating_point():
        raise ValueError("IFS pressure requires at least three floating interfaces")
    torch._assert_async(torch.isfinite(interface_pressure).all(), "nonfinite IFS interface pressure")
    torch._assert_async((interface_pressure[0] == 0).all(), "IFS reconstruction requires the zero-pressure top interface")
    torch._assert_async((torch.diff(interface_pressure, dim=0) > 0).all(), "IFS interfaces must increase from top to surface")
    return (interface_pressure[:-1] + interface_pressure[1:]) / 2.0


def ifs_hydrostatic_columns(interface_pressure: Tensor, temperature: Tensor, specific_humidity: Tensor,
                             surface_geopotential: Tensor) -> tuple[Tensor, Tensor]:
    """IFS arithmetic-mean pressure and vapour-only hydrostatic height in metres.

    Inputs scan top to bottom, with [level, ...column] axes and one more interface.
    Interface zero is the zero-pressure model top. The bottom is the surface.
    Temperature is K, specific humidity kg/kg, pressure Pa, and geopotential m2/s2.
    This is ECMWF's published approximate reconstruction, not the complete IFS
    condensate-loaded/nonhydrostatic state. The caller selects this convention.
    """
    if (temperature.ndim < 1 or temperature.shape[0] < 2 or specific_humidity.shape != temperature.shape
            or surface_geopotential.shape != temperature.shape[1:]
            or interface_pressure.shape != (temperature.shape[0] + 1, *temperature.shape[1:])):
        raise ValueError("IFS reconstruction requires complete collocated layers and surface geopotential")
    if any(not value.is_floating_point() or value.dtype != temperature.dtype or value.device != temperature.device
           for value in (interface_pressure, specific_humidity, surface_geopotential)):
        raise ValueError("IFS reconstruction requires matching floating precision and device")
    pressure = ifs_full_level_pressure(interface_pressure)
    torch._assert_async((torch.isfinite(temperature) & (temperature > 0)).all(), "invalid IFS temperature")
    torch._assert_async((torch.isfinite(specific_humidity) & (specific_humidity >= 0) & (specific_humidity < 1)).all(),
                        "invalid IFS specific humidity")
    torch._assert_async(torch.isfinite(surface_geopotential).all(), "nonfinite surface geopotential")
    half_height, full_height = surface_geopotential, []
    log_two = torch.log(temperature.new_tensor(2.0))
    # Keep the original bottom-up accumulation; the unused top half-level is not returned.
    for level in range(temperature.shape[0] - 1, -1, -1):
        above, below = interface_pressure[level], interface_pressure[level + 1]
        moist_temperature = temperature[level] * (1.0 + 0.609133 * specific_humidity[level])
        if level == 0:
            alpha = log_two
        else:
            dlog_p = torch.log(below / above)
            alpha = 1.0 - ((above / (below - above)) * dlog_p)
        moist_temperature = moist_temperature * 287.06
        full_height.append(half_height + (moist_temperature * alpha))
        if level > 0:
            half_height = half_height + (moist_temperature * dlog_p)
    return pressure, geopotential_to_height(torch.stack(full_height[::-1]))


def interpolate_height(height: Tensor, value: Tensor, target: Tensor,
                       extrapolation: str = "constant") -> Tensor:
    """Interpolate sorted columns [..., source_level] onto [..., target_level].

    Source heights must be strictly increasing. Sorting and representation
    validation precede this kernel. Lapse-rate extrapolation is below-ground
    only; an uncovered model top returns NaN and must fail output validation.
    """
    if extrapolation not in ("constant", "linear", "lapse-rate"):
        raise ValueError(f"unsupported extrapolation: {extrapolation}")
    lower = (torch.searchsorted(height.contiguous(), target.contiguous(), right=True) - 1).clamp(0, height.shape[-1] - 2)
    z0, z1 = torch.gather(height, -1, lower), torch.gather(height, -1, lower + 1)
    v0, v1 = torch.gather(value, -1, lower), torch.gather(value, -1, lower + 1)
    wm, wp = (z1 - target) / (z1 - z0), (target - z0) / (z1 - z0)
    interior = wm * v0 + wp * v1
    if extrapolation == "constant":
        below, above = value[..., :1], value[..., -1:]
    elif extrapolation == "linear":
        bottom_slope = (value[..., 1:2] - value[..., :1]) / (height[..., 1:2] - height[..., :1])
        top_slope = (value[..., -1:] - value[..., -2:-1]) / (height[..., -1:] - height[..., -2:-1])
        below = value[..., :1] + bottom_slope * (target - height[..., :1])
        above = value[..., -1:] + top_slope * (target - height[..., -1:])
    else:
        below = value[..., :1] - (target - height[..., :1]) * 0.0065
        above = torch.full_like(target, torch.nan)
    return torch.where(target < height[..., :1], below,
                       torch.where(target >= height[..., -1:], above, interior))
