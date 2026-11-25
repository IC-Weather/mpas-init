"""Native MPAS four-layer Noah surface cold start, with PyTorch numerics."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections.abc import Mapping

import torch
from torch import Tensor

from mpas_init.horizontal import MISSING, HorizontalGrid, four_point_sequence, global_latlon_indices


SOURCE_FIELDS = frozenset(("landmask", "surface_altitude", "skin_temperature", "snow_water_equivalent",
                           "snow_depth", "sea_ice_fraction", "soil_temperature", "soil_moisture"))
MASKED_SOURCE_FIELDS = frozenset(("soil_temperature", "soil_moisture", "sea_ice_fraction"))
STATIC_FIELDS = frozenset(("latCell", "lonCell", "ter", "landmask", "soiltemp", "ivgtyp", "isltyp",
                           "snoalb", "greenfrac", "albedo12m"))
SURFACE_FIELDS = frozenset(("tmn", "skintemp", "sst", "xice", "seaice", "snow", "snowc", "snowh",
                            "tslb", "smois", "sh2o", "smcrel", "zs", "dzs", "xland", "ivgtyp",
                            "isltyp", "snoalb", "vegfra", "sfc_albbck", "shdmin", "shdmax"))


@dataclass(frozen=True)
class SurfaceSource:
    """Global rectilinear fields and shared explicit soil bounds.

    Axes are degrees; soil bounds are metres below ground, [layer, lower/upper].
    Temperature is K, altitude/depth m, snow water kg m-2, soil moisture total
    volumetric water, and ice fraction unitless. Soil fields have axes
    [layer, latitude, longitude]; the other fields are [latitude, longitude].
    Missing support uses MISSING, not zero: zero snow/moisture is physical.
    Non-landmask fields may own grids. Independent soil/ice grids require their
    own collocated binary field_landmasks; masks are never resampled implicitly.
    The codec boundary validates coordinates, metadata, units and time identity.
    """

    latitude: Tensor
    longitude: Tensor
    soil_bounds: Tensor
    fields: dict[str, Tensor]
    valid_time: datetime
    reference_time: datetime | None = None
    lead: timedelta | None = None
    grid_type: str = "regular_latlon"
    member: int | None = None
    field_grids: dict[str, HorizontalGrid] | None = None
    field_landmasks: dict[str, Tensor] | None = None
    declarations: dict[str, dict[str, str | None]] | None = None


def wps_ncep_gfs_snow(swe: Tensor) -> tuple[Tensor, Tensor]:
    """Declared WPS 4.6 ungrib NCEP GFS snow: SWE doubles, then depth is 0.005 x that SWE.

    This follows rrpr.F (SNOW = 2*SNOW; then SNOWH = 0.005*SNOW, 200:1, for NCEP sources
    without a supplied SNOWH). Missing pixels stay missing in both fields.
    """
    missing = swe == MISSING
    doubled = torch.where(missing, swe, swe * 2.0)
    return doubled, torch.where(missing, swe, doubled * 0.005)


def snow_ocean_zero(field: Tensor, landmask: Tensor) -> Tensor:
    """Apply explicitly declared zero snow at missing water pixels, not land gaps."""
    if field.shape != landmask.shape or field.ndim != 2:
        raise ValueError("snow missing-ocean policy requires a collocated two-dimensional landmask")
    missing = field == MISSING
    torch._assert_async(((landmask == 0) | (landmask == 1)).all(), "snow policy requires a binary landmask")
    torch._assert_async(torch.isfinite(field).all(), "snow policy requires finite values or the missing sentinel")
    torch._assert_async((~missing | (landmask == 0)).all(), "snow zero policy cannot fill missing land pixels")
    return torch.where(missing, torch.zeros_like(field), field)


def soil_water_maximum(field: Tensor, maximum: float | None = None) -> Tensor:
    """Apply a declared volumetric-water cap, then validate physical source bounds."""
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not 0 < maximum <= 1):
        raise ValueError("soil maximum must be a finite number above zero and at most one")
    torch._assert_async(torch.isfinite(field).all(), "soil volumetric water must be finite or the missing sentinel")
    missing = field == MISSING
    value = field if maximum is None else torch.where(missing, torch.full_like(field, MISSING),
                                                     torch.minimum(field, field.new_tensor(maximum)))
    torch._assert_async((torch.isfinite(value) & (missing | ((value >= 0) & (value <= 1)))).all(),
                         "soil volumetric water must lie between zero and one; declare a justified maximum if required")
    return value


def monthly_interpolation(monthly: Tensor, valid_time: datetime) -> Tensor:
    """Original day-only interpolation between month fifteenths, including year wrap."""
    if monthly.shape[-1] != 12:
        raise ValueError("monthly surface climatology requires twelve months")
    year = valid_time.year
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    lengths = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    middle = [sum(lengths[:month]) + 15 for month in range(12)]
    middle = [middle[0] - 31, *middle, middle[-1] + 31]
    day = sum(lengths[:valid_time.month - 1]) + valid_time.day
    month = next(i for i in range(13) if middle[i] < day <= middle[i + 1])
    left, right = (11, 0) if month in (0, 12) else (month - 1, month)
    return ((monthly[..., right] * (day - middle[month])
             + monthly[..., left] * (middle[month + 1] - day)) / (middle[month + 1] - middle[month]))


def _surface_interpolation(field: Tensor, x: Tensor, y: Tensor, *, mask: Tensor | None = None,
                            excluded: float = 0.0, sixteen: bool = False, four: bool = True,
                            search: bool = True, active: Tensor | None = None) -> Tensor:
    if mask is not None:
        field = torch.where(mask == excluded, MISSING, field)
    # Preserve the original three-column periodic halo. SEARCH itself is bounded.
    field = torch.cat((field[:, -3:], field, field[:, :3]), dim=-1)
    return four_point_sequence(field, x + 1.0, y + 1.0, start_x=-2, start_y=1,
                               missing=MISSING, weighted16=False, search=search,
                               sixteen=sixteen, four=four, active=active)


def noah_soil_layers(temperature: Tensor, moisture: Tensor, bounds: Tensor, skin: Tensor,
                       deep: Tensor, landmask: Tensor) -> dict[str, Tensor]:
    """Native depth reconstruction and point interpolation onto four Noah midpoints.

    Inputs are cell-major. The native surface moisture uses source layer two;
    its three-metre moisture endpoint uses the penultimate source layer.
    Liquid soil water is a declared cold start: zero on land, one on ocean.
    Noah/Noah-MP reconstructs frozen/liquid partitioning at model startup.
    """
    layers = bounds.shape[0]
    if layers < 2 or bounds.shape != (layers, 2) or temperature.shape != moisture.shape or temperature.shape[-1] != layers:
        raise ValueError("soil initialization requires matching source layers and explicit bounds")
    cm = bounds * 100.0
    widths = cm[:, 1] - cm[:, 0]
    centers = [0.5 * widths[0]]
    for k in range(1, layers):
        centers.append(centers[-1] + 0.5 * widths[k - 1] + 0.5 * widths[k])
    source_z = torch.stack(centers) / 100.0
    source_z = torch.cat((source_z.new_zeros(1), source_z, source_z.new_tensor([3.0])))
    torch._assert_async((torch.isfinite(bounds).all() & (widths > 0).all()
                         & (source_z[1:] > source_z[:-1]).all() & (bounds[0, 0] == 0)
                         & (bounds[1:, 0] == bounds[:-1, 1]).all()), "invalid or unsupported soil depth coverage")
    thickness = bounds.new_tensor([0.10, 0.30, 0.60, 1.00])
    centers = [0.5 * thickness[0]]
    for k in range(1, 4):
        centers.append(centers[-1] + 0.5 * thickness[k - 1] + 0.5 * thickness[k])
    target_z = torch.stack(centers)
    st = torch.cat((skin[:, None], temperature, deep[:, None]), dim=-1)
    sm = torch.cat((moisture[:, 1:2], moisture, moisture[:, -2:-1]), dim=-1)
    lower = (torch.searchsorted(source_z, target_z, right=False) - 1).clamp(0, layers)
    a, b = source_z[lower], source_z[lower + 1]
    interpolate = lambda value: ((value[:, lower] * (b - target_z)
                                   + value[:, lower + 1] * (target_z - a)) / (b - a))
    land = landmask[:, None] == 1
    tslb = torch.where(land, interpolate(st), skin[:, None])
    smois = torch.where(land, interpolate(sm), 1.0)
    dry = land & (tslb[:, :1] > 170.0) & (tslb[:, :1] < 400.0) & (smois[:, :1] < 0.005)
    smois = torch.where(dry, 0.005, smois)
    return {"tslb": tslb, "smois": smois, "sh2o": torch.where(land, 0.0, 1.0).to(temperature.dtype).expand_as(tslb),
            "smcrel": torch.zeros_like(tslb), "zs": target_z.expand_as(tslb),
            "dzs": thickness.expand_as(tslb)}


def initialize_surface(source: SurfaceSource, static: Mapping[str, Tensor], *, ice_category: int = 15,
                         fractional_seaice: bool = True, skin_ice_threshold: float = 100.0) -> dict[str, Tensor]:
    """Initialize the native surface prefix, not a complete coupled-physics profile.

    Static terrain must be the FINAL smoothed terrain from initialize_dycore.
    Annual soil temperature retains its native sea-level convention. Source and
    target landmasks stay separate. No auxiliary SST/ice update file is implied.
    """
    if set(source.fields) != SOURCE_FIELDS or not STATIC_FIELDS <= set(static):
        raise ValueError("incomplete surface source or static field inventory")
    f = source.fields
    field_grids = {} if source.field_grids is None else source.field_grids
    field_landmasks = {} if source.field_landmasks is None else source.field_landmasks
    if set(field_grids) - SOURCE_FIELDS or "landmask" in field_grids:
        raise ValueError("surface field grids must name non-landmask source fields; landmask owns the primary grid")
    if set(field_landmasks) - MASKED_SOURCE_FIELDS:
        raise ValueError("surface field masks must name soil temperature, soil moisture, or sea ice")
    if MASKED_SOURCE_FIELDS & (set(field_grids) - set(field_landmasks)):
        raise ValueError("each independent masked surface grid requires its collocated landmask")
    s = {name: value.to(dtype=f["skin_temperature"].dtype) if value.is_floating_point() else value
         for name, value in static.items()}
    ny, nx = source.latitude.shape[0], source.longitude.shape[0]
    layers = source.soil_bounds.shape[0]
    if nx < 3 or ny < 2 or source.latitude.ndim != 1 or source.longitude.ndim != 1:
        raise ValueError("surface source requires a global rectilinear grid")
    for name, value in f.items():
        grid = field_grids.get(name)
        if grid is not None and (grid.row_counts is not None or grid.latitude.ndim != 1 or grid.longitude.ndim != 1
                                 or grid.latitude.numel() < 2 or grid.longitude.numel() < 3):
            raise ValueError("surface field requires a rectangular global grid: " + name)
        horizontal = (ny, nx) if grid is None else (grid.latitude.numel(), grid.longitude.numel())
        shape = (layers, *horizontal) if name.startswith("soil_") else horizontal
        if value.shape != shape:
            raise ValueError(f"invalid surface source shape: {name}")
        if name in field_landmasks:
            mask = field_landmasks[name]
            if mask.shape != horizontal:
                raise ValueError("surface field landmask must be collocated: " + name)
            torch._assert_async(((mask == 0) | (mask == 1)).all(), "source field landmask must be binary: " + name)
    nc = s["latCell"].shape[0]
    for name in sorted(STATIC_FIELDS):
        shape = (nc, 12) if name in ("greenfrac", "albedo12m") else (nc,)
        if s[name].shape != shape:
            raise ValueError(f"invalid surface static shape: {name}")
    land = s["landmask"] == 1
    torch._assert_async(((s["landmask"] == 0) | land).all(), "surface landmask must be binary")
    torch._assert_async(((f["landmask"] == 0) | (f["landmask"] == 1)).all(),
                         "source surface landmask must be binary")
    for name, value in f.items():
        torch._assert_async(torch.isfinite(value).all(), f"nonfinite surface source field: {name}")

    def require_support(value, name, active=True):
        torch._assert_async(((torch.isfinite(value) & (value != MISSING)) | ~torch.as_tensor(
            active, device=value.device)).all(), f"no valid required surface source support: {name}")

    x, y = global_latlon_indices(source.latitude, source.longitude, s["latCell"], s["lonCell"],
                                 grid_type=source.grid_type)
    coordinates = {name: global_latlon_indices(grid.latitude, grid.longitude, s["latCell"], s["lonCell"], grid_type=grid.grid_type)
                   for name, grid in field_grids.items()}

    def interpolate(name, value=None, **options):
        xx, yy = coordinates.get(name, (x, y))
        return _surface_interpolation(f[name] if value is None else value, xx, yy, **options)

    skin = interpolate("skin_temperature", sixteen=True, four=False)
    altitude = interpolate("surface_altitude", sixteen=True, four=False)
    snow = interpolate("snow_water_equivalent", search=False)
    snowh = interpolate("snow_depth", search=False)
    xice = interpolate("sea_ice_fraction", mask=field_landmasks.get("sea_ice_fraction", f["landmask"]), excluded=1.0, active=~land)
    for name, value in (("skin_temperature", skin), ("surface_altitude", altitude),
                        ("snow_water_equivalent", snow), ("snow_depth", snowh)):
        require_support(value, name)
    require_support(xice, "sea_ice_fraction", ~land)
    xice = torch.where(land, 0.0, xice).clamp(0.0, 1.0)
    soil_t, soil_m = [], []
    for k in range(layers):
        temp = interpolate("soil_temperature", f["soil_temperature"][k], mask=field_landmasks.get("soil_temperature", f["landmask"]), sixteen=True, active=land)
        moist = interpolate("soil_moisture", f["soil_moisture"][k], mask=field_landmasks.get("soil_moisture", f["landmask"]), active=land)
        require_support(temp, "soil_temperature", land)
        require_support(moist, "soil_moisture", land)
        soil_t.append(torch.where(land, temp, 285.0))
        soil_m.append(torch.where(land, torch.where(moist <= 0, 0.001, moist), 1.0))
    soil_t, soil_m = torch.stack(soil_t, -1), torch.stack(soil_m, -1)
    return _surface_from_interpolated(
        {"skin_temperature": skin, "surface_altitude": altitude, "snow_water_equivalent": snow,
         "snow_depth": snowh, "sea_ice_fraction": xice, "soil_temperature": soil_t, "soil_moisture": soil_m},
        source.soil_bounds, s, source.valid_time, ice_category=ice_category,
        fractional_seaice=fractional_seaice, skin_ice_threshold=skin_ice_threshold)


def _surface_from_interpolated(fields: Mapping[str, Tensor], bounds: Tensor, s: Mapping[str, Tensor],
                                 valid_time: datetime, *, ice_category: int = 15,
                                 fractional_seaice: bool = True, skin_ice_threshold: float = 100.0) -> dict[str, Tensor]:
    """Source-defined surface physics boundary after masked horizontal interpolation."""
    skin, altitude = fields["skin_temperature"], fields["surface_altitude"]
    snow, snowh = fields["snow_water_equivalent"], fields["snow_depth"]
    xice, soil_t, soil_m = fields["sea_ice_fraction"], fields["soil_temperature"], fields["soil_moisture"]
    land = s["landmask"] == 1
    # The legacy fallback applies only when the complete depth field is absent/zero.
    snowh = torch.where((snowh.amin() == 0) & (snowh.amax() == 0), snow * 5.0 / 1000.0, snowh)
    correction = 0.0065 * (s["ter"] - altitude)
    adjusted_skin = torch.where(land, skin - correction, skin)
    soil_t = torch.where(land[:, None], soil_t - correction[:, None], soil_t)
    deep = torch.where(land, s["soiltemp"] - 0.0065 * s["ter"], skin)
    soil = noah_soil_layers(soil_t, soil_m, bounds, adjusted_skin, deep, s["landmask"])
    albedo = monthly_interpolation(s["albedo12m"], valid_time) / 100.0
    vegetation = monthly_interpolation(s["greenfrac"], valid_time)
    if not fractional_seaice:
        xice = (xice >= 0.5).to(skin.dtype)
    threshold = 0.02 if fractional_seaice else 0.5
    ice = (xice >= threshold) | (~land & (adjusted_skin < skin_ice_threshold))
    deep = torch.where(ice & ~land, 271.4, deep)
    # Sea ice uses four equal subdivisions of a three-metre ice column, not Noah zs.
    midpoint = skin.new_tensor(3.0) / 4 / 2.0 + torch.arange(4, device=skin.device) * (skin.new_tensor(3.0) / 4)
    ice_t = ((3.0 - midpoint) * adjusted_skin[:, None] + midpoint * deep[:, None]) / 3.0
    soil.update({"tslb": torch.where(ice[:, None], ice_t, soil["tslb"]),
                 "smois": torch.where(ice[:, None], 1.0, soil["smois"]),
                 "sh2o": torch.where(ice[:, None], 0.0, soil["sh2o"])})
    result = {**soil, "tmn": deep, "skintemp": adjusted_skin, "sst": skin, "xice": xice,
            "seaice": ice.to(skin.dtype), "snow": torch.where(ice, snow.clamp_max(360.0), snow),
            "snowc": (snow >= 10.0).to(skin.dtype), "snowh": torch.where(ice, snowh.clamp_max(1.2), snowh),
            "ivgtyp": torch.where(ice, ice_category, s["ivgtyp"]),
            "isltyp": torch.where(ice, 16, s["isltyp"]), "snoalb": torch.where(ice, 0.75, s["snoalb"]),
            "xland": torch.where(ice | land, 1.0, 2.0).to(skin.dtype), "vegfra": torch.where(ice, 0.0, vegetation),
            "sfc_albbck": torch.where(land, albedo, 0.08), "shdmin": s["greenfrac"].amin(-1),
            "shdmax": s["greenfrac"].amax(-1)}
    for name, value in result.items():
        torch._assert_async((torch.isfinite(value) & (value != MISSING)).all(), f"invalid initialized surface field: {name}")
    for name in ("tslb", "tmn", "skintemp", "sst"):
        torch._assert_async((result[name] > 0).all(), f"nonpositive initialized surface temperature: {name}")
    return result
