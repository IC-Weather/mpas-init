"""Small CF atmosphere and surface codecs. Numerical transforms remain in PyTorch."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from numbers import Integral
import re

from netCDF4 import Dataset, num2date
import numpy as np
import torch
from torch import Tensor

from mpas_init.surface import MISSING, MASKED_SOURCE_FIELDS, SurfaceSource, snow_ocean_zero, soil_water_maximum, wps_ncep_gfs_snow
from mpas_init.horizontal import HorizontalGrid
from mpas_init.vertical import geopotential_to_height, hybrid_pressure, ifs_full_level_pressure, ifs_hydrostatic_columns
from mpas_init.thermodynamics import wps_ice_blend_to_liquid

SourcePaths = str | Path | Sequence[str | Path]

REQUIRED = {"air_temperature": "K", "eastward_wind": "m s-1", "northward_wind": "m s-1",
            "specific_humidity": "kg kg-1", "geopotential_height": "m"}
SURFACE_VARIABLES = {
    "landmask": ("land_binary_mask", "1"), "surface_altitude": ("surface_altitude", "m"),
    "skin_temperature": ("surface_temperature", "K"),
    "snow_water_equivalent": ("surface_snow_amount", "kg m-2"),
    "snow_depth": ("surface_snow_thickness", "m"), "sea_ice_fraction": ("sea_ice_area_fraction", "1"),
    "soil_temperature": ("soil_temperature", "K"),
    "soil_moisture": ("volume_fraction_of_condensed_water_in_soil", "1"),
}
UFS_SURFACE_VARIABLES = {
    "aerodynamic_roughness_length": "m",
    **{name: "1" for name in ("visible_strong_cosine_albedo", "near_ir_strong_cosine_albedo",
                                "visible_weak_cosine_albedo", "near_ir_weak_cosine_albedo",
                                "wind_speed_ratio_10m_to_lowest_model_level", "surface_heat_transfer_function",
                                "surface_momentum_transfer_function")},
    "canopy_water_amount": "kg m-2", "friction_velocity": "m s-1", "sea_ice_thickness": "m",
    "specific_humidity_2m": "kg kg-1", "air_temperature_2m": "K",
    "deep_soil_temperature": "K", "sea_ice_surface_temperature": "K",
}
# CF 1.11 sections 4.1/4.2 list these spellings as equivalent; other units remain unsupported.
LATITUDE_UNITS = ("degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN")
LONGITUDE_UNITS = ("degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE")
STANDARD_CALENDARS = ("standard", "gregorian", "proleptic_gregorian")


def _humidity_reference(variable, mapping):
    declared = getattr(variable, "mpas_init_saturation_reference", None)
    explicit = mapping.get("relative_humidity.saturation_reference")
    if explicit is not None and declared is not None and explicit != declared:
        raise ValueError("conflicting relative humidity saturation reference declarations")
    reference = explicit if explicit is not None else declared
    if reference not in ("liquid_water", "wps_ice_blend"):
        raise ValueError("relative humidity requires explicit liquid_water or wps_ice_blend saturation reference")
    return reference


def _normalize_humidity(fields, field_grids, primary, reference):
    if reference == "liquid_water":
        return
    temperature = field_grids.get("air_temperature", primary)
    humidity = field_grids.get("relative_humidity", primary)
    counts = ((temperature.row_counts is None and humidity.row_counts is None)
              or (temperature.row_counts is not None and humidity.row_counts is not None
                  and torch.equal(temperature.row_counts, humidity.row_counts)))
    if (temperature.grid_type != humidity.grid_type or not counts
            or not torch.equal(temperature.latitude, humidity.latitude)
            or not torch.equal(temperature.longitude, humidity.longitude)):
        raise ValueError("RH phase conversion requires collocated temperature and humidity grids")
    fields["relative_humidity"] = wps_ice_blend_to_liquid(fields["air_temperature"], fields["relative_humidity"])


def _soil_maximum(variable, mapping):
    standard = "volume_fraction_of_condensed_water_in_soil"
    def parse(value):
        if value is None:
            return None
        try:
            maximum = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("soil maximum must be a finite number above zero and at most one") from exc
        if isinstance(value, bool) or not 0 < maximum <= 1:
            raise ValueError("soil maximum must be a finite number above zero and at most one")
        return maximum
    declared = parse(getattr(variable, "mpas_init_maximum", None))
    explicit = parse(mapping.get(standard + ".maximum"))
    if declared is not None and explicit is not None and declared != explicit:
        raise ValueError("conflicting soil maximum declarations")
    return explicit if explicit is not None else declared


def _snow_policy(variable, mapping, standard):
    declared = getattr(variable, "mpas_init_missing_ocean", None)
    explicit = mapping.get(standard + ".missing_ocean")
    if explicit is not None and declared is not None and explicit != declared:
        raise ValueError("conflicting snow missing-ocean policy declarations")
    policy = explicit if explicit is not None else declared
    if policy not in (None, "zero"):
        raise ValueError("snow missing-ocean policy must be zero when declared")
    return policy


def _snow_convention(mapping):
    """Return an explicitly declared SWE/depth producer convention, or None."""
    convention = mapping.get("surface_snow_amount.convention")
    if convention is None:
        return None
    if convention != "wps_ncep_gfs":
        raise ValueError("declared surface_snow_amount.convention must be wps_ncep_gfs")
    if any(key == "surface_snow_thickness" or key.startswith("surface_snow_thickness.") for key in mapping):
        raise ValueError("declared surface_snow_amount.convention=wps_ncep_gfs derives snow depth; "
                         "a mapped snow depth conflicts")
    return convention


def _apply_snow_convention(fields, field_grids, field_landmasks, convention):
    """Apply the declared convention after missing-ocean filling; depth shares the SWE grid."""
    if convention is None:
        return None
    fields["snow_water_equivalent"], fields["snow_depth"] = wps_ncep_gfs_snow(fields["snow_water_equivalent"])
    if "snow_water_equivalent" in field_grids:
        field_grids["snow_depth"] = field_grids["snow_water_equivalent"]
    if field_landmasks is not None and "snow_water_equivalent" in field_landmasks:
        field_landmasks["snow_depth"] = field_landmasks["snow_water_equivalent"]
    return {"surface_snow_amount.convention": {"declared": convention, "source": None}}


def _normalize_surface_snow(fields, field_grids, primary, policies):
    for name, policy in policies.items():
        if policy is None:
            continue
        grid = field_grids.get(name, primary)
        if (grid.grid_type != primary.grid_type or not torch.equal(grid.latitude, primary.latitude)
                or not torch.equal(grid.longitude, primary.longitude)):
            raise ValueError("snow missing-ocean policy requires collocated snow and landmask grids")
        fields[name] = snow_ocean_zero(fields[name], fields["landmask"])


@dataclass(frozen=True)
class PressureLevelSource:
    """Atmospheric full levels; pressure is a Pa vector or collocated 3-D grid.

    The historical name remains compatible with existing pressure-level callers.
    Fields have [level, latitude, longitude] axes, including supplied height in m.
    Pressure and height own the primary grid. Optional field_grids override other
    fields' horizontal coordinates without changing their shared vertical levels.
    Longitude scans eastward from [0, 360); latitude scans north to south.
    Codecs reorder source scans without resampling.
    """

    pressure: Tensor
    latitude: Tensor
    longitude: Tensor
    fields: dict[str, Tensor]
    valid_time: datetime
    reference_time: datetime
    lead: timedelta
    grid_type: str = "regular_latlon"
    member: int | None = None
    vertical_convention: str | None = None
    field_grids: dict[str, HorizontalGrid] | None = None
    row_counts: Tensor | None = None
    field_surface_pressure: dict[str, Tensor] | None = None


@dataclass(frozen=True)
class NearSurfaceSource:
    """Complete 2m temperature/liquid-water RH and earth-relative 10m winds.

    Values use K, percent, and m s-1; axes use degrees. Fields are point
    samples with [latitude, longitude] axes on complete global rectilinear grids.
    Optional field_grids override individual fields without intermediate remapping.
    """

    latitude: Tensor
    longitude: Tensor
    fields: dict[str, Tensor]
    valid_time: datetime
    reference_time: datetime
    lead: timedelta
    grid_type: str = "regular_latlon"
    member: int | None = None
    field_grids: dict[str, HorizontalGrid] | None = None


@dataclass(frozen=True)
class OzoneSource:
    """Point ozone samples [pressure, latitude, longitude], in kg/kg.

    mass_basis is dry_air (native mixing ratio) or moist_air (mass fraction).
    Moist fractions require collocated specific humidity, also in kg/kg.
    Pressure is a Pa vector or a full-level tensor collocated with ozone.
    Coordinates use degrees; identity must match the atmosphere.
    Reduced rows store [pressure, point] values with one row count per latitude.
    """

    pressure: Tensor
    latitude: Tensor
    longitude: Tensor
    ozone: Tensor
    mass_basis: str
    specific_humidity: Tensor | None
    valid_time: datetime
    reference_time: datetime
    lead: timedelta
    member: int | None = None
    grid_type: str = "regular_latlon"
    row_counts: Tensor | None = None
    vertical_convention: str | None = None


@dataclass(frozen=True)
class UfsSurfaceSource:
    """Explicit analyzed point quantities on global rectilinear grids.

    Fields use UFS_SURFACE_VARIABLES names and units, with [latitude, longitude]
    axes in degrees. Optional field_grids override each field's coordinates.
    This is a target convention, not an inferred producer schema.
    Optional declarations record explicit mapping overrides of source metadata,
    keyed by mapping key, as {"declared": value, "source": original label}.
    """

    latitude: Tensor
    longitude: Tensor
    fields: dict[str, Tensor]
    valid_time: datetime
    reference_time: datetime
    lead: timedelta
    member: int | None = None
    field_grids: dict[str, HorizontalGrid] | None = None
    declarations: dict[str, dict[str, str | None]] | None = None


def _same_instant(left: datetime, right: datetime) -> bool:
    # Python ignores fold and UTC offsets when both dates share a tzinfo object.
    # Integer attributes let Dynamo capture the decoder's UTC metadata checks.
    if left.tzinfo is right.tzinfo:
        return (left.year, left.month, left.day, left.hour, left.minute, left.second, left.microsecond) == (
            right.year, right.month, right.day, right.hour, right.minute, right.second, right.microsecond)
    return left == right


def same_forecast_identity(left, right) -> bool:
    """Compare source identity without graph breaks for the codecs' UTC dates.

    Other timezone pairs retain Python's inter-zone datetime equality semantics.
    Timedelta attributes are normalized integers, including microseconds.
    """
    return (_same_instant(left.valid_time, right.valid_time)
            and _same_instant(left.reference_time, right.reference_time)
            and (left.lead.days, left.lead.seconds, left.lead.microseconds) == (
                right.lead.days, right.lead.seconds, right.lead.microseconds)
            and left.member == right.member)


def _find_variable(ds, mapping, standard, dimensions=None, *, ndim=None, kind=None):
    if standard in mapping:
        name = mapping[standard]
        if name not in ds.variables:
            raise ValueError(f"mapped variable is absent: {standard} -> {name}")
        return ds[name]
    attribute = "mpas_init_quantity" if kind == "ufs_surface" and standard in UFS_SURFACE_VARIABLES else "standard_name"
    matches = [v for v in ds.variables.values() if getattr(v, attribute, "") == standard
               and (ndim is None or v.ndim == ndim)
               and (dimensions is None or (len(v.dimensions) == len(dimensions) and set(v.dimensions) == set(dimensions)))]
    if len(matches) != 1:
        raise ValueError(f"expected one {standard} variable, found {len(matches)}")
    return matches[0]


def _array(var, *, masked=False):
    value = var[:]
    if ((not masked and np.ma.is_masked(value))
            or not np.isfinite(np.ma.asarray(value).compressed()).all()):
        raise ValueError(f"missing or nonfinite values in {var.name}")
    return np.asarray(np.ma.asarray(value, dtype=np.float64).filled(MISSING))


def _calendar(var, declared=None):
    """Return the decoding calendar; a declaration may only relabel identical civil dates."""
    source = getattr(var, "calendar", "standard")
    if declared is None:
        if source not in STANDARD_CALENDARS:
            raise ValueError("unsupported source calendar")
        return source
    if declared not in STANDARD_CALENDARS:
        raise ValueError("declared time.calendar must be standard, gregorian or proleptic_gregorian")
    if source != declared:
        if not hasattr(var, "units"):
            raise ValueError(f"missing time units: {var.name}")
        values = _array(var).reshape(-1)
        try:
            original = np.atleast_1d(num2date(values, var.units, calendar=str(source).lower()))
        except ValueError as exc:
            raise ValueError("declared time.calendar cannot be checked against the source calendar") from exc
        relabelled = np.atleast_1d(num2date(values, var.units, calendar=declared))
        label = lambda d: (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond)
        if any(label(a) != label(b) for a, b in zip(original, relabelled)):
            raise ValueError("declared time.calendar conflicts with the source calendar's civil time labels")
    return declared


def _instant(var, calendar=None):
    calendar = _calendar(var, calendar)
    value = _array(var)
    if value.size != 1 or not hasattr(var, "units"):
        raise ValueError(f"select one time before decoding {var.name}")
    date = num2date(value.reshape(-1)[0], var.units, calendar=calendar)
    return datetime(date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond, timezone.utc)


def _forecast_identity(ds, mapping, calendar=None):
    time, reference_variable = (_find_variable(ds, mapping, name) for name in ("time", "forecast_reference_time"))
    # Mapping the reference time to the valid-time variable itself declares an analysis: lead zero.
    analysis = "time" in mapping and mapping.get("forecast_reference_time") == mapping["time"]
    period = None
    if not analysis or "forecast_period" in mapping or any(
            getattr(v, "standard_name", "") == "forecast_period" for v in ds.variables.values()):
        period = _find_variable(ds, mapping, "forecast_period")
    variables = [time, reference_variable] + ([] if period is None else [period])
    if time.ndim != 1:
        raise ValueError("the atmospheric decoder requires one time dimension")
    valid, reference = _instant(time, calendar), _instant(reference_variable, calendar)
    if period is None:
        lead = timedelta(0)
    elif getattr(period, "units", "") not in ("seconds", "hours") or _array(period).size != 1:
        raise ValueError("unsupported forecast period")
    else:
        lead = timedelta(seconds=float(_array(period).reshape(-1)[0]) * (3600 if period.units == "hours" else 1))
    if lead < timedelta(0) or reference + lead != valid:
        raise ValueError("forecast reference time plus lead differs from valid time")
    member = None
    if "realization" in mapping or any(getattr(v, "standard_name", "") == "realization" for v in ds.variables.values()):
        variable = _find_variable(ds, mapping, "realization")
        value = variable[:]
        if (value.size != 1 or np.ma.is_masked(value) or value.dtype.kind not in "iuf"
                or not np.isfinite(value).all() or getattr(variable, "units", "1") != "1"
                or not np.equal(value, np.floor(value)).all()):
            raise ValueError("select one integer realization before decoding")
        member = int(value.reshape(-1)[0])
        variables.append(variable)
    return time, (valid, reference, lead, member), {v.name for v in variables}


@dataclass(frozen=True)
class _NetcdfCollection:
    variables: dict
    dimensions: dict

    def __getitem__(self, name):
        return self.variables[name]


class _SelectedVariable:
    """Read only the selected hyperslab while retaining NetCDF metadata."""

    def __init__(self, variable, indices):
        self.variable = variable
        self.indices = tuple(indices.get(dim, slice(None)) for dim in variable.dimensions)
        self.dimensions = tuple(dim for dim in variable.dimensions if not isinstance(indices.get(dim), int))
        self.ndim = len(self.dimensions)
        self.shape = tuple(1 if isinstance(indices.get(dim), slice) else size
                           for dim, size in zip(variable.dimensions, variable.shape)
                           if not isinstance(indices.get(dim), int))
        self.size = int(np.prod(self.shape))

    def __getattr__(self, name):
        return getattr(self.variable, name)

    def __getitem__(self, key):
        value = self.variable[self.indices]
        return value if key == slice(None) else value[key]


def _utc_selection(value, name):
    if value is not None:
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError(f"{name} selection must be a timezone-aware datetime")
        value = value.astimezone(timezone.utc)
    return value


def _clock_dates(variable, calendar=None):
    calendar = _calendar(variable, calendar)
    if not hasattr(variable, "units"):
        raise ValueError(f"missing time units: {variable.name}")
    values = _array(variable)
    dates = num2date(values, variable.units, calendar=calendar)
    return np.asarray([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, timezone.utc)
                       for d in np.asarray(dates).reshape(-1)], dtype=object).reshape(values.shape)


def _select_dataset(ds, mapping, *, valid_time=None, reference_time=None, member=None, calendar=None):
    has_time = "time" in mapping or any(getattr(v, "standard_name", "") == "time" for v in ds.variables.values())
    if not has_time:
        return ds
    time = _find_variable(ds, mapping, "time")
    has_member = "realization" in mapping or any(getattr(v, "standard_name", "") == "realization" for v in ds.variables.values())
    realization = _find_variable(ds, mapping, "realization") if has_member else None
    # Leave the established singleton-file path unchanged unless a selection is needed.
    if (valid_time is None and reference_time is None and member is None and time.ndim == 1 and time.size == 1
            and (realization is None or realization.ndim == 0)):
        return ds
    indices = {}
    if realization is not None:
        values = realization[:]
        if (realization.ndim > 1 or np.ma.is_masked(values) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all() or getattr(realization, "units", "1") != "1"
                or not np.equal(values, np.floor(values)).all()):
            raise ValueError("select one integer realization before decoding")
        labels = [int(value) for value in values.reshape(-1)]
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate realization labels")
        if member is None and len(labels) != 1:
            raise ValueError("select one integer realization before decoding")
        selected = labels[0] if member is None else member
        matches = [i for i, label in enumerate(labels) if label == selected]
        if len(matches) != 1:
            raise ValueError(f"requested realization is absent: {selected}")
        if realization.ndim:
            dimension = realization.dimensions[0]
            if dimension in time.dimensions:
                raise ValueError("realization axis must be independent of forecast time")
            indices[dimension] = matches[0]
    elif member is not None:
        raise ValueError("requested realization requires a source realization coordinate")
    if time.ndim not in (1, 2) or len(set(time.dimensions)) != time.ndim:
        raise ValueError("selection requires one or two independent forecast time dimensions")
    dates = _clock_dates(time, calendar)
    if valid_time is None and dates.size != 1:
        raise ValueError("select one valid time before decoding")
    matches = np.ones(dates.shape, dtype=bool) if valid_time is None else dates == valid_time
    if reference_time is not None:
        reference = _find_variable(ds, mapping, "forecast_reference_time")
        if any(dim not in time.dimensions for dim in reference.dimensions):
            raise ValueError("forecast cycle dimensions must belong to the valid-time coordinate")
        order = tuple(reference.dimensions.index(dim) for dim in time.dimensions if dim in reference.dimensions)
        shape = tuple(len(ds.dimensions[dim]) if dim in reference.dimensions else 1 for dim in time.dimensions)
        references = _clock_dates(reference, calendar).transpose(order).reshape(shape)
        matches &= references == reference_time
    positions = np.argwhere(matches)
    if len(positions) != 1:
        raise ValueError("requested valid time/cycle is absent or ambiguous; select a unique forecast cycle")
    for dim, position in zip(time.dimensions, positions[0]):
        position = int(position)
        indices[dim] = slice(position, position + 1) if dim == time.dimensions[-1] else position
    dimensions = {name: range(1) if isinstance(indices.get(name), slice) else dimension
                  for name, dimension in ds.dimensions.items() if not isinstance(indices.get(name), int)}
    variables = {name: _SelectedVariable(var, indices) for name, var in ds.variables.items()}
    return _NetcdfCollection(variables, dimensions)


@contextmanager
def _open_netcdf(paths: SourcePaths, mapping, *, valid_time=None, reference_time=None, member=None, calendar=None):
    valid_time = _utc_selection(valid_time, "valid time")
    reference_time = _utc_selection(reference_time, "forecast cycle")
    if member is not None and (isinstance(member, bool) or not isinstance(member, Integral)):
        raise ValueError("realization selection must be an integer label")
    paths = (paths,) if isinstance(paths, (str, Path)) else tuple(paths)
    if not paths:
        raise ValueError("at least one source file is required")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate source file")
    with ExitStack() as stack:
        datasets = [_select_dataset(stack.enter_context(Dataset(path)), mapping, valid_time=valid_time,
                                     reference_time=reference_time, member=member, calendar=calendar) for path in resolved]
        if len(datasets) == 1:
            yield datasets[0]
            return
        shared, static = set(), set()
        for ds in datasets:
            for name, var in ds.variables.items():
                if var.dimensions == (name,):
                    shared.add(name)
                    static.add(name)
                if (getattr(var, "standard_name", "") in ("latitude", "longitude") and var.ndim == 1
                        and var.dimensions[0] in ds.variables
                        and hasattr(ds[var.dimensions[0]], "compress")):
                    static.add(name)
                for attribute in ("coordinates", "bounds", "grid_mapping"):
                    names = getattr(var, attribute, "").split()
                    shared.update(names)
                    if attribute != "coordinates":
                        static.update(names)
                terms = re.findall(r"([A-Za-z_]\w*)\s*:\s*([^\s:]+)", getattr(var, "formula_terms", ""))
                shared.update(value for _, value in terms)
                static.update(value for key, value in terms if key in ("a", "ap", "b", "p0"))
        variables, dimensions, clocks, identity = {}, {}, set(), None
        semantic_attributes = ("standard_name", "units", "calendar", "positive", "axis", "bounds", "formula_terms",
                               "coordinates", "cell_methods", "mpas_init_mass_basis", "mpas_init_quantity",
                               "mpas_init_saturation_reference", "compress")
        for path, ds in zip(resolved, datasets):
            has_clock = any(getattr(v, "standard_name", "") in ("time", "forecast_reference_time", "forecast_period", "realization")
                            for v in ds.variables.values()) or any(mapping.get(key) in ds.variables for key in (
                                "time", "forecast_reference_time", "forecast_period", "realization"))
            timeless = not has_clock and all(name in static and var.ndim <= 2 for name, var in ds.variables.items())
            file_clocks = set()
            if not timeless:
                try:
                    _, current, file_clocks = _forecast_identity(ds, mapping, calendar)
                except ValueError as exc:
                    raise ValueError(f"{path.name}: {exc}") from exc
                if identity is not None and identity != current:
                    raise ValueError(f"forecast time/member identities differ across source files: {path.name}")
                identity = current
            for name, dim in ds.dimensions.items():
                if name in dimensions and len(dim) != len(dimensions[name]):
                    raise ValueError(f"conflicting source dimension {name}: {path.name}")
                dimensions.setdefault(name, dim)
            for name, var in ds.variables.items():
                if name in variables:
                    previous = variables[name]
                    if name in clocks and name in file_clocks:
                        continue
                    if name not in shared:
                        raise ValueError(f"duplicate source field {name}: {path.name}")
                    if (var.dimensions != previous.dimensions
                            or any(getattr(var, key, None) != getattr(previous, key, None) for key in semantic_attributes)
                            or not np.array_equal(np.ma.getmaskarray(var[:]), np.ma.getmaskarray(previous[:]))
                            or not np.array_equal(np.asarray(var[:]), np.asarray(previous[:]))):
                        raise ValueError(f"conflicting shared source coordinate or vertical metadata {name}: {path.name}")
                else:
                    variables[name] = var
            clocks.update(file_clocks)
        if identity is None:
            raise ValueError("source collection lacks forecast identity")
        yield _NetcdfCollection(variables, dimensions)


def _coordinate_variable(ds, mapping, standard, units, *, kind, owner=None):
    if owner is None or standard in mapping:
        var = _find_variable(ds, mapping, standard, kind=kind)
    else:
        candidates = [v for v in ds.variables.values() if getattr(v, "standard_name", "") == standard
                      and v.ndim == 1 and v.dimensions[0] in owner.dimensions]
        associated = [v for v in candidates if v.name in getattr(owner, "coordinates", "").split()]
        matches = associated or candidates
        if len(matches) != 1:
            raise ValueError(f"expected one {standard} coordinate for {owner.name}, found {len(matches)}")
        var = matches[0]
    if (var.ndim != 1 or getattr(var, "units", None) not in units
            or owner is not None and var.dimensions[0] not in owner.dimensions):
        raise ValueError(f"unsupported coordinate or units: {standard}")
    return var


def _coordinate(ds, mapping, standard, units, *, kind, owner=None, monotonic=True):
    var = _coordinate_variable(ds, mapping, standard, units, kind=kind, owner=owner)
    values = _array(var)
    if values.size < 2 or monotonic and not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
        raise ValueError(f"coordinate must be strictly monotonic: {standard}")
    return var, values


def _reduced_coordinates(ds, lat, lon, lat_values, lon_values):
    """Decode CF 5.3/8.2 gathering; reorder complete rows without resampling."""
    point_dim = lat.dimensions[0]
    if point_dim not in ds.variables:
        raise ValueError("reduced coordinates require a CF gathering index")
    gathered = ds[point_dim]
    dimensions = getattr(gathered, "compress", "").split()
    if (gathered.dimensions != (point_dim,) or len(dimensions) != 2 or len(set(dimensions)) != 2
            or any(d not in ds.dimensions or d == point_dim for d in dimensions)):
        raise ValueError("reduced coordinates require two declared CF gathering dimensions")
    nrows, width = (len(ds.dimensions[d]) for d in dimensions)
    indices = gathered[:]
    if (indices.dtype.kind not in "iu" or np.ma.is_masked(indices) or nrows < 3 or width < 4
            or np.any(indices < 0) or np.any(indices >= nrows*width)
            or np.unique(indices).size != indices.size):
        raise ValueError("invalid or duplicate reduced gathering indices")
    row_ids = np.asarray(indices) // width
    counts = np.bincount(row_ids.astype(np.int64), minlength=nrows)
    if np.any(counts < 4):
        raise ValueError("every reduced row requires at least four periodic points")
    row_latitude, rows = [], []
    canonical_lon = np.remainder(lon_values, 360.)
    grouped = np.argsort(row_ids, kind="stable")
    bounds = np.r_[0, np.cumsum(counts)]
    for row in range(nrows):
        members = grouped[bounds[row]:bounds[row+1]]
        if not np.all(lat_values[members] == lat_values[members[0]]):
            raise ValueError("gathered row must have one constant latitude")
        row_latitude.append(lat_values[members[0]])
        ordered = members[np.argsort(canonical_lon[members])]
        values = canonical_lon[ordered]
        # Each gap contains rounding from two stored coordinates; retain their values.
        epsilon = np.finfo(lon.dtype).eps if lon.dtype.kind == "f" else 0.
        raw = np.abs(lon_values[ordered])
        tolerance = np.maximum(1e-8, epsilon*(raw+np.roll(raw,-1)))
        if np.any(np.abs(np.diff(np.r_[values, values[0]+360])-360./members.size) > tolerance):
            raise ValueError("reduced longitude row must cover one complete regular periodic circuit")
        rows.append(ordered)
    row_latitude = np.asarray(row_latitude)
    order = np.argsort(-row_latitude)
    latitude = row_latitude[order]
    if (np.any(np.diff(latitude) >= 0) or latitude[0] > 90 or latitude[-1] < -90
            or 90-latitude[0] > latitude[0]-latitude[1]+1e-8
            or latitude[-1]+90 > latitude[-2]-latitude[-1]+1e-8):
        raise ValueError("reduced latitude lacks unique globally covered rows")
    points = np.concatenate([rows[i] for i in order])
    return lat, lon, latitude, canonical_lon[points], points, None, "reduced_gaussian", counts[order]


def _global_coordinates(ds, mapping, *, kind, owner=None, cache=None):
    if cache is not None:
        lat = _coordinate_variable(ds, mapping, "latitude", LATITUDE_UNITS, kind=kind, owner=owner)
        lon = _coordinate_variable(ds, mapping, "longitude", LONGITUDE_UNITS, kind=kind, owner=owner)
        key = (kind, lat.name, lon.name)
        if key not in cache:
            cache[key] = _global_coordinates(ds, mapping, kind=kind, owner=owner)
        return cache[key]
    lat, lat_values = _coordinate(ds, mapping, "latitude", LATITUDE_UNITS, kind=kind, owner=owner, monotonic=False)
    lon, lon_values = _coordinate(ds, mapping, "longitude", LONGITUDE_UNITS, kind=kind, owner=owner, monotonic=False)
    if lat.dimensions == lon.dimensions:
        if kind not in ("atmosphere", "ozone"):
            raise ValueError("reduced coordinates require atmospheric or ozone fields")
        return _reduced_coordinates(ds, lat, lon, lat_values, lon_values)
    for name, values in (("latitude", lat_values), ("longitude", lon_values)):
        if not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
            raise ValueError(f"coordinate must be strictly monotonic: {name}")
    native_grid = kind in ("pressure", "atmosphere", "surface", "near_surface")
    point_grid = kind in ("ufs_surface", "ozone")
    regular_latitude = np.allclose(np.diff(lat_values), lat_values[1] - lat_values[0], rtol=0, atol=1e-8)
    grid_type = "regular_latlon"
    for var, values in ((lon, lon_values),) if point_grid or native_grid else ((lat, lat_values), (lon, lon_values)):
        if not np.allclose(np.diff(values), values[1] - values[0], rtol=0, atol=1e-8):
            raise ValueError(f"non-regular source coordinate: {var.name}")
    if point_grid or native_grid and not regular_latitude:
        ordered = np.sort(lat_values)
        if (ordered.size < (2 if kind == "ozone" else 3) or ordered[0] < -90 or ordered[-1] > 90
                or ordered[0] + 90 > ordered[1] - ordered[0] + 1.e-8
                or 90 - ordered[-1] > ordered[-1] - ordered[-2] + 1.e-8):
            raise ValueError("UFS surface latitude lacks global polar coverage" if kind == "ufs_surface"
                             else "nonuniform latitude lacks global polar coverage")
        if not regular_latitude:
            grid_type = "rectilinear_latlon"
    elif not np.allclose(sorted([lat_values[0], lat_values[-1]]), [-90, 90], rtol=0, atol=1e-8):
        raise ValueError("latitude must cover both poles")
    if abs(abs(lon_values[1] - lon_values[0]) * lon_values.size - 360) > 1e-8:
        raise ValueError("longitude must cover one complete non-repeated periodic grid")
    longitude_order, latitude_order = np.arange(lon_values.size), np.arange(lat_values.size)
    if native_grid:
        canonical_lon = np.remainder(lon_values, 360.)
        longitude_order, latitude_order = np.argsort(canonical_lon), np.argsort(-lat_values)
        lon_values, lat_values = canonical_lon[longitude_order], lat_values[latitude_order]
    return lat, lon, lat_values, lon_values, latitude_order, longitude_order, grid_type, None


def _read_regular(path: SourcePaths, mapping: dict[str, str] | None, *, kind: str,
                    include_relative_humidity: bool = False, hybrid_convention: str | None = None,
                    valid_time=None, reference_time=None, member=None):
    """Shared coordinate, identity, and field checks for the first CF codecs."""
    mapping = {} if mapping is None else mapping
    convention = {"atmosphere": "ifs-vapour-only", "ozone": "ifs-arithmetic"}.get(kind)
    if hybrid_convention is not None and (convention is None or hybrid_convention != convention):
        raise ValueError("unsupported hybrid reconstruction convention")
    calendar = mapping.get("time.calendar")
    if calendar is not None and kind != "ufs_surface":
        raise ValueError("time.calendar declarations apply only to analyzed UFS surface sources")
    with _open_netcdf(path, mapping, valid_time=valid_time, reference_time=reference_time, member=member,
                      calendar=calendar) as ds:
        coordinate_cache = {}
        def find(standard, dimensions=None, *, ndim=None):
            return _find_variable(ds, mapping, standard, dimensions, ndim=ndim, kind=kind)

        array = _array

        def coordinate(standard, units):
            return _coordinate(ds, mapping, standard, units, kind=kind)

        def atmospheric_matches(standard):
            return [var for var in ds.variables.values() if getattr(var, "standard_name", "") == standard
                    and (var.ndim == 4 or var.ndim == 3 and any(
                        dim in ds.variables and hasattr(ds[dim], "compress") for dim in var.dimensions))]

        def atmosphere_variable(standard):
            if standard in mapping:
                return find(standard)
            matches = atmospheric_matches(standard)
            if len(matches) != 1:
                raise ValueError(f"expected one atmospheric {standard} variable, found {len(matches)}")
            return matches[0]

        height_name = "geopotential_height"
        anchor_name = "air_temperature"
        if kind == "atmosphere":
            height_quantities = ("geopotential_height", "geopotential")
            explicit = [name for name in height_quantities if name in mapping]
            supplied = [name for name in height_quantities if name in mapping or atmospheric_matches(name)]
            if hybrid_convention is not None and supplied:
                raise ValueError("supplied geopotential height or geopotential conflicts with selected reconstruction")
            selected = explicit or supplied
            if len(selected) > 1:
                raise ValueError("conflicting geopotential height and geopotential; map exactly one quantity explicitly")
            if selected:
                height_name = selected[0]
                anchor_name = height_name
            anchor = atmosphere_variable(anchor_name)
        elif kind == "surface":
            anchor = find("land_binary_mask", ndim=3)
        elif kind == "near_surface":
            anchor = find("air_temperature", ndim=3)
        elif kind == "ufs_surface":
            anchor = find("aerodynamic_roughness_length", ndim=3)
        elif kind == "ozone":
            anchor = find("ozone_mass_mixing_ratio" if "ozone_mass_mixing_ratio" in mapping else "mass_fraction_of_ozone_in_air")
        else:
            anchor = None
        anchor_mapping = mapping
        if kind in ("atmosphere", "near_surface", "surface", "ufs_surface"):
            anchor_mapping = dict(mapping)
            prefix = {"atmosphere": anchor_name, "near_surface": "air_temperature", "surface": "land_binary_mask",
                      "ufs_surface": "aerodynamic_roughness_length"}[kind]
            for standard in ("latitude", "longitude"):
                if f"{prefix}.{standard}" in mapping:
                    anchor_mapping[standard] = mapping[f"{prefix}.{standard}"]
        lat, lon, lat_values, lon_values, latitude_order, longitude_order, grid_type, row_counts = _global_coordinates(
            ds, anchor_mapping, kind=kind, owner=anchor, cache=coordinate_cache)
        horizontal_dimensions = ((lat.dimensions[0],) if row_counts is not None else (lat.dimensions[0], lon.dimensions[0]))
        horizontal_orders = (((lat.dimensions[0], latitude_order),) if row_counts is not None
                             else ((lat.dimensions[0], latitude_order), (lon.dimensions[0], longitude_order)))

        time, (valid, reference, lead, member), _ = _forecast_identity(ds, mapping, calendar)

        def read_field(var, units, dimension_order, *, masked=False, orders=None, declared_units=None):
            if (getattr(var, "units", None) if declared_units is None else declared_units) != units:
                raise ValueError(f"unsupported or ambiguous units for {var.name}; expected {units}")
            if len(var.dimensions) != len(dimension_order) or set(var.dimensions) != set(dimension_order):
                raise ValueError(f"unsupported dimensions for {var.name}: {var.dimensions}")
            values = array(var, masked=masked).transpose(tuple(var.dimensions.index(d) for d in dimension_order))[0]
            # Canonical scans keep the native stencil and reduction order independent of file layout.
            orders = horizontal_orders if orders is None else orders
            for dimension, order in orders:
                if dimension in dimension_order:
                    values = np.take(values, order, axis=dimension_order.index(dimension) - 1)
            return torch.tensor(values.copy(), dtype=torch.float64)

        def field(name, units, dimension_order, *, masked=False):
            return read_field(find(name, dimension_order), units, dimension_order, masked=masked)

        def liquid_humidity(dimensions, *, atmospheric=False):
            variable = atmosphere_variable("relative_humidity") if atmospheric else find("relative_humidity", dimensions)
            _humidity_reference(variable, mapping)
            units = getattr(variable, "units", None)
            if units not in ("%", "1"):
                raise ValueError("unsupported relative humidity units")
            value = atmospheric_field("relative_humidity", units) if atmospheric else field("relative_humidity", units, dimensions)
            return torch.where(value == MISSING, MISSING, value * 100.0) if units == "1" else value

        def require_height(variable, expected_height, name, declared=None):
            heights = [ds[n] for n in getattr(variable, "coordinates", "").split() if n in ds.variables
                       and getattr(ds[n], "standard_name", "") == "height"]
            if declared is not None:
                try:
                    value = float(declared)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"declared {name}.height must be {expected_height} m") from exc
                if isinstance(declared, bool) or value != expected_height:
                    raise ValueError(f"declared {name}.height must be {expected_height} m")
                if not heights:
                    return
            if (len(heights) != 1 or getattr(heights[0], "units", "") != "m"
                    or array(heights[0]).size != 1 or array(heights[0]).reshape(-1)[0] != expected_height
                    or getattr(heights[0], "positive", "up") != "up"):
                raise ValueError(f"near-surface {name} requires an explicit {expected_height}m height")

        if kind == "ufs_surface":
            fields, field_grids, declarations = {}, {}, {}
            if calendar is not None and getattr(time, "calendar", "standard") != calendar:
                declarations["time.calendar"] = {"declared": calendar, "source": getattr(time, "calendar", None)}
            for name, units in UFS_SURFACE_VARIABLES.items():
                variable = find(name, ndim=3)
                coordinate_mapping = {}
                for standard in ("latitude", "longitude"):
                    key = f"{name}.{standard}"
                    if key in mapping:
                        coordinate_mapping[standard] = mapping[key]
                    elif standard in mapping and ds[mapping[standard]].dimensions[0] in variable.dimensions:
                        coordinate_mapping[standard] = mapping[standard]
                own_lat, own_lon, own_y, own_x, row_order, column_order, own_type, _ = _global_coordinates(
                    ds, coordinate_mapping, kind=kind, owner=variable, cache=coordinate_cache)
                dimensions = (time.dimensions[0], own_lat.dimensions[0], own_lon.dimensions[0])
                if getattr(variable, "mpas_init_quantity", name) != name:
                    raise ValueError(f"conflicting analyzed UFS surface quantity: {name}")
                if getattr(variable, "cell_methods", "").strip() not in ("", "time: point"):
                    raise ValueError("analyzed UFS surface requires instantaneous point samples")
                if name in ("air_temperature_2m", "specific_humidity_2m"):
                    height = mapping.get(name + ".height")
                    require_height(variable, 2, name, height)
                    if height is not None:
                        declarations[name + ".height"] = {"declared": str(height), "source": None}
                declared_units = mapping.get(name + ".units")
                if declared_units is not None:
                    # A declaration replaces a nonconforming source label; it never selects a conversion.
                    if declared_units != units:
                        raise ValueError(f"declared {name}.units must be {units}, not {declared_units!r}")
                    if getattr(variable, "units", None) != units:
                        declarations[name + ".units"] = {"declared": declared_units, "source": getattr(variable, "units", None)}
                # FV3 writes a non-CF "missing" attribute; reject its sentinel rather than read it as data.
                if hasattr(variable, "missing") and np.any(
                        np.ma.asarray(variable[:]) == np.asarray(variable.missing).astype(variable.dtype)):
                    raise ValueError(f"source missing-value sentinel in {variable.name}")
                fields[name] = read_field(variable, units, dimensions, declared_units=declared_units, orders=(
                    (own_lat.dimensions[0], row_order), (own_lon.dimensions[0], column_order)))
                if (own_lat.name, own_lon.name) != (lat.name, lon.name):
                    field_grids[name] = HorizontalGrid(torch.tensor(own_y.copy()), torch.tensor(own_x.copy()), own_type)
            return UfsSurfaceSource(torch.tensor(lat_values.copy()), torch.tensor(lon_values.copy()), fields,
                                    valid, reference, lead, member, field_grids or None, declarations or None)

        if kind == "near_surface":
            fields, field_grids = {}, {}
            for name, units, expected_height in (("air_temperature", "K", 2), ("relative_humidity", "%", 2),
                                                 ("eastward_wind", "m s-1", 10), ("northward_wind", "m s-1", 10)):
                variable = find(name, ndim=3)
                coordinate_mapping = {}
                for standard in ("latitude", "longitude"):
                    key = f"{name}.{standard}"
                    if key in mapping:
                        coordinate_mapping[standard] = mapping[key]
                    elif standard in mapping and ds[mapping[standard]].dimensions[0] in variable.dimensions:
                        coordinate_mapping[standard] = mapping[standard]
                own_lat, own_lon, own_y, own_x, row_order, column_order, own_type, _ = _global_coordinates(
                    ds, coordinate_mapping, kind=kind, owner=variable, cache=coordinate_cache)
                dimensions = (time.dimensions[0], own_lat.dimensions[0], own_lon.dimensions[0])
                require_height(variable, expected_height, name)
                if getattr(variable, "cell_methods", "").strip() not in ("", "time: point"):
                    raise ValueError("near-surface fields require instantaneous point samples")
                if name == "relative_humidity":
                    humidity_reference = _humidity_reference(variable, mapping)
                    units = getattr(variable, "units", None)
                    if units not in ("%", "1"):
                        raise ValueError("unsupported relative humidity units")
                fields[name] = read_field(variable, units, dimensions, orders=(
                    (own_lat.dimensions[0], row_order), (own_lon.dimensions[0], column_order)))
                if name == "relative_humidity" and units == "1":
                    fields[name] = fields[name] * 100.0
                if (own_lat.name, own_lon.name) != (lat.name, lon.name):
                    field_grids[name] = HorizontalGrid(torch.tensor(own_y.copy()), torch.tensor(own_x.copy()), own_type)
            _normalize_humidity(fields, field_grids, HorizontalGrid(torch.tensor(lat_values.copy()),
                                torch.tensor(lon_values.copy()), grid_type), humidity_reference)
            return NearSurfaceSource(torch.tensor(lat_values.copy()), torch.tensor(lon_values.copy()), fields, valid,
                                     reference, lead, grid_type, member, field_grids or None)

        if kind == "surface":
            depth, depth_values = coordinate("depth", ("m",))
            if getattr(depth, "positive", None) != "down" or getattr(depth, "bounds", "") not in ds.variables:
                raise ValueError("soil depth requires positive=down and explicit bounds")
            variable = ds[depth.bounds]
            if (variable.ndim != 2 or variable.dimensions[0] != depth.dimensions[0]
                    or variable.shape != (len(depth_values), 2) or getattr(variable, "units", "m") != "m"):
                raise ValueError("unsupported soil depth bounds")
            bounds = array(variable)
            order = np.argsort(depth_values)
            bounds = bounds[order]
            if (np.any(bounds[:, 1] <= bounds[:, 0]) or bounds[0, 0] != 0
                    or np.any(bounds[1:, 0] != bounds[:-1, 1])
                    or not np.allclose(depth_values[order], bounds.mean(axis=1), rtol=0, atol=1e-8)):
                raise ValueError("soil layers require contiguous positive bounds and midpoint depths")
            fields, field_grids, field_landmasks, snow_policies = {}, {}, {}, {}
            snow_convention = _snow_convention(mapping)
            for name, (standard, units) in SURFACE_VARIABLES.items():
                if name == "snow_depth" and snow_convention is not None:
                    if any(getattr(v, "standard_name", "") == standard for v in ds.variables.values()):
                        raise ValueError("declared surface_snow_amount.convention=wps_ncep_gfs derives snow depth; "
                                         "a supplied snow depth conflicts")
                    continue
                var = find(standard, ndim=4 if name.startswith("soil_") else 3)
                if name in ("snow_water_equivalent", "snow_depth"):
                    snow_policies[name] = _snow_policy(var, mapping, standard)
                if getattr(var, "cell_methods", "").strip() not in ("", "time: point"):
                    raise ValueError("surface fields require instantaneous point samples")
                coordinate_mapping = {}
                for coord in ("latitude", "longitude"):
                    if f"{standard}.{coord}" in mapping:
                        coordinate_mapping[coord] = mapping[f"{standard}.{coord}"]
                    elif coord in mapping and ds[mapping[coord]].dimensions[0] in var.dimensions:
                        coordinate_mapping[coord] = mapping[coord]
                own_lat, own_lon, own_y, own_x, row_order, column_order, own_type, _ = _global_coordinates(
                    ds, coordinate_mapping, kind="surface", owner=var, cache=coordinate_cache)
                axes = (own_lat.dimensions[0], own_lon.dimensions[0])
                horizontal = (time.dimensions[0], *axes)
                dims = (horizontal[0], depth.dimensions[0], *axes) if name.startswith("soil_") else horizontal
                orders = ((own_lat.dimensions[0], row_order), (own_lon.dimensions[0], column_order))
                fields[name] = read_field(var, units, dims, masked=name != "landmask", orders=orders)
                if name.startswith("soil_"):
                    fields[name] = fields[name][torch.tensor(order.copy())]
                if name == "soil_moisture":
                    fields[name] = soil_water_maximum(fields[name], _soil_maximum(var, mapping))
                independent = (own_lat.name, own_lon.name) != (lat.name, lon.name)
                if independent:
                    field_grids[name] = HorizontalGrid(torch.tensor(own_y.copy()), torch.tensor(own_x.copy()), own_type)
                mask_key = f"{standard}.land_binary_mask"
                if name in MASKED_SOURCE_FIELDS and (independent or mask_key in mapping):
                    mask_mapping = {"land_binary_mask": mapping[mask_key]} if mask_key in mapping else {}
                    coast = _find_variable(ds, mask_mapping, "land_binary_mask", horizontal)
                    if getattr(coast, "cell_methods", "").strip() not in ("", "time: point"):
                        raise ValueError("surface field masks require instantaneous point samples")
                    mask_lat, mask_lon, *_ = _global_coordinates(ds, coordinate_mapping, kind="surface", owner=coast, cache=coordinate_cache)
                    if (mask_lat.name, mask_lon.name) != (own_lat.name, own_lon.name):
                        raise ValueError("surface field mask coordinates differ from its field: " + name)
                    mask = read_field(coast, "1", horizontal, orders=orders)
                    if not torch.all((mask == 0) | (mask == 1)):
                        raise ValueError("source field landmask must be binary: " + name)
                    field_landmasks[name] = mask
            mask = fields["landmask"]
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("source surface landmask must be binary")
            _normalize_surface_snow(fields, field_grids, HorizontalGrid(torch.tensor(lat_values.copy()),
                                    torch.tensor(lon_values.copy()), grid_type), snow_policies)
            declarations = _apply_snow_convention(fields, field_grids, field_landmasks, snow_convention)
            return SurfaceSource(torch.tensor(lat_values.copy()), torch.tensor(lon_values.copy()),
                                  torch.tensor(bounds.copy()), fields, valid, reference, lead, grid_type, member,
                                  field_grids or None, field_landmasks or None, declarations)

        pressure_samples = []
        if kind in ("atmosphere", "ozone"):
            horizontal_time = (time.dimensions[0], *horizontal_dimensions)
            vertical = [d for d in anchor.dimensions if d not in horizontal_time]
            if (anchor.ndim != len(horizontal_time)+1 or len(vertical) != 1
                    or not set(horizontal_time).issubset(anchor.dimensions) or len(ds.dimensions[vertical[0]]) < 2):
                raise ValueError("atmosphere requires collocated full levels on one vertical dimension")
            dimension_order = (horizontal_time[0], vertical[0], *horizontal_time[1:])
            physical = [v for v in ds.variables.values() if getattr(v, "standard_name", "") == "air_pressure"]
            if "air_pressure" in mapping or physical:
                if hybrid_convention is not None:
                    raise ValueError("explicit air_pressure conflicts with interface reconstruction; select one representation")
                pressure = find("air_pressure")
                if pressure.ndim == 1:
                    pressure, values = coordinate("air_pressure", ("Pa", "hPa"))
                    if pressure.dimensions != (vertical[0],):
                        raise ValueError("pressure and atmospheric full levels must be collocated")
                    p = torch.tensor(values.copy(), dtype=torch.float64)
                else:
                    if getattr(pressure, "units", None) not in ("Pa", "hPa"):
                        raise ValueError("unsupported pressure units")
                    if getattr(pressure, "cell_methods", "").strip() not in ("", "time: point"):
                        raise ValueError("pressure requires instantaneous point samples")
                    p = read_field(pressure, pressure.units, dimension_order)
                    pressure_samples.append(pressure)
                if pressure.units == "hPa":
                    p = p * 100.0
            else:
                pressure = find("atmosphere_hybrid_sigma_pressure_coordinate")
                if hybrid_convention is None and pressure.dimensions != (vertical[0],):
                    raise ValueError("hybrid coordinate must be collocated with atmospheric full levels, not interfaces")
                if hybrid_convention is not None and (pressure.ndim != 1 or pressure.dimensions == (vertical[0],)
                        or pressure.size != len(ds.dimensions[vertical[0]]) + 1):
                    raise ValueError("IFS reconstruction requires one more interface than atmospheric layers")
                array(pressure)
                formula = getattr(pressure, "formula_terms", "")
                if not re.fullmatch(r"\s*(?:[A-Za-z_]\w*\s*:\s*[^\s:]+\s*)+", formula):
                    raise ValueError("invalid hybrid formula_terms")
                pairs = re.findall(r"([A-Za-z_]\w*)\s*:\s*([^\s:]+)", formula)
                terms = dict(pairs)
                if len(terms) != len(pairs) or set(terms) not in ({"a", "b", "ps", "p0"}, {"ap", "b", "ps"}):
                    raise ValueError("hybrid formula_terms must specify a/b/ps/p0 or ap/b/ps exactly once")

                def term(key, dimensions, units):
                    if terms[key] not in ds.variables:
                        raise ValueError(f"missing hybrid formula term: {key}")
                    var = ds[terms[key]]
                    if var.dimensions != dimensions or getattr(var, "units", None) not in units:
                        raise ValueError(f"unsupported hybrid {key} dimensions or units; full-level coefficients are required")
                    value = torch.tensor(array(var).copy(), dtype=torch.float64)
                    return value * 100.0 if var.units == "hPa" else value

                coefficient_dimensions = pressure.dimensions
                a = term("a" if "a" in terms else "ap", coefficient_dimensions, ("1",) if "a" in terms else ("Pa", "hPa"))
                b = term("b", coefficient_dimensions, ("1",))
                if terms["ps"] not in ds.variables:
                    raise ValueError("missing hybrid formula term: ps")
                ps_var = ds[terms["ps"]]
                if getattr(ps_var, "units", None) not in ("Pa", "hPa"):
                    raise ValueError("unsupported hybrid ps units")
                if getattr(ps_var, "cell_methods", "").strip() not in ("", "time: point"):
                    raise ValueError("hybrid ps requires instantaneous point samples")
                ps = read_field(ps_var, ps_var.units, horizontal_time)
                pressure_samples.append(ps_var)
                if ps_var.units == "hPa":
                    ps = ps * 100.0
                p0 = term("p0", (), ("Pa", "hPa")) if "a" in terms else None
                p = hybrid_pressure(a, b, ps, p0, interfaces=hybrid_convention is not None)
                if hybrid_convention is not None:
                    # The explicit convention supports either stored scan, without inferring layer ownership.
                    level_values = array(ds[vertical[0]]) if vertical[0] in ds.variables else None
                    interface_values = array(pressure)
                    if (level_values is None or ds[vertical[0]].dimensions != (vertical[0],)
                            or getattr(ds[vertical[0]], "units", "") != "1"
                            or getattr(pressure, "units", "") != "1"
                            or level_values.shape != (anchor.shape[anchor.dimensions.index(vertical[0])],)
                            or not (np.array_equal(np.sort(level_values), np.arange(1, level_values.size + 1))
                                    and np.array_equal(np.sort(interface_values), np.arange(interface_values.size)))):
                        raise ValueError("IFS reconstruction requires complete layer IDs 1..N and interface IDs 0..N")
                    layer_order, interface_order = np.argsort(level_values), np.argsort(interface_values)
                    p = p[torch.tensor(interface_order.copy())]
                    if not torch.all(p[-1] == ps):
                        raise ValueError("bottom IFS interface must equal supplied surface pressure")
            if hybrid_convention is None and not torch.all(torch.isfinite(p) & (p > 0)):
                raise ValueError("pressure must be finite and positive")
        else:
            pressure, pressure_values = coordinate("air_pressure", ("Pa", "hPa"))
            if np.any(pressure_values <= 0):
                raise ValueError("pressure must be positive")
            dimension_order = (time.dimensions[0], pressure.dimensions[0], *horizontal_dimensions)
            p = torch.tensor(pressure_values.copy(), dtype=torch.float64)
            if pressure.units == "hPa":
                p = p * 100.0
        if kind == "ozone":
            if hybrid_convention is not None:
                p = ifs_full_level_pressure(p)
            if p.ndim > 1:
                dp = torch.diff(p, dim=0)
                if not (torch.all(dp > 0) or torch.all(dp < 0)):
                    raise ValueError("ozone full-level pressure requires one strictly monotonic scan")
            if {"ozone_mass_mixing_ratio", "mass_fraction_of_ozone_in_air"}.issubset(mapping):
                raise ValueError("conflicting ozone quantity mappings")
            dry = "ozone_mass_mixing_ratio" in mapping
            name = "ozone_mass_mixing_ratio" if dry else "mass_fraction_of_ozone_in_air"
            variable = find(name, dimension_order)
            units = getattr(variable, "units", None)
            if units not in ("kg kg-1", "1"):
                raise ValueError("ozone requires mass units kg kg-1 or 1, not mole/volume units")
            basis = "dry_air" if dry else "moist_air"
            declared = getattr(variable, "mpas_init_mass_basis", None)
            if (dry and (declared != basis or getattr(variable, "standard_name", "") != "")
                    or not dry and declared not in (None, basis)):
                raise ValueError("conflicting or missing ozone mass basis declaration")
            samples = [variable] if dry else [variable, find("specific_humidity", dimension_order)]
            for sample in (*samples[1:], *pressure_samples):
                for standard, primary in (("latitude", lat), ("longitude", lon)):
                    own, own_values = _coordinate(ds, mapping, standard,
                                                  LATITUDE_UNITS if standard == "latitude" else LONGITUDE_UNITS,
                                                  kind=kind, owner=sample, monotonic=False)
                    if own.dimensions != primary.dimensions or not np.array_equal(own_values, array(primary)):
                        raise ValueError("ozone requires collocated humidity and pressure coordinates")
            if any(getattr(sample, "cell_methods", "").strip() not in ("", "time: point") for sample in samples):
                raise ValueError("ozone normalization requires instantaneous point samples; cell means are unsupported")
            ozone = field(name, units, dimension_order)
            humidity = None if dry else field("specific_humidity", "kg kg-1", dimension_order)
            if hybrid_convention is not None:
                order = torch.tensor(layer_order.copy())
                ozone = ozone[order]
                humidity = None if humidity is None else humidity[order]
            # Retain reduced longitude storage precision for its periodic-row validity bound.
            longitude = torch.tensor(lon_values.copy(), dtype=torch.float32 if row_counts is not None and lon.dtype == np.dtype("f4") else torch.float64)
            return OzoneSource(p, torch.tensor(lat_values.copy()), longitude,
                               ozone, basis, humidity, valid, reference, lead, member, grid_type,
                               None if row_counts is None else torch.tensor(row_counts.copy()), hybrid_convention)
        field_grids, atmospheric_variables, field_surface_pressure = {}, {}, {}

        def atmospheric_field(name, units):
            quantity = height_name if name == "geopotential_height" else name
            var = atmosphere_variable(quantity)
            if quantity == "geopotential":
                units = getattr(var, "units", None)
                if units not in ("m2 s-2", "m**2 s**-2"):
                    raise ValueError("geopotential requires units m2 s-2 or m**2 s**-2")
            coordinate_mapping = {}
            for standard in ("latitude", "longitude"):
                key = f"{quantity}.{standard}"
                if key in mapping:
                    coordinate_mapping[standard] = mapping[key]
                elif standard in mapping and ds[mapping[standard]].dimensions[0] in var.dimensions:
                    coordinate_mapping[standard] = mapping[standard]
            own_lat, own_lon, own_y, own_x, row_order, column_order, own_type, own_counts = _global_coordinates(
                ds, coordinate_mapping, kind="atmosphere", owner=var, cache=coordinate_cache)
            axes = (own_lat.dimensions[0],) if own_counts is not None else (own_lat.dimensions[0], own_lon.dimensions[0])
            dims = (time.dimensions[0], vertical[0], *axes)
            orders = ((own_lat.dimensions[0], row_order),) if own_counts is not None else (
                (own_lat.dimensions[0], row_order), (own_lon.dimensions[0], column_order))
            value = read_field(var, units, dims, orders=orders, masked=kind == "atmosphere")
            if kind == "atmosphere" and torch.any(value == MISSING):
                if p.ndim != 1 or own_counts is not None or hybrid_convention is not None:
                    raise ValueError("missing or nonfinite atmospheric data require an isobaric axis and rectangular grid")
                surface_mapping = dict(mapping)
                key = f"{quantity}.surface_air_pressure"
                if key in mapping:
                    surface_mapping["surface_air_pressure"] = mapping[key]
                surface_dims = (time.dimensions[0], *axes)
                try:
                    surface = _find_variable(ds, surface_mapping, "surface_air_pressure", surface_dims)
                except ValueError as exc:
                    raise ValueError("missing or nonfinite atmospheric data require collocated surface_air_pressure: " + name) from exc
                if (getattr(surface, "units", None) not in ("Pa", "hPa")
                        or getattr(surface, "cell_methods", "").strip() not in ("", "time: point")):
                    raise ValueError("below-ground surface pressure requires Pa/hPa instantaneous point samples")
                ps = read_field(surface, surface.units, surface_dims, orders=orders)
                ps = ps * 100. if surface.units == "hPa" else ps
                above = p[:, None, None] <= ps
                if (not torch.all(torch.isfinite(ps) & (ps > 0)) or not torch.all(above.sum(0) >= 2)
                        or torch.any((value == MISSING) & above)):
                    raise ValueError("missing above-ground atmospheric data or invalid surface pressure: " + name)
                field_surface_pressure[name] = ps
            if (own_lat.name, own_lon.name) != (lat.name, lon.name):
                if quantity == anchor_name:
                    raise ValueError("the pressure/height anchor must use the primary grid coordinates")
                field_grids[name] = HorizontalGrid(torch.tensor(own_y.copy()), torch.tensor(own_x.copy()), own_type,
                                                   None if own_counts is None else torch.tensor(own_counts.copy()))
            atmospheric_variables[name] = var
            return torch.where(value == MISSING, value, geopotential_to_height(value)) if quantity == "geopotential" else value

        required = {name: units for name, units in REQUIRED.items() if hybrid_convention is None or name != "geopotential_height"}
        fields = {name: atmospheric_field(name, units) if kind == "atmosphere" else field(name, units, dimension_order)
                  for name, units in required.items()}
        if hybrid_convention is not None:
            if any(name in field_grids for name in ("air_temperature", "specific_humidity")):
                raise ValueError("IFS height reconstruction requires collocated temperature and specific humidity")
            fields = {name: value[torch.tensor(layer_order.copy())] for name, value in fields.items()}
            phi_var = find("surface_geopotential", horizontal_time)
            if getattr(phi_var, "cell_methods", "").strip() not in ("", "time: point"):
                raise ValueError("surface geopotential requires instantaneous point samples")
            phi = read_field(phi_var, "m2 s-2", horizontal_time)
            p, fields["geopotential_height"] = ifs_hydrostatic_columns(p, fields["air_temperature"], fields["specific_humidity"], phi)
        if include_relative_humidity:
            fields["relative_humidity"] = liquid_humidity(dimension_order, atmospheric=kind == "atmosphere")
            if hybrid_convention is not None:
                fields["relative_humidity"] = fields["relative_humidity"][torch.tensor(layer_order.copy())]
            variable = atmosphere_variable("relative_humidity") if kind == "atmosphere" else find("relative_humidity", dimension_order)
            _normalize_humidity(fields, field_grids, HorizontalGrid(torch.tensor(lat_values.copy()),
                                torch.tensor(lon_values.copy()), grid_type,
                                None if row_counts is None else torch.tensor(row_counts.copy())),
                                _humidity_reference(variable, mapping))
        if kind == "atmosphere":
            for name in fields:
                if name == "geopotential_height" and hybrid_convention is not None:
                    continue
                var = atmospheric_variables[name]
                if getattr(var, "cell_methods", "").strip() not in ("", "time: point"):
                    raise ValueError("atmosphere requires instantaneous point samples")
            if p.ndim > 1:
                dp, dz = torch.diff(p, dim=0), torch.diff(fields["geopotential_height"], dim=0)
                if not (torch.all((dp > 0) & (dz < 0)) or torch.all((dp < 0) & (dz > 0))):
                    raise ValueError("collocated pressure must be strictly monotonic and decrease with increasing height")
        return PressureLevelSource(p, torch.tensor(lat_values.copy(), dtype=torch.float64),
                                   torch.tensor(lon_values.copy(), dtype=torch.float64), fields, valid, reference, lead, grid_type, member,
                                   hybrid_convention, field_grids or None,
                                   None if row_counts is None else torch.tensor(row_counts.copy()), field_surface_pressure or None)


def _grib_paths(path):
    paths = (path,) if isinstance(path, (str, Path)) else tuple(path)
    formats = []
    for source in paths:
        with Path(source).open("rb") as stream:
            formats.append(stream.read(4) == b"GRIB")
    if any(formats) and not all(formats):
        raise ValueError("source groups cannot mix GRIB and NetCDF files; use separate physical source options")
    return paths if formats and all(formats) else None


def read_atmosphere(path: SourcePaths, mapping: dict[str, str] | None = None, *,
                    include_relative_humidity: bool = False, hybrid_convention: str | None = None,
                    valid_time: datetime | None = None, reference_time: datetime | None = None,
                    member: int | None = None) -> PressureLevelSource:
    """Read global CF full levels, isobaric GRIB, or explicitly interpreted hybrid GRIB.

    Geopotential height uses m; geopotential uses m2 s-2 or m**2 s**-2 and
    divides by standard gravity 9.80665 m/s2. Both quantities are geoid-relative,
    not geometric altitude or surface-relative height. If both are supplied,
    map exactly one quantity explicitly. The tensor source always stores height.

    Accept a pressure axis, explicit collocated 3-D pressure, or either CF hybrid
    sigma-pressure formula with full-level coefficients. Explicit physical pressure
    takes precedence over formula coordinates. No interface averaging, missing-height
    reconstruction or producer semantics is inferred. Select a multi-time/member
    forecast explicitly by UTC-aware valid_time, optional reference_time, and member.
    Explicit ``hybrid_convention="ifs-vapour-only"`` instead requires complete
    half-level coefficients, layer/interface IDs, and surface geopotential.
    It reconstructs arithmetic-mean full pressure and approximate hydrostatic height.
    For GRIB, require complete hybrid layers 1..N and matching pv arrays in
    ap-then-b interface order. Surface pressure uses Pa or explicitly mapped
    logarithm_of_surface_air_pressure=shortName for ln(Pa); no convention is inferred.
    Accept pole-to-pole regular latitude or globally covered nonuniform latitude,
    including regular Gaussian grids. Supplied coordinates are not replaced by roots.
    Isobaric rectangular fields may have masked pixels below collocated source
    surface_air_pressure (Pa/hPa). Missing above-ground data remain errors.
    Non-height fields can use their own global rectilinear grids with the
    same vertical dimension. Select coordinates through CF metadata or explicit
    FIELD.latitude/FIELD.longitude mapping keys. No intermediate grid is created.
    """
    paths = _grib_paths(path)
    if paths is not None:
        from mpas_init.grib import read_grib_atmosphere

        return read_grib_atmosphere(paths, mapping, include_relative_humidity=include_relative_humidity,
                                    hybrid_convention=hybrid_convention,
                                    valid_time=valid_time, reference_time=reference_time, member=member)
    return _read_regular(path, mapping, kind="atmosphere", include_relative_humidity=include_relative_humidity,
                         hybrid_convention=hybrid_convention, valid_time=valid_time, reference_time=reference_time, member=member)


def read_pressure_levels(path: SourcePaths, mapping: dict[str, str] | None = None, *,
                           include_relative_humidity: bool = False, valid_time: datetime | None = None,
                           reference_time: datetime | None = None, member: int | None = None) -> PressureLevelSource:
    """Read one global rectilinear latitude-longitude isobaric state.

    Mapping keys are CF standard names. Values are file variable names. Explicit
    mappings do not waive units or dimension checks. This first decoder rejects
    hybrid and missing atmospheric data. Select multiple times/members explicitly.
    """
    if _grib_paths(path) is not None:
        source = read_atmosphere(path, mapping, include_relative_humidity=include_relative_humidity,
                                 valid_time=valid_time, reference_time=reference_time, member=member)
        if source.field_surface_pressure:
            raise ValueError("missing atmospheric data require read_atmosphere, not read_pressure_levels")
        return source
    return _read_regular(path, mapping, kind="pressure", include_relative_humidity=include_relative_humidity,
                         valid_time=valid_time, reference_time=reference_time, member=member)


def read_surface(path: SourcePaths, mapping: dict[str, str] | None = None, *, valid_time: datetime | None = None,
                  reference_time: datetime | None = None, member: int | None = None) -> SurfaceSource:
    """Read explicit surface inputs, including total soil water and depth bounds.

    Require globally covered rectilinear fields and one valid time.
    Each field may own coordinates through CF metadata or explicit
    FIELD.latitude/FIELD.longitude mapping keys. Independent soil/ice grids
    require collocated binary masks, selected by FIELD.land_binary_mask when
    ambiguous. FIELD is the field's CF standard name. Soil bounds stay shared.
    Missing pixels retain a distinct sentinel for native coastal selection.
    Zero snow and zero soil water remain physical values. No units are guessed.
    Mapping keys are CF standard names, as in read_pressure_levels.
    Optional surface_snow_amount.convention=wps_ncep_gfs declares the WPS 4.6
    NCEP GFS snow: SWE doubles after missing-ocean filling, and depth becomes
    0.005 x that SWE. A supplied or mapped snow depth then rejects.
    """
    paths = _grib_paths(path)
    if paths is not None:
        from mpas_init.grib import read_grib_surface

        return read_grib_surface(paths, mapping, valid_time=valid_time, reference_time=reference_time, member=member)
    return _read_regular(path, mapping, kind="surface", valid_time=valid_time, reference_time=reference_time, member=member)


def read_near_surface(path: SourcePaths, mapping: dict[str, str] | None = None, *, valid_time: datetime | None = None,
                       reference_time: datetime | None = None, member: int | None = None) -> NearSurfaceSource:
    """Read complete CF/GRIB 2m/10m fields and normalize explicitly declared RH.

    Each field can own a global rectilinear grid through CF coordinates or explicit
    FIELD.latitude/FIELD.longitude mapping keys. No intermediate grid is created.
    """
    paths = _grib_paths(path)
    if paths is not None:
        from mpas_init.grib import read_grib_near_surface

        return read_grib_near_surface(paths, mapping, valid_time=valid_time, reference_time=reference_time, member=member)
    return _read_regular(path, mapping, kind="near_surface", valid_time=valid_time, reference_time=reference_time, member=member)


def read_ozone(path: SourcePaths, mapping: dict[str, str] | None = None, *, valid_time: datetime | None = None,
                 reference_time: datetime | None = None, member: int | None = None,
                 hybrid_convention: str | None = None) -> OzoneSource:
    """Read CF or isobaric GRIB ozone with explicit mass conventions.

    Native dry-air ratios instead require an explicit ozone_mass_mixing_ratio
    mapping and mpas_init_mass_basis="dry_air". This project-specific mapping
    does not redefine CF mass_fraction_of_ozone_in_air or infer producer semantics.
    Accept globally covered nonuniform latitude or CF gathered periodic rows.
    Humidity must share ozone's pressure and horizontal samples before conversion.
    Accept a pressure axis, collocated full-level pressure, or either CF hybrid
    formula with full-level coefficients; never infer an interface mean.
    Explicit ``ifs-arithmetic`` instead requires complete interfaces and surface
    pressure, and applies the IFS arithmetic mean without reconstructing height.
    Reduced longitude tensors retain their stored precision; numerical remapping
    promotes coordinates to the field dtype without replacing supplied nodes.
    GRIB requires quantity.mass_basis=moist_air or dry_air in the mapping;
    moist fractions require collocated specific humidity on matching levels.
    """
    paths = _grib_paths(path)
    if paths is not None:
        from mpas_init.grib import read_grib_ozone

        return read_grib_ozone(paths, mapping, valid_time=valid_time, reference_time=reference_time, member=member,
                               hybrid_convention=hybrid_convention)
    return _read_regular(path, mapping, kind="ozone", valid_time=valid_time, reference_time=reference_time, member=member,
                         hybrid_convention=hybrid_convention)


def read_ufs_surface(path: SourcePaths, mapping: dict[str, str] | None = None, *, valid_time: datetime | None = None,
                       reference_time: datetime | None = None, member: int | None = None) -> UfsSurfaceSource:
    """Read CF/GRIB analyzed quantities, canonical units, and forecast identity.

    Mapping keys are UFS_SURFACE_VARIABLES quantities, not producer short names
    or CF standard names. Unmapped variables need matching mpas_init_quantity.
    Latitude can be nonuniform; longitude must cover one regular periodic grid.
    Fields can own separate axes through CF coordinates or quantity.latitude/
    quantity.longitude mappings; the roughness field owns the primary grid.
    No unit, calendar, level, or missing-value override is inferred.
    NetCDF mappings can declare explicit overrides, recorded in ``declarations``:
    quantity.units names the canonical unit for a nonconforming label (other
    values reject; no conversion follows); time.calendar=standard, gregorian or
    proleptic_gregorian decodes clocks whose source-calendar civil labels are
    identical; quantity.height="2" (metres) supplies an absent 2m T/q height.
    Mapping forecast_reference_time to the time variable declares a zero-lead analysis.
    GRIB requires all quantity mappings, fixed 2m T/q, and an explicitly located
    deep soil point. Numeric selectors include centre and table versions.
    Unknown codec units require quantity.units; percent albedos become fractions.
    """
    paths = _grib_paths(path)
    if paths is not None:
        from mpas_init.grib import read_grib_ufs_surface
        return read_grib_ufs_surface(paths, mapping, valid_time=valid_time, reference_time=reference_time, member=member)
    return _read_regular(path, mapping, kind="ufs_surface", valid_time=valid_time, reference_time=reference_time, member=member)
