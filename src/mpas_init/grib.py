"""ecCodes atmosphere, ozone and complete surface decoding; no remapping here."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from mpas_init.horizontal import HorizontalGrid
from mpas_init.netcdf import (NearSurfaceSource, OzoneSource, PressureLevelSource, UfsSurfaceSource, REQUIRED, SourcePaths,
                              SURFACE_VARIABLES, UFS_SURFACE_VARIABLES, _humidity_reference, _normalize_humidity,
                              _apply_snow_convention, _normalize_surface_snow, _snow_convention, _snow_policy,
                              _soil_maximum, _utc_selection)
from mpas_init.surface import MASKED_SOURCE_FIELDS, MISSING, SurfaceSource, soil_water_maximum
from mpas_init.vertical import geopotential_to_height, hybrid_pressure, ifs_full_level_pressure, ifs_hydrostatic_columns


_UNITS = {"air_temperature": ("K",), "eastward_wind": ("m s**-1", "m s-1"),
          "northward_wind": ("m s**-1", "m s-1"),
          "specific_humidity": ("kg kg**-1", "kg kg-1"),
          "geopotential_height": ("gpm", "m"), "geopotential": ("m**2 s**-2", "m2 s-2"),
          "relative_humidity": ("%",),
          "mass_fraction_of_ozone_in_air": ("kg kg**-1", "kg kg-1", "1"),
          "ozone_mass_mixing_ratio": ("kg kg**-1", "kg kg-1", "1")}
_UNITS.update({"land_binary_mask": ("1", "(0 - 1)"), "surface_altitude": ("m", "gpm"),
               "surface_temperature": ("K",), "surface_snow_amount": ("kg m**-2", "kg m-2"),
               "surface_snow_thickness": ("m",), "sea_ice_area_fraction": ("1", "(0 - 1)"),
               "soil_temperature": ("K",), "volume_fraction_of_condensed_water_in_soil": ("1", "Proportion", "m**3 m**-3", "m3 m-3")})
_NEAR_HEIGHTS = {"air_temperature": 2., "relative_humidity": 2., "eastward_wind": 10., "northward_wind": 10.}
_SOIL = {"soil_temperature", "volume_fraction_of_condensed_water_in_soil"}
_HYBRID_SURFACE = {"surface_geopotential", "surface_air_pressure", "logarithm_of_surface_air_pressure"}
_UNITS.update({"surface_geopotential": _UNITS["geopotential"], "surface_air_pressure": ("Pa",),
               "logarithm_of_surface_air_pressure": ("1", "Numeric")})
_UFS_UNITS = {name: (("1", "Numeric", "(0 - 1)") if units == "1" else (units,))
              for name, units in UFS_SURFACE_VARIABLES.items()}
_UFS_UNITS.update({"canopy_water_amount": ("kg m-2", "kg m**-2"),
                   "friction_velocity": ("m s-1", "m s**-1"),
                   "specific_humidity_2m": ("kg kg-1", "kg kg**-1")})
for _name in UFS_SURFACE_VARIABLES:
    if _name.endswith("albedo"):
        _UFS_UNITS[_name] += ("%",)
_PARAMETER_KEYS = {1: ("centre", "table2Version", "indicatorOfParameter"),
                   2: ("centre", "tablesVersion", "localTablesVersion", "discipline", "parameterCategory", "parameterNumber")}


def _parameter_selector(value):
    if not value.startswith(("grib1:", "grib2:")):
        return value
    kind, *numbers = value.split(":")
    edition = int(kind[-1])
    if (len(numbers) != len(_PARAMETER_KEYS[edition])
            or any(not number.isascii() or not number.isdecimal() for number in numbers)):
        raise ValueError("invalid GRIB numeric parameter selector")
    values = tuple(int(number) for number in numbers)
    if any(value > (65535 if edition == 2 and index == 0 else 255) for index, value in enumerate(values)):
        raise ValueError("invalid GRIB numeric parameter selector")
    return edition, values


def _matches_parameter(codes, handle, selector, short_name):
    if isinstance(selector, str):
        return short_name == selector
    edition, values = selector
    return (codes.codes_get(handle, "edition", int) == edition
            and all(codes.codes_get(handle, key, int) == value
                    for key, value in zip(_PARAMETER_KEYS[edition], values)))


def _soil_bounds(codes, handle):
    if codes.codes_get(handle, "edition", int) == 1:
        return tuple(float(Decimal(codes.codes_get(handle, key, int)) / 100) for key in ("topLevel", "bottomLevel"))
    bounds = []
    for side in ("First", "Second"):
        if codes.codes_get(handle, f"typeOf{side}FixedSurface", int) != 106:
            raise ValueError("GRIB soil bounds require two depth-below-land surfaces")
        keys = (f"scaleFactorOf{side}FixedSurface", f"scaledValueOf{side}FixedSurface")
        if any(codes.codes_is_missing(handle, key) for key in keys):
            raise ValueError("GRIB soil bounds require explicit scaled depths")
        scale, value = (codes.codes_get(handle, key, int) for key in keys)
        bounds.append(float(Decimal(value).scaleb(-scale)))
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("GRIB soil bounds must be finite")
    return tuple(bounds)


def _identity(codes, handle):
    reference = datetime.strptime(f"{codes.codes_get(handle, 'dataDate'):08d}"
                                  f"{codes.codes_get(handle, 'dataTime'):04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    if codes.codes_is_defined(handle, "second"):
        reference += timedelta(seconds=codes.codes_get(handle, "second"))
    # stepUnits is transient: ask ecCodes for seconds instead of assuming hours.
    codes.codes_set(handle, "stepUnits", "s")
    seconds = codes.codes_get(handle, "endStep", float)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("GRIB forecast lead must be finite and nonnegative")
    lead = timedelta(seconds=seconds)
    valid = reference + lead
    if (int(valid.strftime("%Y%m%d")), valid.hour * 100 + valid.minute) != (
            codes.codes_get(handle, "validityDate"), codes.codes_get(handle, "validityTime")):
        raise ValueError("GRIB forecast reference time plus lead differs from valid time")
    member = None
    ensemble = (codes.codes_is_defined(handle, "typeOfEnsembleForecast")
                or (codes.codes_is_defined(handle, "numberOfForecastsInEnsemble")
                    and not codes.codes_is_missing(handle, "numberOfForecastsInEnsemble")
                    and codes.codes_get(handle, "numberOfForecastsInEnsemble") > 0))
    if ensemble and codes.codes_is_defined(handle, "perturbationNumber") and not codes.codes_is_missing(handle, "perturbationNumber"):
        member = codes.codes_get(handle, "perturbationNumber", int)
    return valid, reference, lead, member


def _grid(codes, handle):
    grid_type = codes.codes_get(handle, "gridType")
    if grid_type not in ("regular_ll", "regular_gg", "reduced_gg"):
        raise ValueError("unsupported GRIB grid: " + grid_type)
    nj = codes.codes_get(handle, "Nj", int)
    reduced = grid_type == "reduced_gg"
    ni = None if reduced else codes.codes_get(handle, "Ni", int)
    if nj < 3 or (ni is not None and ni < 3):
        raise ValueError("GRIB global atmosphere requires at least three points per coordinate")
    if reduced:
        if (nj != 2 * codes.codes_get(handle, "N", int) or codes.codes_get(handle, "global", int) != 1):
            raise ValueError("GRIB reduced Gaussian rows must cover a complete global grid")
        if any(codes.codes_get(handle, key, int) for key in (
                "iScansNegatively", "jScansPositively", "jPointsAreConsecutive", "alternativeRowScanning")):
            raise ValueError("GRIB reduced Gaussian decoding requires north-first eastward row scanning")
        counts = codes.codes_get_array(handle, "pl", int)
        if counts.shape != (nj,) or np.any(counts < 4):
            raise ValueError("GRIB reduced Gaussian rows require at least four points each")
    latitude = codes.codes_get_array(handle, "latitudes")
    longitude = np.remainder(codes.codes_get_array(handle, "longitudes"), 360.)
    size = int(counts.sum()) if reduced else ni * nj
    if (latitude.size != size or longitude.size != size
            or not np.isfinite(latitude).all() or not np.isfinite(longitude).all()):
        raise ValueError("GRIB coordinates do not describe a complete finite grid")
    if reduced:
        lat, actual_counts = np.unique(latitude, return_counts=True)
        lat = lat[::-1]
        if lat.size != nj or not np.array_equal(actual_counts[::-1], counts):
            raise ValueError("GRIB reduced coordinates differ from their declared row counts")
        order = np.lexsort((longitude, -latitude))
        offsets = np.r_[0, np.cumsum(counts)]
        for start, end in zip(offsets[:-1], offsets[1:]):
            row = longitude[order[start:end]]
            if not np.allclose(np.diff(np.r_[row, row[0] + 360.]), 360. / row.size, rtol=0, atol=1.e-8):
                raise ValueError("GRIB reduced longitude row must cover a complete periodic circuit")
        return HorizontalGrid(torch.tensor(lat.copy()), torch.tensor(longitude[order].copy()),
                              "reduced_gaussian", torch.tensor(counts.copy())), order
    lat, lon = np.unique(latitude)[::-1], np.unique(longitude)
    order = np.lexsort((longitude, -latitude))
    if (lat.size != nj or lon.size != ni or not np.array_equal(latitude[order], np.repeat(lat, ni))
            or not np.array_equal(longitude[order], np.tile(lon, nj))):
        raise ValueError("GRIB coordinates repeat or omit grid points")
    if not np.allclose(np.diff(np.r_[lon, lon[0] + 360.]), 360. / ni, rtol=0, atol=1.e-8):
        raise ValueError("GRIB longitude must cover one complete non-repeated periodic grid")
    if lat[0] > 90 or lat[-1] < -90:
        raise ValueError("GRIB latitude is out of range")
    if grid_type == "regular_ll":
        if (not np.allclose(np.diff(lat), lat[1] - lat[0], rtol=0, atol=1.e-8)
                or not np.allclose([lat[0], lat[-1]], [90, -90], rtol=0, atol=1.e-8)):
            raise ValueError("GRIB regular latitude must be uniform and cover both poles")
    elif (90 - lat[0] > lat[0] - lat[1] + 1.e-8
          or lat[-1] + 90 > lat[-2] - lat[-1] + 1.e-8):
        raise ValueError("GRIB Gaussian latitude lacks global polar coverage")
    # Gaussian coordinate iterators do not consume jPointsAreConsecutive.
    # Associate raw values with coordinates from the encoded scan flags instead.
    first = codes.codes_get(handle, "longitudeOfFirstGridPointInDegrees") % 360.
    start = np.argmin(np.abs((lon - first + 180.) % 360. - 180.))
    if abs((lon[start] - first + 180.) % 360. - 180.) > 1.e-8:
        raise ValueError("GRIB first longitude differs from its global coordinates")
    rows = np.arange(nj)[::-1] if codes.codes_get(handle, "jScansPositively", int) else np.arange(nj)
    sign = -1 if codes.codes_get(handle, "iScansNegatively", int) else 1
    columns = (start + sign * np.arange(ni)) % ni
    consecutive = codes.codes_get(handle, "jPointsAreConsecutive", int)
    positions = np.arange(size).reshape((ni, nj) if consecutive else (nj, ni))
    if codes.codes_get(handle, "alternativeRowScanning", int):
        positions[1::2] = positions[1::2, ::-1]
    if consecutive:
        positions = positions.T
    normalized = np.empty((nj, ni), dtype=np.int64)
    normalized[np.ix_(rows, columns)] = positions
    order = normalized.reshape(-1)
    return HorizontalGrid(torch.tensor(lat.copy()), torch.tensor(lon.copy()),
                          "regular_latlon" if grid_type == "regular_ll" else "rectilinear_latlon"), order


def _same_grid(left, right):
    return (left.grid_type == right.grid_type and torch.equal(left.latitude, right.latitude)
            and torch.equal(left.longitude, right.longitude)
            and ((left.row_counts is None and right.row_counts is None)
                 or (left.row_counts is not None and right.row_counts is not None
                     and torch.equal(left.row_counts, right.row_counts))))


def read_grib_atmosphere(paths: SourcePaths, mapping: dict[str, str] | None = None, *,
                         include_relative_humidity: bool = False, valid_time: datetime | None = None,
                         reference_time: datetime | None = None, member: int | None = None,
                         hybrid_convention: str | None = None) -> PressureLevelSource:
    """Read complete global isobaric or explicitly interpreted hybrid GRIB1/2.

    Automatic matching uses ecCodes CF quantity names, never a producer whitelist.
    Explicit CF-name mappings select shortName or table-qualified numeric parameters, never waive units,
    level, time, mask, geometry or earth-relative wind checks. Supplied geopotential
    converts with standard gravity through the shared tensor implementation.
    All fields need identical pressure levels. Different fields may own different
    global regular latitude-longitude, regular Gaussian or reduced Gaussian grids.
    Reduced rows require north-first eastward scanning and at least four points.
    Explicit ifs-vapour-only requires interface pv=(ap[0:N+1], b[0:N+1]) in
    Pa/1, complete layers 1..N, collocated T/q and surface geopotential/pressure.
    Map logarithm_of_surface_air_pressure explicitly for ln(Pa) instead of Pa.
    Isobaric rectangular bitmaps require one complete surface pressure field per
    distinct masked grid, in Pa, and two above-ground levels per source column.
    Hybrid/reduced masks, rotated grids and multi-field messages reject.
    RH requires an explicit liquid_water or wps_ice_blend saturation reference.
    The latter converts on collocated source T/RH grids, never by producer name.
    """
    return _read_grib(paths, mapping, include_relative_humidity=include_relative_humidity,
                      valid_time=valid_time, reference_time=reference_time, member=member,
                      hybrid_convention=hybrid_convention)


def read_grib_near_surface(paths: SourcePaths, mapping: dict[str, str] | None = None, *,
                           valid_time: datetime | None = None, reference_time: datetime | None = None,
                           member: int | None = None) -> NearSurfaceSource:
    """Read complete 2m T/RH and earth-relative 10m winds, without remapping.

    CF quantity names plus exact heightAboveGround levels select point samples.
    Explicit mappings select shortName, not units, height, geometry or identity.
    All four fields are mandatory. RH needs an explicit saturation reference.
    Missing values and multi-field messages remain unsupported.
    """
    return _read_grib(paths, mapping, near_surface=True, include_relative_humidity=True,
                      valid_time=valid_time, reference_time=reference_time, member=member)


def read_grib_surface(paths: SourcePaths, mapping: dict[str, str] | None = None, *,
                       valid_time: datetime | None = None, reference_time: datetime | None = None,
                       member: int | None = None) -> SurfaceSource:
    """Read complete global surface quantities and explicit soil-layer means.

    Bitmap masks remain distinct from valid zeros. Soil/ice grids must share
    the binary landmask. No moisture cap, snow depth or landmask is inferred.
    Optional quantity.missing_ocean=zero fills only declared missing water snow.
    Optional surface_snow_amount.convention=wps_ncep_gfs doubles SWE and derives
    depth as 0.005 x that SWE (WPS 4.6 NCEP GFS); a mapped depth then rejects.
    An explicit volumetric-water maximum uses the shared Torch normalization.
    """
    return _read_grib(paths, mapping, surface=True, valid_time=valid_time, reference_time=reference_time, member=member)


def read_grib_ozone(paths: SourcePaths, mapping: dict[str, str] | None = None, *,
                     valid_time: datetime | None = None, reference_time: datetime | None = None,
                     member: int | None = None, hybrid_convention: str | None = None) -> OzoneSource:
    """Read isobaric ozone with an explicit quantity.mass_basis declaration.

    Use mass_fraction_of_ozone_in_air.mass_basis=moist_air with collocated
    specific humidity, or ozone_mass_mixing_ratio.mass_basis=dry_air without it.
    ShortName mappings never infer a basis, override units or perform conversion.
    The shared Torch tracer initializer converts moist fractions before remapping.
    Explicit ifs-arithmetic also accepts complete hybrid layers and ap/b interfaces,
    with collocated surface pressure or explicitly mapped ln(Pa); no height is needed.
    """
    return _read_grib(paths, mapping, ozone=True, valid_time=valid_time, reference_time=reference_time, member=member,
                       hybrid_convention=hybrid_convention)


def read_grib_ufs_surface(paths: SourcePaths, mapping: dict[str, str] | None = None, *,
                         valid_time: datetime | None = None, reference_time: datetime | None = None,
                         member: int | None = None) -> UfsSurfaceSource:
    """Read all analyzed UFS quantities with explicit mappings, never fills.

    Map every quantity to shortName or a table-qualified numeric selector.
    Fields are surface points, except fixed 2m T/q and explicitly located deep
    soil temperature. Unknown codec units require quantity.units; known units
    cannot be overridden. Percent albedos convert to fractions in Torch.
    """
    return _read_grib(paths, mapping, ufs_surface=True,
                      valid_time=valid_time, reference_time=reference_time, member=member)


def _read_grib(paths, mapping, *, near_surface=False, surface=False, ozone=False, ufs_surface=False, include_relative_humidity=False,
                valid_time=None, reference_time=None, member=None, hybrid_convention=None):
    try:
        import eccodes as codes
    except ImportError as exc:
        raise RuntimeError("GRIB decoding requires mpas-init[grib] and its ecCodes native library") from exc
    except RuntimeError as exc:
        raise RuntimeError("ecCodes native library unavailable; install mpas-init[grib] or configure ECCODES_DIR") from exc
    mapping = {} if mapping is None else mapping
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("GRIB mappings must contain string quantity names and shortName values")
    valid_time, reference_time = _utc_selection(valid_time, "valid time"), _utc_selection(reference_time, "reference time")
    if member is not None and (isinstance(member, bool) or not isinstance(member, Integral)):
        raise ValueError("GRIB member selection must be an integer label")
    atmosphere = not (near_surface or surface or ozone or ufs_surface)
    convention = "ifs-vapour-only" if atmosphere else "ifs-arithmetic" if ozone else None
    if hybrid_convention is not None and (convention is None or hybrid_convention != convention):
        raise ValueError("unsupported GRIB hybrid reconstruction convention")
    hybrid = hybrid_convention is not None
    if hybrid and atmosphere and ("geopotential" in mapping or "geopotential_height" in mapping):
        raise ValueError("supplied height/geopotential conflicts with GRIB interface reconstruction")
    if atmosphere and "geopotential" in mapping and "geopotential_height" in mapping:
        raise ValueError("map exactly one GRIB height or geopotential quantity")
    if include_relative_humidity:
        humidity_reference = _humidity_reference(None, mapping)
    if ufs_surface:
        names = set(UFS_SURFACE_VARIABLES)
        if any(not mapping.get(name) for name in names):
            raise ValueError("GRIB analyzed UFS surface requires an explicit mapping for every quantity")
        soil_type = mapping.get("deep_soil_temperature.type_of_level")
        try:
            soil_level = float(mapping["deep_soil_temperature.level"])
        except (KeyError, ValueError) as exc:
            raise ValueError("GRIB deep soil temperature requires an explicit point type_of_level and level") from exc
        if (soil_type not in ("surface", "depthBelowLand") or not math.isfinite(soil_level)
                or (soil_level != 0 if soil_type == "surface" else soil_level <= 0)):
            raise ValueError("GRIB deep soil temperature requires surface level zero or positive depthBelowLand")
        ufs_levels = {name: ("surface", 0.) for name in names}
        ufs_levels.update({name: ("heightAboveGround", 2.) for name in ("air_temperature_2m", "specific_humidity_2m")})
        ufs_levels["deep_soil_temperature"] = soil_type, soil_level
        for name in names:
            if name + ".units" in mapping and mapping[name + ".units"] not in _UFS_UNITS[name]:
                raise ValueError("unsupported GRIB units declaration for " + name)
    elif surface:
        names = {standard for standard, _ in SURFACE_VARIABLES.values()}
        snow_convention = _snow_convention(mapping)
        if snow_convention is not None:
            names.remove("surface_snow_thickness")
    elif near_surface:
        names = set(_NEAR_HEIGHTS)
    elif ozone:
        if {"ozone_mass_mixing_ratio", "mass_fraction_of_ozone_in_air"}.issubset(mapping):
            raise ValueError("conflicting ozone quantity mappings")
        dry = "ozone_mass_mixing_ratio" in mapping
        ozone_name = "ozone_mass_mixing_ratio" if dry else "mass_fraction_of_ozone_in_air"
        basis = "dry_air" if dry else "moist_air"
        if (mapping.get(ozone_name + ".mass_basis") != basis
                or any(key.endswith(".mass_basis") and key != ozone_name + ".mass_basis" for key in mapping)):
            raise ValueError("GRIB ozone requires one explicit, consistent quantity.mass_basis declaration")
        names = {ozone_name} if dry else {ozone_name, "specific_humidity"}
        if hybrid:
            names |= {"surface_air_pressure", "logarithm_of_surface_air_pressure"}
    elif hybrid:
        names = set(REQUIRED) | _HYBRID_SURFACE
    else:
        names = set(REQUIRED) | {"geopotential", "surface_air_pressure"}
    if atmosphere and "geopotential" in mapping:
        names.remove("geopotential_height")
    elif atmosphere and "geopotential_height" in mapping:
        names.remove("geopotential")
    if include_relative_humidity:
        names.add("relative_humidity")
    selectors = {name: _parameter_selector(mapping[name]) for name in names if name in mapping}
    paths = (paths,) if isinstance(paths, (str, Path)) else tuple(paths)
    paths = tuple(Path(path).resolve() for path in paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("GRIB source paths must be nonempty and unique")
    initial_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    records = []
    identities = set()
    with ExitStack() as stack:
        streams = {path: stack.enter_context(path.open("rb")) for path in paths}
        for path, stream in streams.items():
            # Rejected multi-field messages can leave unread subfields in C state.
            stack.callback(codes.codes_grib_multi_support_reset_file, stream)
            offsets = set()
            while (handle := codes.codes_grib_new_from_file(stream)) is not None:
                try:
                    offset = codes.codes_get_message_offset(handle)
                    if offset in offsets:
                        raise ValueError("GRIB multi-field messages are not implemented; supply single-field messages")
                    offsets.add(offset)
                    cf_name, short_name = codes.codes_get(handle, "cfName"), codes.codes_get(handle, "shortName")
                    matches = [name for name in names if (_matches_parameter(codes, handle, selectors[name], short_name) if name in selectors else
                               cf_name == ("geopotential" if hybrid and name == "surface_geopotential" else name))]
                    if not matches:
                        continue
                    level_type = codes.codes_get(handle, "typeOfLevel")
                    allowed = (("surface", "heightAboveGround", "depthBelowLand") if ufs_surface else
                               ("surface", "hybrid") if hybrid else ("surface", "depthBelowLandLayer") if surface
                               else ("heightAboveGround",) if near_surface else
                               ("surface", "isobaricInPa", "isobaricInhPa") if atmosphere else
                               ("isobaricInPa", "isobaricInhPa"))
                    if level_type not in allowed:
                        continue
                    level = codes.codes_get(handle, "level", float)
                    if ufs_surface:
                        matches = [name for name in matches if (level_type, level) == ufs_levels[name]]
                        if not matches:
                            continue
                    if atmosphere and not hybrid:
                        matches = [name for name in matches if (name == "surface_air_pressure" and
                                   level_type == "surface" and level == 0) or
                                   (name != "surface_air_pressure" and level_type != "surface")]
                        if not matches:
                            continue
                    if surface:
                        matches = [name for name in matches if (name in _SOIL) == (level_type == "depthBelowLandLayer")]
                        if not matches:
                            continue
                    if near_surface:
                        matches = [name for name in matches if level == _NEAR_HEIGHTS[name]]
                        if not matches:
                            continue
                    identity = _identity(codes, handle)
                    if any(value is not None and value != actual for value, actual in (
                            (valid_time, identity[0]), (reference_time, identity[1]), (member, identity[3]))):
                        continue
                    if hybrid:
                        matches = [name for name in matches if (name in _HYBRID_SURFACE and
                                   ((level_type == "surface" and level == 0) or (level_type == "hybrid" and level == 1)))
                                   or (name not in _HYBRID_SURFACE and level_type == "hybrid")]
                        if not matches:
                            raise ValueError("GRIB IFS reconstruction requires model layers and surface auxiliaries at surface/level 1")
                        if level_type == "hybrid" and (not math.isfinite(level) or level < 1 or level != int(level)):
                            raise ValueError("GRIB hybrid layers require positive integer IDs")
                    if len(matches) != 1:
                        raise ValueError("GRIB record maps to more than one physical quantity")
                    name = matches[0]
                    if codes.codes_get(handle, "stepType") != "instant":
                        raise ValueError("GRIB fields require instantaneous point samples")
                    units = codes.codes_get(handle, "units")
                    if ufs_surface:
                        declared = mapping.get(name + ".units")
                        if units in ("unknown", "~", ""):
                            units = declared
                        elif declared is not None and declared != units:
                            raise ValueError("known GRIB units cannot be overridden for " + name)
                    if units not in (_UFS_UNITS if ufs_surface else _UNITS)[name]:
                        raise ValueError("unsupported GRIB units for " + name)
                    if name in ("eastward_wind", "northward_wind") and codes.codes_get(handle, "uvRelativeToGrid") != 0:
                        raise ValueError("GRIB winds must be earth-relative; grid-relative winds require an explicit rotation")
                    if ufs_surface:
                        pressure = 0.
                    elif surface:
                        pressure = _soil_bounds(codes, handle) if name in _SOIL else 0.
                    elif (atmosphere or hybrid) and name in _HYBRID_SURFACE:
                        pressure = 0.
                    else:
                        pressure = level * (100. if level_type == "isobaricInhPa" else 1.)
                    if not (surface or ufs_surface) and name not in _HYBRID_SURFACE and (not math.isfinite(pressure) or pressure <= 0):
                        raise ValueError("GRIB pressure levels must be finite and positive")
                    identities.add(identity)
                    records.append((path, offset, codes.codes_get(handle, "totalLength", int),
                                    name, pressure, identity))
                finally:
                    codes.codes_release(handle)
        if hybrid and not records:
            raise ValueError("GRIB hybrid reconstruction requires complete model-level fields")
        if len(identities) != 1:
            raise ValueError("select exactly one GRIB forecast time/cycle/member identity; found " + str(len(identities)))
        if hybrid:
            present = {name for _, _, _, name, _, _ in records}
            pressure_names = present & {"surface_air_pressure", "logarithm_of_surface_air_pressure"}
            if len(pressure_names) != 1:
                raise ValueError("GRIB IFS reconstruction requires exactly one surface pressure or explicit ln(Pa)")
            surface_pressure_name = pressure_names.pop()
        if ufs_surface:
            height_name, expected = "aerodynamic_roughness_length", names
        elif surface:
            height_name, expected = "land_binary_mask", names
        elif near_surface:
            height_name, expected = "air_temperature", set(_NEAR_HEIGHTS)
        elif ozone:
            height_name, expected = ozone_name, names - {"surface_air_pressure", "logarithm_of_surface_air_pressure"}
            if hybrid:
                expected = expected | {surface_pressure_name}
        elif hybrid:
            if "geopotential_height" in present:
                raise ValueError("supplied height conflicts with GRIB interface reconstruction")
            height_name = "air_temperature"
            expected = set(REQUIRED) - {"geopotential_height"} | {"surface_geopotential", surface_pressure_name}
        else:
            heights = {name for _, _, _, name, _, _ in records} & {"geopotential_height", "geopotential"}
            if len(heights) != 1:
                raise ValueError("GRIB requires exactly one supplied height or geopotential quantity; map one explicitly")
            height_name = heights.pop()
            expected = set(REQUIRED) - {"geopotential_height"} | {height_name}
            if any(name == "surface_air_pressure" for _, _, _, name, _, _ in records):
                expected.add("surface_air_pressure")
        if include_relative_humidity:
            expected.add("relative_humidity")
        by_field = {name: {} for name in expected}
        for path, offset, length, name, pressure, _ in records:
            # Isobaric surface pressure is unique by normalized grid, not level 0.
            auxiliary = atmosphere and not hybrid and name == "surface_air_pressure"
            key = len(by_field[name]) if auxiliary else pressure
            if key in by_field[name]:
                raise ValueError(f"duplicate GRIB record for {name} at {pressure} {'m' if near_surface or surface else 'Pa'}")
            by_field[name][key] = (path, offset, length)
        levels = sorted(by_field[height_name], reverse=not hybrid)
        if ufs_surface and any(len(field) != 1 for field in by_field.values()):
            raise ValueError("GRIB analyzed UFS surface requires every complete point quantity at its declared level")
        if near_surface and any(len(field) != 1 for field in by_field.values()):
            raise ValueError("GRIB near-surface source requires all four complete 2m/10m fields")
        if surface:
            soil_bounds = sorted(by_field["soil_temperature"])
            if (len(soil_bounds) < 2 or set(soil_bounds) != set(by_field["volume_fraction_of_condensed_water_in_soil"])
                    or any(len(field) != 1 for name, field in by_field.items() if name not in _SOIL)):
                raise ValueError("GRIB surface requires all eight quantities, or seven with a declared snow convention, "
                                 "and matching complete soil layer bounds")
            if (soil_bounds[0][0] != 0 or any(a < 0 or b <= a for a, b in soil_bounds)
                    or any(a[1] != b[0] for a, b in zip(soil_bounds, soil_bounds[1:]))):
                raise ValueError("GRIB soil layers require contiguous positive bounds from zero depth")
        if hybrid and (levels != list(range(1, len(levels) + 1))
                       or any(len(by_field[name]) != 1 for name in expected & _HYBRID_SURFACE)):
            raise ValueError("GRIB IFS reconstruction requires every model layer 1..N and both surface auxiliaries")
        if (atmosphere or ozone) and (len(levels) < 2 or any(set(field) != set(levels)
                for name, field in by_field.items() if name not in _HYBRID_SURFACE)):
            raise ValueError("GRIB required fields need identical complete pressure levels, including at least two heights")
        grids, geometry, fields, coefficients, masked_fields = {}, {}, {}, None, set()
        surface_pressures = []
        # CFFI owns C stdio state for the catalog streams. Python seek cannot reset
        # that EOF/buffering state; use separate byte streams for indexed messages.
        payloads = {path: stack.enter_context(path.open("rb")) for path in paths}
        for name, field in by_field.items():
            auxiliary = atmosphere and not hybrid and name == "surface_air_pressure"
            values, field_grid = [], None
            for pressure in sorted(field, reverse=not (surface or hybrid)):
                path, offset, length = field[pressure]
                stream = payloads[path]
                stream.seek(offset)
                payload = stream.read(length)
                if len(payload) != length:
                    raise ValueError("GRIB source changed during decoding")
                handle = codes.codes_new_from_message(payload)
                try:
                    if hybrid and (name not in _HYBRID_SURFACE or codes.codes_get(handle, "NV", int) > 0):
                        if (codes.codes_get(handle, "NV", int) != 2 * (len(levels) + 1)
                                or not codes.codes_is_defined(handle, "pv")):
                            raise ValueError("GRIB IFS pv requires complete ap/b interface coefficients")
                        pv = codes.codes_get_array(handle, "pv")
                        if pv.shape != (2 * (len(levels) + 1),) or not np.isfinite(pv).all():
                            raise ValueError("GRIB IFS pv requires finite complete interface coefficients")
                        if coefficients is not None and not np.array_equal(pv, coefficients):
                            raise ValueError("GRIB hybrid coefficients differ between fields or levels")
                        coefficients = pv
                    missing = codes.codes_get(handle, "numberOfMissing", int)
                    if missing and not ((surface and name != "land_binary_mask") or
                                        (atmosphere and not hybrid and name not in _HYBRID_SURFACE)):
                        raise ValueError("missing GRIB pixels are unsupported for this quantity or vertical convention")
                    key = codes.codes_get(handle, "md5GridSection")
                    if key not in geometry:
                        geometry[key] = _grid(codes, handle)
                    grid, order = geometry[key]
                    if missing and atmosphere:
                        if grid.row_counts is not None:
                            raise ValueError("missing GRIB atmospheric pixels require a rectangular isobaric grid")
                        masked_fields.add(name)
                    if grid.row_counts is not None and (surface or near_surface or ufs_surface):
                        raise ValueError("GRIB surface and near-surface fields require rectangular grids")
                    if not auxiliary and field_grid is not None and not _same_grid(field_grid, grid):
                        raise ValueError("GRIB field grid changes between pressure levels: " + name)
                    field_grid = grid
                    raw = codes.codes_get_values(handle)
                    if raw.size != order.size:
                        raise ValueError("GRIB values must match their grid")
                    if missing:
                        if codes.codes_get(handle, "bitmapPresent", int) != 1:
                            raise ValueError("missing GRIB pixels require an explicit bitmap")
                        bitmap = codes.codes_get_array(handle, "bitmap")
                        if (bitmap.shape != raw.shape or not np.isin(bitmap, (0, 1)).all()
                                or np.count_nonzero(bitmap == 0) != missing):
                            raise ValueError("GRIB bitmap differs from its declared missing pixels")
                        raw = np.where(bitmap == 0, MISSING, raw)
                    if not np.isfinite(raw).all():
                        raise ValueError("GRIB values must be finite and match their grid")
                    ordered = raw[order]
                    if grid.row_counts is None:
                        ordered = ordered.reshape(grid.latitude.numel(), grid.longitude.numel())
                    decoded = torch.tensor(ordered.copy())
                    if ufs_surface and name.endswith("albedo"):
                        units = codes.codes_get(handle, "units")
                        if units in ("unknown", "~", ""):
                            units = mapping.get(name + ".units")
                        if units == "%":
                            decoded = decoded / 100.
                    if auxiliary:
                        if any(_same_grid(own, grid) for _, own in surface_pressures):
                            raise ValueError("duplicate GRIB surface_air_pressure on the same normalized grid")
                        surface_pressures.append((decoded, grid))
                    else:
                        values.append(decoded)
                finally:
                    codes.codes_release(handle)
            if auxiliary:
                continue
            stacked = ((atmosphere or ozone) and name not in _HYBRID_SURFACE) or name in _SOIL
            fields[name], grids[name] = torch.stack(values) if stacked else values[0], field_grid
    if any((path.stat().st_size, path.stat().st_mtime_ns) != initial_stats[path] for path in paths):
        raise ValueError("GRIB source changed during decoding")
    grid = grids[height_name]
    pressure = torch.tensor(levels, dtype=torch.float64)
    if hybrid:
        if any(not _same_grid(grid, grids[name]) for name in expected
               if name in _HYBRID_SURFACE or name == "specific_humidity"):
            raise ValueError("GRIB IFS reconstruction requires collocated thermodynamic and surface grids")
        ps = fields.pop(surface_pressure_name)
        if surface_pressure_name == "logarithm_of_surface_air_pressure":
            ps = torch.exp(ps)
        ap, b = torch.tensor(coefficients.copy()).chunk(2)
        interfaces = hybrid_pressure(ap, b, ps, interfaces=True)
        if not torch.all(interfaces[-1] == ps):
            raise ValueError("bottom IFS interface must equal supplied surface pressure")
        if ozone:
            pressure = ifs_full_level_pressure(interfaces)
        else:
            pressure, fields["geopotential_height"] = ifs_hydrostatic_columns(
                interfaces, fields["air_temperature"], fields["specific_humidity"], fields.pop("surface_geopotential"))
        grids = {name: own for name, own in grids.items() if name not in _HYBRID_SURFACE}
    if height_name == "geopotential":
        value = fields.pop("geopotential")
        fields["geopotential_height"] = torch.where(value == MISSING, value, geopotential_to_height(value))
        grids["geopotential_height"] = grids.pop("geopotential")
        if "geopotential" in masked_fields:
            masked_fields.remove("geopotential")
            masked_fields.add("geopotential_height")
        height_name = "geopotential_height"
    field_surface_pressure = {}
    if atmosphere and not hybrid:
        for name in masked_fields:
            candidates = [ps for ps, own in surface_pressures if _same_grid(grids[name], own)]
            if len(candidates) != 1:
                raise ValueError("missing GRIB atmospheric pixels require one complete collocated surface_air_pressure field")
            ps = candidates[0]
            above = pressure[:, None, None] <= ps
            if (not torch.all(torch.isfinite(ps) & (ps > 0)) or not torch.all(above.sum(0) >= 2)
                    or torch.any((fields[name] == MISSING) & above)):
                raise ValueError("missing above-ground GRIB atmospheric data or invalid surface pressure: " + name)
            field_surface_pressure[name] = ps
    if include_relative_humidity:
        _normalize_humidity(fields, grids, grid, humidity_reference)
    field_grids = {name: own for name, own in grids.items() if name != height_name and not _same_grid(own, grid)}
    identity = identities.pop()
    if ufs_surface:
        return UfsSurfaceSource(grid.latitude, grid.longitude, fields, *identity[:3], identity[3], field_grids or None)
    if ozone:
        if not dry and not _same_grid(grid, grids["specific_humidity"]):
            raise ValueError("ozone requires collocated specific humidity and pressure coordinates")
        return OzoneSource(pressure, grid.latitude, grid.longitude,
                           fields[ozone_name], basis, None if dry else fields["specific_humidity"],
                           *identity[:3], identity[3], grid.grid_type, grid.row_counts, hybrid_convention)
    if surface:
        renamed = {standard: name for name, (standard, _) in SURFACE_VARIABLES.items()}
        fields = {renamed[name]: value for name, value in fields.items()}
        field_grids = {renamed[name]: value for name, value in field_grids.items()}
        if MASKED_SOURCE_FIELDS & field_grids.keys():
            raise ValueError("GRIB soil/ice fields require their collocated primary landmask grid")
        if not torch.all((fields["landmask"] == 0) | (fields["landmask"] == 1)):
            raise ValueError("GRIB source landmask must be binary")
        fields["soil_moisture"] = soil_water_maximum(fields["soil_moisture"], _soil_maximum(None, mapping))
        policies = {name: _snow_policy(None, mapping, SURFACE_VARIABLES[name][0]) for name in ("snow_water_equivalent", "snow_depth")}
        _normalize_surface_snow(fields, field_grids, grid, policies)
        declarations = _apply_snow_convention(fields, field_grids, None, snow_convention)
        return SurfaceSource(grid.latitude, grid.longitude, torch.tensor(soil_bounds, dtype=torch.float64), fields,
                             *identity[:3], grid.grid_type, identity[3], field_grids=field_grids or None,
                             declarations=declarations)
    if near_surface:
        return NearSurfaceSource(grid.latitude, grid.longitude, fields, *identity[:3], grid.grid_type,
                                 identity[3], field_grids=field_grids or None)
    return PressureLevelSource(pressure, grid.latitude, grid.longitude, fields,
                               *identity[:3], grid.grid_type, identity[3], field_grids=field_grids or None,
                               row_counts=grid.row_counts, vertical_convention=hybrid_convention,
                               field_surface_pressure=field_surface_pressure or None)
