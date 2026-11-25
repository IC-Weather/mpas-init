"""Native dycore NetCDF serialization. Host copies occur only at this I/O boundary."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import torch
from torch import Tensor

from mpas_init.mesh import GlobalStatic
from mpas_init.mesh import _validate_surface, _validate_climatology, _validate_soil_temperature, _validate_classic_gwd, _validate_ugwp, CLASSIC_GWD_UNITS, UGWP_UNITS
from mpas_init.diagnostics import DIAGNOSTIC_FIELDS
from mpas_init.surface import SURFACE_FIELDS
from mpas_init.tracers import UFS_TRACER_FIELDS
from mpas_init.profiles import NATIVE_DIMENSIONS, NATIVE_ZERO_DIMENSIONS, NATIVE_GWD_FIELDS, NATIVE_UNITS, UFS_STATIC_FIELDS
from mpas_init.gwd import CLASSIC_FIELDS
from mpas_init.ugwp import UGWP_FIELDS
from mpas_init.ufs_surface import UFS_SURFACE_SOURCES, UFS_SURFACE_UNITS


FIELD_DIMENSIONS = {
    **{name: ("Time", "nCells", "nVertLevels") for name in
       ("rho", "theta", "qv", "qc", "qr", "qi", "qs", "qg", "rho_base", "theta_base",
        "uReconstructZonal", "uReconstructMeridional")},
    "u": ("Time", "nEdges", "nVertLevels"), "w": ("Time", "nCells", "nVertLevelsP1"),
    "surface_pressure": ("Time", "nCells"), "precipw": ("Time", "nCells"),
    "zgrid": ("nCells", "nVertLevelsP1"), "zz": ("nCells", "nVertLevels"),
    "zxu": ("nEdges", "nVertLevels"), "dss": ("nCells", "nVertLevels"), "t_init": ("nCells", "nVertLevels"),
    **{name: ("nEdges", "TWO", "nVertLevelsP1") for name in ("zb", "zb3")},
    **{name: ("nVertLevels",) for name in ("rdzw", "dzu", "rdzu", "fzm", "fzp", "u_init", "v_init", "qv_init")},
    **{name: () for name in ("cf1", "cf2", "cf3")},
    "edgeNormalVectors": ("nEdges", "R3"), "localVerticalUnitVectors": ("nCells", "R3"),
    "cellTangentPlane": ("nCells", "TWO", "R3"), "coeffs_reconstruct": ("nCells", "maxEdges", "R3"),
}
SURFACE_DIMENSIONS = {
    **{name: ("Time", "nCells", "nSoilLevels") for name in ("tslb", "smois", "sh2o", "zs", "dzs")},
    **{name: ("Time", "nCells") for name in
       ("tmn", "skintemp", "sst", "xice", "seaice", "snow", "snowc", "snowh", "xland", "vegfra", "sfc_albbck")},
    **{name: ("nCells",) for name in ("ivgtyp", "isltyp", "snoalb", "shdmin", "shdmax")},
}
DIAGNOSTIC_DIMENSIONS = {name: ("Time", "nCells", "nVertLevels") if name == "relhum" else ("Time", "nCells")
                          for name in ("relhum", "t2m", "rh2", "q2", "u10", "v10")}


def inventory_record(fields, source, ic_path):
    """Serialize inventory tensors at the file boundary, not in the tensor kernel."""
    from mpas_init.horizontal import MISSING
    from mpas_init.inventory import WATER_SPECIES, atmosphere_inventory

    water = {name: fields[name] for name in WATER_SPECIES if name in fields}
    inventory = atmosphere_inventory(fields["areaCell"], fields["rho"], fields["zgrid"], water)
    height = source.fields["geopotential_height"]
    valid = torch.isfinite(height) & (height != MISSING)
    torch._assert_async(valid.any(), "inventory source has no valid height samples")
    height_range = torch.stack((torch.where(valid, height, torch.inf).amin(),
                                torch.where(valid, height, -torch.inf).amax()))
    return {"schema": "mpas-init-atmospheric-inventory-v1", "ic_file": str(Path(ic_path).resolve()),
            "valid_time": source.valid_time.isoformat(), "reference_time": source.reference_time.isoformat(),
            "lead_seconds": source.lead.total_seconds(), "member": source.member,
            "source_sample_height_range_m": height_range.detach().cpu().tolist(),
            "source_inventory": {"status": "unavailable",
                                 "reason": "Canonical source point samples do not expose finite-volume layer means, cell areas, and vertical bounds."},
            "target_inventory": {"method": "Shallow-atmosphere rho_dry * dz * area; water mass per dry-air mass",
                                 "water_species": list(water),
                                 "totals": {name: value.detach().cpu().tolist() for name, value in inventory.items() if value.ndim == 0}},
            "mass_correction_applied": False,
            "comparison": "Source sample heights are not cell bounds. Target terrain/top need not match; no source/target mass-closure ratio is defined."}


def write_inventory_report(path, record):
    """Bind a report to a closed IC and create the JSON sidecar exclusively."""
    ic_path = Path(record["ic_file"])
    with ic_path.open("rb") as handle:
        identity = {"bytes": ic_path.stat().st_size, "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    payload = json.dumps({**record, "ic_identity": identity}, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="ascii") as handle:
        handle.write(payload)


def write_dycore(path: str | Path, fields: dict[str, Tensor], static: GlobalStatic,
                  valid_time: datetime, *, surface: dict[str, Tensor] | None = None,
                  diagnostics: dict[str, Tensor] | None = None,
                  tracers: dict[str, Tensor] | None = None, ozone_extension: str = "reject", ozone_polar_extension: str = "reject",
                  ozone_negative_source: str = "reject", ozone_vertical_convention: str | None = None,
                  native: dict[str, Tensor] | None = None, landuse_name: str | None = None,
                  ufs_surface: dict[str, Tensor] | None = None, ufs_surface_polar_extension: str = "reject",
                  ufs_static: dict[str, Tensor] | None = None, source_vertical_convention: str | None = None,
                  source_forecast_identity: tuple[datetime, timedelta, int | None] | None = None,
                  ugwp_polar_extension: tuple[str, str] | None = None,
                  ufs_surface_declarations: dict[str, dict[str, str | None]] | None = None,
                  surface_declarations: dict[str, dict[str, str | None]] | None = None) -> None:
    """Write a new file exclusively; reject partial or nonfinite field inventories."""
    if surface_declarations and (surface is None or not isinstance(surface_declarations, dict)):
        raise ValueError("surface source declarations require surface output")
    if ufs_surface_declarations and (ufs_surface is None or not isinstance(ufs_surface_declarations, dict)):
        raise ValueError("analyzed UFS surface declarations require analyzed UFS surface output")
    if ugwp_polar_extension is not None and (ufs_static is None or not isinstance(ugwp_polar_extension, tuple)
            or len(ugwp_polar_extension) != 2
            or any(not isinstance(value, str) or not value.strip() for value in ugwp_polar_extension)):
        raise ValueError("UGWP polar provenance requires complete UFS statics and two nonempty convention strings")
    if valid_time.microsecond:
        raise ValueError("MPAS initialization time must use whole seconds")
    if ozone_polar_extension not in ("reject", "clamp") or tracers is None and ozone_polar_extension != "reject":
        raise ValueError("ozone polar extension requires tracer output and reject or clamp")
    if ozone_negative_source not in ("reject", "legacy") or tracers is None and ozone_negative_source != "reject":
        raise ValueError("ozone negative source policy requires tracer output and reject or legacy")
    if source_vertical_convention not in (None, "ifs-vapour-only"):
        raise ValueError("unsupported source vertical convention")
    if ozone_vertical_convention not in (None, "ifs-arithmetic") or tracers is None and ozone_vertical_convention is not None:
        raise ValueError("ozone vertical convention requires tracer output and explicit ifs-arithmetic")
    if source_forecast_identity is not None:
        reference, lead, member = source_forecast_identity
        if (not isinstance(reference, datetime) or not isinstance(lead, timedelta)
                or reference.utcoffset() is None or valid_time.utcoffset() is None or lead < timedelta(0)
                or reference + lead != valid_time
                or member is not None and (isinstance(member, bool) or not isinstance(member, Integral))):
            raise ValueError("inconsistent source forecast identity")
    dimensions = {**static.dimensions, **FIELD_DIMENSIONS}
    if set(fields) != set(dimensions):
        raise ValueError(f"dycore inventory mismatch: missing={sorted(set(dimensions)-set(fields))}, extra={sorted(set(fields)-set(dimensions))}")
    nlevels = fields["rho"].shape[-1]
    sizes = {**static.sizes, "nVertLevels": nlevels, "nVertLevelsP1": nlevels + 1, "StrLen": 64, "R3": 3}
    if native is None and landuse_name is not None:
        raise ValueError("landuse_name requires the native Noah profile")
    if native is not None and (surface is None or diagnostics is None or (tracers is not None and ufs_static is None)):
        raise ValueError("native Noah requires complete surface and diagnostics, without UFS tracer-prefix fields")
    if ufs_static is not None and any(value is None for value in (native, surface, diagnostics, tracers, ufs_surface)):
        raise ValueError("complete UFS profile requires native, surface, diagnostics, tracers, and analyzed surface state")
    if ufs_surface is not None and native is None:
        raise ValueError("analyzed UFS surface output requires the native Noah file contract")
    if surface is not None:
        if set(surface) != SURFACE_FIELDS:
            raise ValueError("surface inventory mismatch")
        for name, value in surface.items():
            shape = (sizes["nCells"], 4) if name in ("tslb", "smois", "sh2o", "smcrel", "zs", "dzs") else (sizes["nCells"],)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid or incomplete surface output field: {name}")
        # smcrel is internal cold-start state, not a native initialization-file variable.
        fields = {**fields, **{name: surface[name] for name in SURFACE_DIMENSIONS}}
        dimensions.update(SURFACE_DIMENSIONS)
        sizes["nSoilLevels"] = 4
    if diagnostics is not None:
        if set(diagnostics) != DIAGNOSTIC_FIELDS:
            raise ValueError("diagnostic inventory mismatch")
        fields = {**fields, **diagnostics}
        dimensions.update(DIAGNOSTIC_DIMENSIONS)
        if torch.any(diagnostics["t2m"] <= 0) or torch.any(diagnostics["q2"] < 0):
            raise ValueError("invalid near-surface temperature or moisture")
    if tracers is not None:
        if (set(tracers) != UFS_TRACER_FIELDS or ozone_extension not in ("reject", "constant")
                or ozone_polar_extension not in ("reject", "clamp")):
            raise ValueError("invalid UFS tracer inventory or ozone extension")
        if any(torch.any(value < 0) for value in tracers.values()):
            raise ValueError("negative UFS tracer output")
        seed = tracers["sgs_tke"].new_tensor(1.e-4, dtype=torch.float32).to(tracers["sgs_tke"].dtype)
        if (torch.any(tracers["ni"] != 0) or torch.any(tracers["nr"] != 0)
                or torch.any(tracers["sgs_tke"] != seed)
                or any(torch.any(fields[name] != 0) for name in ("qc", "qr", "qi", "qs", "qg"))):
            raise ValueError("UFS tracer prefix violates its declared cold-start policy")
        fields = {**fields, **tracers}
        dimensions.update({name: ("Time", "nCells", "nVertLevels") for name in UFS_TRACER_FIELDS})
    if native is not None:
        if set(native) != set(NATIVE_DIMENSIONS):
            raise ValueError("incomplete native Noah output inventory")
        ice, water = (24, 16) if landuse_name == "USGS" else (15, 17)
        ocean = native["landmask"] == 0
        # Seasonal sea ice changes surface categories, not the prepared geographic mask or climatology.
        categories = {"lu_index": torch.where(ocean, water, surface["ivgtyp"]),
                      "soilcat_top": torch.where(ocean, 14, surface["isltyp"]), "landmask": native["landmask"],
                      "landusef": native["landusef"].T, "soilf": native["soilf"].T}
        sizes["nlcat"], sizes["nscat"] = _validate_surface(categories, sizes["nCells"], landuse_name)
        sizes["nMonths"] = 12
        climatology = {**{name: native[name].T for name in ("greenfrac", "albedo12m", "lai12m")},
                       "snoalb": torch.where(ocean, 0.0, surface["snoalb"]),
                       **{name: surface[name] for name in ("shdmin", "shdmax")}}
        _validate_climatology(climatology, categories, landuse_name)
        _validate_soil_temperature(native["soiltemp"], categories)
        _validate_classic_gwd({name: native[name] for name in NATIVE_GWD_FIELDS}, sizes["nCells"], required=NATIVE_GWD_FIELDS)
        if (native["isice_lu"].dtype not in (torch.int32, torch.int64)
                or native["iswater_lu"].dtype not in (torch.int32, torch.int64)
                or torch.any(native["isice_lu"] != ice) or torch.any(native["iswater_lu"] != water)
                or any(torch.any(native[name] != 0) for name in NATIVE_ZERO_DIMENSIONS)
                or any(torch.any(fields[name] != 0) for name in ("qc", "qr", "qi", "qs", "qg"))):
            raise ValueError("native Noah violates its declared cold-start policy")
        fields = {**fields, **native}
        dimensions.update(NATIVE_DIMENSIONS)
    if ufs_surface is not None:
        if set(ufs_surface) != set(UFS_SURFACE_SOURCES) or ufs_surface_polar_extension not in ("reject", "clamp"):
            raise ValueError("incomplete analyzed UFS surface inventory or polar policy")
        for name, value in ufs_surface.items():
            stored = value.to(torch.float32)
            if value.shape != (sizes["nCells"],) or not torch.isfinite(value).all() or not torch.isfinite(stored).all():
                raise ValueError(f"invalid analyzed UFS surface output: {name}")
            if torch.any(value < 0) or (name in ("ufs_sfc_znt", "ufs_sfc_z0", "ufs_sfc_t2m", "ufs_sfc_tg3", "ufs_sfc_tisfc") and torch.any(stored <= 0)):
                raise ValueError(f"negative or nonpositive analyzed UFS surface output: {name}")
            if name in ("ufs_sfc_alvsf", "ufs_sfc_alnsf", "ufs_sfc_alvwf", "ufs_sfc_alnwf") and torch.any(value > 1):
                raise ValueError(f"analyzed UFS surface fraction exceeds one: {name}")
            if name == "ufs_sfc_q2" and torch.any(stored >= 1):
                raise ValueError("analyzed UFS specific humidity must remain below one after serialization")
        eligible = (native["landmask"] == 1) & (surface["ivgtyp"] != native["isice_lu"])
        if torch.any(ufs_surface["ufs_sfc_canwat"][~eligible] != 0) or not torch.equal(ufs_surface["ufs_sfc_znt"], ufs_surface["ufs_sfc_z0"]):
            raise ValueError("analyzed UFS surface violates canopy ownership or common roughness")
        fields = {**fields, **ufs_surface}
        dimensions.update({name: ("Time", "nCells") for name in UFS_SURFACE_SOURCES})
    if ufs_static is not None:
        if set(ufs_static) != set(UFS_STATIC_FIELDS):
            raise ValueError("incomplete UFS GWD output inventory")
        _validate_classic_gwd({name: ufs_static[name] for name in CLASSIC_FIELDS}, sizes["nCells"])
        _validate_ugwp({name: ufs_static[name] for name in UGWP_FIELDS}, sizes["nCells"])
        if any(not torch.equal(ufs_static[name], native[name]) for name in NATIVE_GWD_FIELDS):
            raise ValueError("native and UFS shared GWD statics differ")
        if torch.any(ufs_static["gwd_elvmax"] < fields["ter"]):
            raise ValueError("UFS sampled maximum is below final model terrain")
        for name in ("o3mr", "sgs_tke"):
            if not torch.isfinite(fields[name].to(torch.float32)).all():
                raise ValueError("invalid float32 UFS tracer output: " + name)
        fields = {**fields, **ufs_static}
        dimensions.update({name: ("nCells",) for name in UFS_STATIC_FIELDS})
    for name, value in fields.items():
        dims = tuple(dim for dim in dimensions[name] if dim != "Time")
        if value.shape != tuple(sizes[dim] for dim in dims) or not torch.isfinite(value).all():
            raise ValueError(f"invalid or incomplete output field: {name}")
    for name in ("rho", "rho_base", "theta", "theta_base", "surface_pressure", "zz"):
        if torch.any(fields[name] <= 0):
            raise ValueError(f"nonpositive output field: {name}")
    if torch.any(fields["qv"] < 0) or torch.any(fields["zgrid"][:, 1:] <= fields["zgrid"][:, :-1]):
        raise ValueError("negative water vapor or nonmonotonic target heights")
    stamp = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
    path = Path(path)
    # NetCDF NC_NOCLOBBER protects existing user files, including a sole previous result.
    dataset = Dataset(path, "w", clobber=False, format="NETCDF3_64BIT_OFFSET")
    try:
        with dataset as ds:
            for name in sorted({dim for dims in dimensions.values() for dim in dims} | {"Time", "StrLen"}):
                ds.createDimension(name, None if name == "Time" else sizes[name])
            ds.setncatts({"model_name": "mpas", "core_name": "init_atmosphere", "Conventions": "MPAS",
                          "on_a_sphere": "YES", "sphere_radius": static.sphere_radius, "is_periodic": "NO",
                          "config_start_time": stamp, "config_theta_adv_order": 3, "config_coef_3rd_order": 0.25,
                          "config_interface_projection": "linear_interpolation",
                          "mpas_init_profile": ("global-dycore" + ("-surface" if surface is not None else "")
                                                + ("-diagnostic" if diagnostics is not None else "")
                                                + ("-tracer" if tracers is not None else "")
                                                + ("-prefix" if surface is not None or diagnostics is not None or tracers is not None else "")),
                          "history": "mpas-init PyTorch pressure-level initialization"})
            if source_vertical_convention is not None:
                ds.mpas_init_source_vertical_convention = source_vertical_convention
                ds.mpas_init_source_height_reconstruction = "ECMWF published vapour-only hydrostatic approximation; condensate loading and nonhydrostatic effects omitted"
            if source_forecast_identity is not None:
                ds.mpas_init_source_reference_time = reference.astimezone(timezone.utc).isoformat()
                ds.mpas_init_source_forecast_period_seconds = lead.total_seconds()
                if member is not None:
                    # Classic NetCDF attributes cannot preserve every int64 realization label numerically.
                    ds.mpas_init_source_realization = str(member)
            if surface is not None:
                ds.mpas_init_cold_start = "native four-layer Noah; land liquid soil water=0; ocean=1; SST from skin temperature"
                if surface_declarations:
                    ds.mpas_init_surface_declarations = json.dumps(surface_declarations, sort_keys=True)
            if tracers is not None:
                ds.mpas_init_tracer_cold_start = "analyzed dry-air ozone; ni=nr=0; original float32 1e-4 m2/s2 MYNN TKE seed"
                ds.mpas_init_ozone_extension = ozone_extension
                if ozone_vertical_convention is not None:
                    ds.mpas_init_ozone_vertical_convention = ozone_vertical_convention
                if ozone_negative_source != "reject":
                    ds.mpas_init_ozone_negative_source = ozone_negative_source
                    ds.mpas_init_ozone_negative_source_rule = "preserve signed source samples without clipping; require finite nonnegative output"
                if ozone_polar_extension != "reject":
                    ds.mpas_init_ozone_polar_extension = ozone_polar_extension
            if native is not None:
                ds.mpas_init_profile = "global-ncar-noah-cold-start"
                ds.mpas_init_optional_cold_start = "qc qr qi qs qg qh nc ni nr ns ng nifa nwfa volg refl10cm canwat h_oml_initial dz = 0; dz is unused legacy state, not soil thickness"
                ds.mpas_init_physics_qualification = "complete native file contract; coupled physics and seasonal startup not yet qualified"
            if ufs_surface is not None:
                ds.mpas_init_profile = "global-ncar-noah-cold-start-ufs-surface-prefix"
                ds.mpas_init_surface_polar_extension = ufs_surface_polar_extension
                ds.mpas_init_surface_remap = "periodic bilinear point interpolation; canopy positive on land excluding glacier vegetation"
                if ufs_surface_declarations:
                    ds.mpas_init_ufs_surface_declarations = json.dumps(ufs_surface_declarations, sort_keys=True)
                ds.mpas_init_physics_qualification = "native file contract plus analyzed UFS surface prefix; complete UFS profile and coupled physics not yet qualified"
            if ufs_static is not None:
                ds.mpas_init_profile = "global-ufs-ms-r4-cold-start"
                ds.mpas_init_gwd_elvmax_policy = "max(prepared sampled maximum, final model terrain)"
                ds.mpas_init_physics_qualification = "complete UFS R4 file contract; producer conventions, seasonal cases, and coupled physics not yet qualified"
                if ugwp_polar_extension is not None:
                    ds.mpas_init_ugwp_polar_extension, ds.mpas_init_ugwp_polar_directional_convention = ugwp_polar_extension
            real_storage = "f4" if fields["rho"].dtype == torch.float32 else "f8"
            for name, value in fields.items():
                dims = dimensions[name]
                # Native stream reindexing writes zero in unused global connectivity slots.
                if name in ("cellsOnCell", "edgesOnCell", "verticesOnCell", "edgesOnEdge"):
                    counts = fields["nEdgesOnEdge" if name == "edgesOnEdge" else "nEdgesOnCell"]
                    active = torch.arange(value.shape[-1], device=value.device)[None, :] < counts[:, None]
                    value = torch.where(active, value, 0)
                analyzed_surface_field = ufs_surface is not None and name in UFS_SURFACE_SOURCES
                original_ufs_tracer = ufs_static is not None and name in ("o3mr", "sgs_tke")
                variable = ds.createVariable(name, "f4" if analyzed_surface_field or original_ufs_tracer else real_storage if value.is_floating_point() else "i4", dims)
                if native is not None:
                    variable.units = NATIVE_UNITS.get(name, "unitless")
                if tracers is not None and name in UFS_TRACER_FIELDS:
                    if native is None or name not in ("ni", "nr"):
                        variable.units = "m2 s-2" if name == "sgs_tke" else "kg-1" if name in ("ni", "nr") else "kg kg-1"
                    if name != "sgs_tke":
                        variable.mpas_init_mass_basis = "dry_air"
                if analyzed_surface_field:
                    variable.units = UFS_SURFACE_UNITS[name]
                    variable.mpas_init_quantity = UFS_SURFACE_SOURCES[name]
                    variable.surface_analysis_interpolation = "periodic bilinear"
                    if name == "ufs_sfc_q2":
                        variable.mpas_init_mass_basis = "moist_air"
                if ufs_static is not None and name in UFS_STATIC_FIELDS:
                    variable.units = (CLASSIC_GWD_UNITS | UGWP_UNITS)[name]
                array = value.detach().cpu().numpy()
                variable[:] = array[None] if dims and dims[0] == "Time" else array
            if native is not None:
                classification = np.full(64, b" ", dtype="S1")
                encoded = landuse_name.encode("ascii")
                classification[:len(encoded)] = np.frombuffer(encoded, dtype="S1")
                variable = ds.createVariable("mminlu", "S1", ("StrLen",))
                variable[:] = classification
                variable.units = "unitless"
            chars = np.full(64, b" ", dtype="S1")
            chars[:len(stamp)] = np.frombuffer(stamp.encode("ascii"), dtype="S1")
            ds.createVariable("initial_time", "S1", ("StrLen",))[:] = chars
            ds.createVariable("xtime", "S1", ("Time", "StrLen"))[0] = chars
            if native is not None:
                ds["initial_time"].units = "YYYY-MM-DD_hh:mm:ss"
                ds["xtime"].units = "YYYY-MM-DD_hh:mm:ss"
            time = ds.createVariable("Time", real_storage, ("Time",))
            time.units = "seconds since " + valid_time.strftime("%Y-%m-%d %H:%M:%S")
            time.calendar = "gregorian"
            time[0] = 0.0
    except BaseException:
        path.unlink(missing_ok=True)
        raise
