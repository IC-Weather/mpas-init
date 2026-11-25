"""Command line for the currently implemented global dycore path."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

def _utc_time(value):
    try:
        date = datetime.fromisoformat(value)
        if date.utcoffset() is None:
            raise ValueError("timezone required")
        return date.astimezone(timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use an ISO-8601 timestamp with an explicit timezone") from exc


def _prepare_main(argv) -> None:
    parser = argparse.ArgumentParser(prog="mpas-init prepare", description="Prepare reusable global statics from a global mesh and local geography")
    parser.add_argument("--mesh", required=True, help="Local raw global mesh or managed x1.2562/x1.4002/x1.10242")
    parser.add_argument("--mesh-cache", type=Path, help="Explicit cache directory for managed global meshes")
    parser.add_argument("--cache-dir", type=Path, help="Explicit geometry and completed-static cache; inputs must remain immutable")
    parser.add_argument("--download-mesh", action="store_true", help="Permit an uncached managed mesh download from NCAR")
    parser.add_argument("--nominal-min-dc", type=float, help="Explicit nominal minimum spacing in metres, only if the raw mesh omits nominalMinDc")
    parser.add_argument("--data-root", type=Path, required=True, help="Local native MPAS geography directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("dycore", "native-noah", "ufs-ms"), default="native-noah")
    for name, default in (("supersample", 1), ("landuse-supersample", 1), ("soil-supersample", 1),
                          ("vegetation-supersample", 1), ("block-side", 32), ("row-chunk", 128)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args(argv)
    from mpas_init.prepare import prepare_static

    prepare_static(args.mesh, args.data_root, args.output, profile=args.profile, supersample=args.supersample,
                    landuse_supersample=args.landuse_supersample, soil_supersample=args.soil_supersample,
                    vegetation_supersample=args.vegetation_supersample, block_side=args.block_side, row_chunk=args.row_chunk,
                    mesh_cache=args.mesh_cache, download_mesh=args.download_mesh, nominal_min_dc=args.nominal_min_dc,
                    cache_dir=args.cache_dir)
    print(f"Wrote {args.output}: reusable global {args.profile} statics; no meteorological state")


def main() -> None:
    if sys.argv[1:2] == ['prepare-geography']:
        parser = argparse.ArgumentParser(prog='mpas-init prepare-geography', description='Build a verified local indexed geography collection')
        parser.add_argument('--data-root', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--profile', choices=('dycore', 'native-noah', 'ufs-ms'), default='ufs-ms')
        args = parser.parse_args(sys.argv[2:])
        from mpas_init.prepare import prepare_geography
        record = prepare_geography(args.data_root, args.output, profile=args.profile)
        print(f"Wrote {args.output}: {len(record['datasets'])} verified local geography datasets")
        return
    if sys.argv[1:2] == ["prepare"]:
        _prepare_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Initialize global MPAS from complete CF NetCDF or isobaric GRIB atmospheres")
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, help="JSON mapping from CF standard names to source variable names")
    parser.add_argument("--inventory", type=Path, help="Optional atmospheric mass/water JSON report; never applies a mass correction")
    parser.add_argument("--valid", type=_utc_time, help="Select one valid time, with an explicit timezone")
    parser.add_argument("--cycle", type=_utc_time, help="Select the forecast reference time when valid times repeat")
    parser.add_argument("--member", type=int, help="Select the integer realization label, not its array index")
    parser.add_argument("--hybrid-convention", choices=("ifs-vapour-only",),
                        help="Explicit complete-interface IFS hydrostatic approximation; excludes condensate loading")
    parser.add_argument("--profile", choices=("prefix", "native-noah", "ufs-ms"), default="prefix",
                        help="Write selected prefixes, the native Noah contract, or the complete UFS R4 cold-start file contract")
    parser.add_argument("--surface-source", type=Path, nargs="+", help="Complete CF NetCDF or GRIB surface/soil files; this is not a full physics profile")
    parser.add_argument("--surface-mapping", type=Path, help="JSON surface quantity mappings and explicit snow/soil policies")
    parser.add_argument("--near-surface-source", type=Path, nargs="+", help="Complete CF NetCDF or GRIB 2m/10m files; requires explicit RH saturation reference")
    parser.add_argument("--near-surface-mapping", type=Path, help="JSON mapping from CF near-surface standard names to variable names or GRIB shortName, plus RH convention")
    parser.add_argument("--ozone-source", type=Path, nargs="+", help="CF NetCDF or isobaric GRIB ozone files for the UFS R4 cold-start tracer prefix")
    parser.add_argument("--ozone-mapping", type=Path, help="JSON ozone/humidity mapping; GRIB requires explicit quantity.mass_basis, dry ratios require ozone_mass_mixing_ratio")
    parser.add_argument("--ozone-hybrid-convention", choices=("ifs-arithmetic",),
                        help="Explicit IFS full-level pressure from complete ozone interfaces; no height reconstruction")
    parser.add_argument("--ozone-extension", choices=("reject", "constant"), default="reject",
                        help="Reject insufficient coverage, or explicitly extend endpoint ozone values")
    parser.add_argument("--ozone-polar-extension", choices=("reject", "clamp"), default="reject",
                        help="Reject uncovered ozone polar targets, or explicitly clamp to the nearest source latitude")
    parser.add_argument("--ozone-negative-source", choices=("reject", "legacy"), default="reject",
                        help="Reject negative source ozone, or preserve original signed samples; output must remain nonnegative")
    parser.add_argument("--ufs-surface-source", type=Path, nargs="+", help="Complete analyzed UFS NetCDF or explicitly mapped GRIB files; requires native-noah or ufs-ms")
    parser.add_argument("--ufs-surface-mapping", type=Path, help="JSON mapping from analyzed physical quantities to source variables")
    parser.add_argument("--ufs-surface-polar-extension", choices=("reject", "clamp"), default="reject",
                        help="Reject uncovered polar targets, or explicitly clamp to the nearest source latitude")
    parser.add_argument("--levels", type=int, default=55)
    parser.add_argument("--top", type=float, default=30000.0)
    args = parser.parse_args()
    if args.profile == "native-noah" and (not args.surface_source or not args.near_surface_source or args.ozone_source):
        parser.error("--profile native-noah requires --surface-source and --near-surface-source, without --ozone-source")
    if args.profile == "ufs-ms" and not all((args.surface_source, args.near_surface_source, args.ozone_source, args.ufs_surface_source)):
        parser.error("--profile ufs-ms requires --surface-source, --near-surface-source, --ozone-source, and --ufs-surface-source")
    if args.surface_mapping and not args.surface_source:
        parser.error("--surface-mapping requires --surface-source")
    if args.near_surface_mapping and not args.near_surface_source:
        parser.error("--near-surface-mapping requires --near-surface-source")
    if args.ozone_mapping and not args.ozone_source:
        parser.error("--ozone-mapping requires --ozone-source")
    if args.ozone_hybrid_convention and not args.ozone_source:
        parser.error("--ozone-hybrid-convention requires --ozone-source")
    if args.ozone_polar_extension != "reject" and not args.ozone_source:
        parser.error("--ozone-polar-extension requires --ozone-source")
    if args.ozone_negative_source != "reject" and not args.ozone_source:
        parser.error("--ozone-negative-source requires --ozone-source")
    if args.ufs_surface_source and args.profile not in ("native-noah", "ufs-ms"):
        parser.error("--ufs-surface-source requires --profile native-noah or ufs-ms")
    if args.ufs_surface_mapping and not args.ufs_surface_source:
        parser.error("--ufs-surface-mapping requires --ufs-surface-source")
    if args.inventory:
        if args.inventory.resolve() == args.output.resolve():
            parser.error("--inventory and --output must be different files")
        for path in (args.output, args.inventory):
            if path.exists() or path.is_symlink():
                parser.error(f"output already exists: {path}")
            if not path.parent.is_dir():
                parser.error(f"output directory does not exist: {path.parent}")
    from mpas_init.initialize import initialize_dycore
    from mpas_init.mesh import read_static, read_surface_categories, read_climatology, read_soil_temperature, read_native_noah_static, read_ufs_static
    from mpas_init.netcdf import read_atmosphere, read_surface, read_near_surface, read_ozone, read_ufs_surface
    from mpas_init.output import write_dycore
    from mpas_init.surface import initialize_surface
    from mpas_init.diagnostics import initialize_diagnostics
    from mpas_init.tracers import initialize_ufs_tracers
    from mpas_init.profiles import initialize_native_noah, initialize_ufs_static
    from mpas_init.ufs_surface import initialize_ufs_surface

    mapping = json.loads(args.mapping.read_text()) if args.mapping else None
    selection = {"valid_time": args.valid, "reference_time": args.cycle, "member": args.member}
    source = read_atmosphere(args.source, mapping, include_relative_humidity=args.near_surface_source is not None,
                             hybrid_convention=args.hybrid_convention, **selection)
    static = read_static(args.static)
    fields = initialize_dycore(source, static, n_levels=args.levels, model_top=args.top)
    surface, surface_declarations = None, None
    if args.surface_source:
        mapping = json.loads(args.surface_mapping.read_text()) if args.surface_mapping else None
        meteorology = read_surface(args.surface_source, mapping, **selection)
        if (meteorology.valid_time, meteorology.reference_time, meteorology.lead, meteorology.member) != (
                source.valid_time, source.reference_time, source.lead, source.member):
            raise ValueError("surface and atmospheric forecast identities differ")
        categories, landuse_name = read_surface_categories(args.static)
        climatology = read_climatology(args.static)
        surface_static = {**fields, "landmask": categories["landmask"], "ivgtyp": categories["lu_index"],
                          "isltyp": categories["soilcat_top"], "soiltemp": read_soil_temperature(args.static),
                          **{name: value.T if value.ndim == 2 else value for name, value in climatology.items()}}
        surface = initialize_surface(meteorology, surface_static, ice_category=24 if landuse_name == "USGS" else 15)
        surface_declarations = meteorology.declarations
    diagnostics = None
    if args.near_surface_source:
        mapping = json.loads(args.near_surface_mapping.read_text()) if args.near_surface_mapping else None
        diagnostics = initialize_diagnostics(source, read_near_surface(args.near_surface_source, mapping, **selection), fields)
    tracers = None
    ozone = None
    if args.ozone_source:
        mapping = json.loads(args.ozone_mapping.read_text()) if args.ozone_mapping else None
        ozone = read_ozone(args.ozone_source, mapping, hybrid_convention=args.ozone_hybrid_convention, **selection)
        tracers = initialize_ufs_tracers(ozone, source, fields,
                                         extension=args.ozone_extension, polar_extension=args.ozone_polar_extension,
                                         negative_source=args.ozone_negative_source)
    native, landuse_name = None, None
    if args.profile in ("native-noah", "ufs-ms"):
        profile_static, landuse_name = read_native_noah_static(args.static)
        native = initialize_native_noah(profile_static, fields, landuse_name)
    ufs_surface, ufs_surface_declarations = None, None
    if args.ufs_surface_source:
        mapping = json.loads(args.ufs_surface_mapping.read_text()) if args.ufs_surface_mapping else None
        analyzed = read_ufs_surface(args.ufs_surface_source, mapping, **selection)
        ufs_surface_declarations = analyzed.declarations
        ufs_surface = initialize_ufs_surface(analyzed, source, fields, native, surface,
                                             polar_extension=args.ufs_surface_polar_extension)
    ufs_static = initialize_ufs_static(read_ufs_static(args.static), fields) if args.profile == "ufs-ms" else None
    ugwp_polar_extension = None
    if ufs_static is not None:
        from netCDF4 import Dataset

        names = ("mpas_init_ugwp_polar_extension", "mpas_init_ugwp_polar_directional_convention")
        with Dataset(args.static) as ds:
            if any(name in ds.ncattrs() for name in names):
                ugwp_polar_extension = tuple(getattr(ds, name, "") for name in names)
    report = None
    if args.inventory:
        from mpas_init.output import inventory_record, write_inventory_report

        report = inventory_record({**fields, **(native or {})}, source, args.output)
    write_dycore(args.output, fields, static, source.valid_time, surface=surface, diagnostics=diagnostics,
                  tracers=tracers, ozone_extension=args.ozone_extension, ozone_polar_extension=args.ozone_polar_extension,
                  ozone_negative_source=args.ozone_negative_source,
                  ozone_vertical_convention=None if ozone is None else ozone.vertical_convention,
                  native=native, landuse_name=landuse_name,
                  ufs_surface=ufs_surface, ufs_surface_polar_extension=args.ufs_surface_polar_extension, ufs_static=ufs_static,
                  ufs_surface_declarations=ufs_surface_declarations, surface_declarations=surface_declarations,
                  ugwp_polar_extension=ugwp_polar_extension,
                  source_vertical_convention=source.vertical_convention,
                  source_forecast_identity=(source.reference_time, source.lead, source.member)
                  if source.member is not None or any(value is not None for value in selection.values()) else None)
    if report is not None:
        write_inventory_report(args.inventory, report)
    profile = ("complete native Noah cold-start file contract; coupled physics not yet qualified" if native is not None
               else "dycore only" if surface is None and diagnostics is None and tracers is None
               else "dycore and physics prefixes; not a full physics profile")
    if ufs_static is not None:
        profile = "complete UFS R4 cold-start file contract; producer conventions and coupled physics not yet qualified"
    elif ufs_surface is not None:
        profile += "; analyzed UFS surface prefix, not a complete UFS profile"
    print(f"Wrote {args.output}: {static.sizes['nCells']} cells, {args.levels} levels; {profile}")


if __name__ == "__main__":
    main()
