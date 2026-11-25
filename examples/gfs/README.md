# GFS Analysis Recipe

Build a complete `--profile ufs-ms` initial condition from two unmodified public NOAA GFS analysis files:

| File | Content | Groups it supplies |
| --- | --- | --- |
| `gfs.tHHz.pgrb2.0p25.f000` | GRIB2, 0.25° global, isobaric levels | atmosphere, surface/soil, 2 m/10 m, ozone |
| `gfs.tHHz.sfcanl.nc` | FV3 Gaussian-grid surface analysis, NetCDF | analyzed UFS surface (15 quantities) |

Both come from the public bucket `s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/HH/atmos/`.
The mapping files in this directory declare every producer convention explicitly.

## 1. Download

Choose the cycle. Each file is 0.3-0.5 GB. No account is needed:

```sh
D=20240521 HH=00
mkdir -p gfs.$D/$HH
for f in gfs.t${HH}z.pgrb2.0p25.f000 gfs.t${HH}z.sfcanl.nc; do
  curl -fSL -o gfs.$D/$HH/$f https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.$D/$HH/atmos/$f
done
```

`aws s3 cp --no-sign-request s3://noaa-gfs-bdp-pds/gfs.$D/$HH/atmos/$f gfs.$D/$HH/` is equivalent.
The cycle folder must contain both files. Older GFS versions wrote other surface formats; those reject.

## 2. Prepare Statics Once Per Mesh

Statics do not depend on the date. Prepare them once, as described in the [project README](../../README.md#static-preparation):

```sh
python -m mpas_init prepare \
  --mesh x1.10242 --mesh-cache mesh-cache --download-mesh \
  --data-root "$GEOGRAPHY" --profile ufs-ms --output x1.10242.ufs-ms.static.nc
```

`GEOGRAPHY` is a complete local MPAS geography root, including `topo_ugwp_30s` and `topo_ugwp_2.5m`.
Static preparation needs the [Atlas candidate compiler](../../atlas_geometry/README.md).
Prepared statics for x1.163842 and x1.1024002 are attached to the GitHub release.

## 3. Build The Initial Condition

Run from the project root, with the `grib` extra installed. Set thread limits before Python starts:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G=gfs.$D/$HH/gfs.t${HH}z.pgrb2.0p25.f000
S=gfs.$D/$HH/gfs.t${HH}z.sfcanl.nc
python -m mpas_init --profile ufs-ms \
  --source "$G" --mapping examples/gfs/atmosphere.json \
  --surface-source "$G" --surface-mapping examples/gfs/surface.json \
  --near-surface-source "$G" --near-surface-mapping examples/gfs/near-surface.json \
  --ozone-source "$G" --ozone-mapping examples/gfs/ozone.json \
  --ozone-extension constant --ozone-negative-source legacy \
  --ufs-surface-source "$S" --ufs-surface-mapping examples/gfs/ufs-surface.json \
  --ufs-surface-polar-extension clamp \
  --static x1.10242.ufs-ms.static.nc --output gfs.$D.${HH}Z.x1.10242.ufs-ms.init.nc
```

The output must not exist. The default target is 55 levels with a 30 km top.
The output records the analyzed-surface declarations in the global attribute `mpas_init_ufs_surface_declarations`,
and the snow convention in `mpas_init_surface_declarations`.

## Declared Conventions

Each choice copies the existing Fortran initialization chain for these files: WPS 4.6 `ungrib` with a specific-humidity
Vtable, `init_atmosphere` case 7, and the ufs-ms surface and ozone preparation. The reader infers none of them.

| Mapping | Declaration | Reason |
| --- | --- | --- |
| `atmosphere.json`, `near-surface.json` | `relative_humidity.saturation_reference=wps_ice_blend` | WPS converts GFS RH, including 2 m RH, from its ice blend to liquid-water RH |
| atmosphere (no key) | specific humidity from `q` | Moisture comes from specific humidity, as `config_use_spechumd=true`; RH feeds diagnostics only |
| `surface.json` | short names for terrain, skin T, SWE, soil; ocean snow fill `zero`; soil-water maximum `0.468` | WPS `ungrib` caps GFS soil moisture at 0.468 (`rrpr.F` 1287-1291) |
| `surface.json` | `surface_snow_amount.convention=wps_ncep_gfs` | WPS doubles GFS SWE (`rrpr.F` 878-890), then derives depth as 0.005 × that SWE (`rrpr.F` 896-903). `Vtable.GFS` line 35 gives SNOWH no GRIB code, so the supplied `sde` is not used |
| `ozone.json` | `o3mr` as `dry_air` mixing ratio | The chain writes GFS O3MR into `o3mr` unchanged, without a moist-to-dry conversion |
| `--ozone-extension constant` | endpoint ozone above/below the source levels | The chain clamps the target pressure to the source range |
| `--ozone-negative-source legacy` | keep signed source ozone; reject negative output | GFS O3MR has small negative samples near 150-200 hPa; the chain does not clip them |
| `ufs-surface.json` `latitude`, `longitude`, `time` | `grid_yt`, `grid_xt`, `time` | The file has no CF standard names; `degrees_N`/`degrees_E` are CF unit spellings |
| `forecast_reference_time=time` | zero-lead analysis | `sfcanl.nc` has no reference-time or lead variables |
| `time.calendar=standard` | decode the civil label as a standard date | The file labels its clock `JULIAN`; the chain compares the civil label without conversion. A calendar that changes the label rejects |
| `QUANTITY.units` | `1` for the four albedos, `f10m`, `ffhh`, `ffmm`; `kg m-2` canopy water; `m s-1` friction velocity; `m` ice thickness; `kg kg-1` 2 m humidity | The file labels these `%`, `XXX`, `N/A` or `kg/kg`. The chain reads the stored values unchanged in these units. Albedo values are fractions (≤ 0.85) despite the `%` label |
| `air_temperature_2m.height`, `specific_humidity_2m.height` | `2` (m) | `tmp2m` and `spfh2m` carry no height coordinate |
| `--ufs-surface-polar-extension clamp` | nearest Gaussian latitude beyond ±89.909° | The chain clamps polar queries the same way |

The snow convention applies after the declared missing-ocean fill. It is a named WPS convention, not a free scale; with it, a mapped or supplied snow depth rejects.
The `rrpr.F` line numbers refer to WPS v4.6.0 `ungrib/src/rrpr.F`; `Vtable.GFS` is `ungrib/Variable_Tables/Vtable.GFS` of the same release.
A unit declaration must name the canonical unit of its quantity; it never selects a conversion.
A label that already conforms needs no declaration. Undeclared nonconforming labels or calendars still reject.
Values equal to the file's non-CF `missing` sentinel reject.

## Limits

This recipe was checked end to end for 2024-05-21 00Z on x1.10242 only. Other dates, cycles and meshes use the same files and code path but are not separately checked.
It does not establish forecast skill. Data acquisition is the user's step.
