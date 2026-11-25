# mpas-init

mpas-init builds global MPAS-Atmosphere initial conditions from complete meteorological
analyses or forecasts, such as the NOAA GFS analysis. The numerical work is written in PyTorch:
terrain-following coordinates, horizontal and vertical interpolation, the MPAS discrete moist
hydrostatic adjustment, surface and soil initialization, ozone, and the static fields
(terrain, land use, soil, climatology and subgrid orography for gravity-wave drag). The
calculations are PyTorch tensor operations that keep their derivatives, so an initial condition can
be differentiated with respect to its inputs.

You do not need WPS, MPI, ESMF or a Fortran build to make an initial condition. The command line
reads GRIB or NetCDF files and writes a new MPAS NetCDF initial-condition file.

It follows the MPAS `init_atmosphere` real-data initialization (case 7) and the WPS `ungrib`
conventions for GFS input.

## Install

mpas-init needs Python 3.11 or newer; it is tested with Python 3.11.

```sh
pip install .            # NetCDF input
pip install '.[grib]'    # adds GRIB1/GRIB2 input through ecCodes
pip install '.[test]'    # adds pytest
```

The `grib` extra installs the `eccodes` Python package, which brings its own ecCodes library on
common platforms. Check it with `python -m eccodes selfcheck`. mpas-init needs torch 2.8 or newer,
below 2.16; the tests pass with torch 2.8.0 and 2.15.0.

Set the numerical thread limits before Python starts when you need reproducible output:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Quick start: GFS analyses

[`examples/gfs/README.md`](examples/gfs/README.md) builds a complete `ufs-ms` initial condition
from two public GFS analysis files, `gfs.tHHz.pgrb2.0p25.f000` (GRIB2) and `gfs.tHHz.sfcanl.nc`.
The mapping files in `examples/gfs/` declare every GFS convention that the readers need, such as
the WPS treatment of relative humidity and snow.

## Prepared statics

An initial condition needs a static file for the target mesh: geometry, terrain, land and soil
categories, climatology, soil temperature and gravity-wave-drag statistics. Statics do not depend on
the date, so you prepare them once per mesh.

Prepared `ufs-ms` statics for the quasi-uniform meshes x1.163842 and x1.1024002 are attached to the
GitHub release; see the release notes for SHA256. The `ufs-ms` statics also contain every
native-Noah field.

## Static preparation

`mpas-init prepare` makes a static file from a global mesh and a local copy of the MPAS geography
datasets:

```sh
mpas-init prepare --mesh x1.10242 --mesh-cache mesh-cache --download-mesh \
  --data-root GEOGRAPHY --profile ufs-ms --output x1.10242.ufs-ms.static.nc
```

`--mesh` takes a local global grid file, or one of the names `x1.2562`, `x1.4002` and `x1.10242`.
A name downloads the mesh from the [NCAR mesh archive](https://mpas-dev.github.io/atmosphere/atmosphere_meshes.html)
only when `--download-mesh` is given; later runs reuse `--mesh-cache` offline. For a mesh file
without `nominalMinDc`, give `--nominal-min-dc METRES`.

| Profile | Contents |
| --- | --- |
| `dycore` | Geometry and terrain |
| `native-noah` | Geometry, terrain, land use and soil categories, six climatology fields, annual soil temperature and fourteen classic gravity-wave-drag fields |
| `ufs-ms` | The native-Noah fields plus twenty UGWP orography fields |

The geography root is the standard MPAS/WPS static geography directory. The `dycore` profile needs
`topo_gmted2010_30s`. The `native-noah` profile also needs `modis_landuse_20class_30s`,
`soiltype_top_30s`, `maxsnowalb_modis`, `greenfrac_fpar_modis`, `albedo_modis`, `lai_modis_30s`,
`soiltemp_1deg` and `landuse_30s`. The `ufs-ms` profile also needs `topo_ugwp_30s` and
`topo_ugwp_2.5m`. The MODIS land-use index must identify `MODIFIED_IGBP_MODIS_NOAH`; classic
gravity-wave drag uses the 24-category USGS `landuse_30s`.

When you prepare several meshes, convert the geography once into an indexed collection and pass that
as `--data-root`:

```sh
mpas-init prepare-geography --data-root GEOGRAPHY --output geography-index --profile ufs-ms
```

The conversion keeps every raw value and adds block summaries that make preparation faster. Keep the
converted files unchanged. `--cache-dir DIRECTORY` reuses verified geometry and completed statics
between runs.

Static preparation needs the small Atlas candidate compiler in [`atlas_geometry/`](atlas_geometry/README.md)
on `PATH`, or at `MPAS_INIT_ATLAS_EXECUTABLE`. It finds candidate mesh cells for blocks of
geography pixels; PyTorch chooses the owners and computes every field. Building an initial condition
from a prepared static file does not need Atlas.

## Making an initial condition

```sh
mpas-init --profile native-noah \
  --source atmosphere.nc --surface-source surface.nc --near-surface-source surface.nc \
  --static x1.10242.ufs-ms.static.nc --output init.nc
```

`python -m mpas_init` is equivalent to `mpas-init`. The output file must not exist.

Each input group can span several files and use its own grid:

| Option | Content |
| --- | --- |
| `--source` | Atmosphere: temperature, winds, specific humidity, and geopotential height or geopotential, on pressure levels, hybrid levels or explicit 3-D pressure |
| `--surface-source` | Land mask, terrain, skin temperature, snow, sea ice, soil temperature and soil water |
| `--near-surface-source` | 2 m temperature and relative humidity, 10 m winds |
| `--ozone-source` | Ozone mixing ratio |
| `--ufs-surface-source` | Analyzed surface fields used by `ufs-ms` (albedos, roughness, deep soil temperature and others) |

Source grids can be regular latitude-longitude, regular or reduced Gaussian, or CF gathered reduced
grids. The readers identify fields by CF standard names or ecCodes CF names. For other names,
pass a JSON mapping with `--mapping`, `--surface-mapping`, `--near-surface-mapping`,
`--ozone-mapping` or `--ufs-surface-mapping`. A mapping also declares conventions that the file
does not state, such as the saturation reference of relative humidity. The readers reject
ambiguous or incomplete input; they do not guess. Use `--valid`, `--cycle` and `--member` to select
one forecast from files that hold several. `--levels` (default 55) and `--top` (default 30000 m)
set the target vertical grid.

## Output profiles

| `--profile` | Output |
| --- | --- |
| `prefix` (default) | Dycore state, metrics and geometry, plus the optional groups given |
| `native-noah` | The complete native MPAS four-layer Noah cold-start file (149 variables); needs surface and near-surface input |
| `ufs-ms` | The complete cold-start file for the ufs-ms-torch forecast model, MPAS dynamics with GFS physics (191 variables); needs surface, near-surface, ozone and UFS surface input |

All profiles are cold starts. They contain no analyzed cloud condensate. `--inventory report.json`
also writes a JSON summary of dry-air and water totals for the new state; it never changes the
initial condition.

## Python API

The command line is a thin layer over the readers, tensor functions and writer. The main entry
points are:

| Function | Module |
| --- | --- |
| `read_atmosphere`, `read_surface`, `read_near_surface`, `read_ozone`, `read_ufs_surface` | `mpas_init.netcdf` (GRIB and NetCDF) |
| `read_static`, `read_native_noah_static`, `read_ufs_static` | `mpas_init.mesh` |
| `initialize_dycore`, `prepare_dycore_geometry` | `mpas_init.initialize` |
| `initialize_surface` | `mpas_init.surface` |
| `initialize_diagnostics` | `mpas_init.diagnostics` |
| `initialize_ufs_tracers` | `mpas_init.tracers` |
| `initialize_native_noah`, `initialize_ufs_static` | `mpas_init.profiles` |
| `initialize_ufs_surface` | `mpas_init.ufs_surface` |
| `write_dycore` | `mpas_init.output` |
| `prepare_static`, `prepare_geography` | `mpas_init.prepare` |
| `atmosphere_inventory` | `mpas_init.inventory` |

```python
from mpas_init.netcdf import read_atmosphere
from mpas_init.mesh import read_static
from mpas_init.initialize import initialize_dycore
from mpas_init.output import write_dycore

source = read_atmosphere("atmosphere.nc")
static = read_static("x1.10242.ufs-ms.static.nc")
fields = initialize_dycore(source, static)        # dict of torch tensors
write_dycore("dycore.nc", fields, static, source.valid_time)
```

`mpas_init/__main__.py` composes the complete profiles the same way. For several dates on one mesh, call `prepare_dycore_geometry(static)` once and pass it as `geometry=` to
`initialize_dycore`.

## Limits

- Global meshes only. Regional meshes and lateral boundary conditions are not supported.
- Cold starts only; there is no warm-restart translation or analyzed cloud condensate.
- The input must be complete. Arbitrary cubed-sphere, curvilinear or spectral source grids are not
  read; convert them to one of the supported grids first.
- Interpolation is not mass-conserving. The inventory report shows the totals of the result but does
  not correct them.

## Tests

```sh
pip install '.[test,grib]'
pytest -q
```

The tests build their own small synthetic inputs and need no network or external data. GRIB tests
skip when ecCodes is unavailable.
