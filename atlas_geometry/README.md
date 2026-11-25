# Atlas Candidate Compiler

`mpas-init prepare` needs this small C++ program to prepare statics for a mesh that has no
prepared geometry yet. It builds a Cartesian Atlas tree over the mesh cell centres and returns
conservative candidate cells for blocks of geography pixels. It does not compute field values or
choose the final owner of a pixel; PyTorch does that. Building an initial condition from prepared
statics does not need it.

Build it against an installed Atlas prefix:

```sh
cmake -S atlas_geometry -B build/atlas-geometry -DCMAKE_PREFIX_PATH=/path/to/atlas-prefix
cmake --build build/atlas-geometry --parallel 1
build/atlas-geometry/mpas-atlas-candidates --version
```

Set `CMAKE_INSTALL_PREFIX` to the Atlas prefix before `cmake --install build/atlas-geometry`.
The installed executable searches that prefix for its libraries and checks the loaded Atlas version
and revision. `mpas-init prepare` finds the program on `PATH` as `mpas-atlas-candidates`, or at the
path given by the `MPAS_INIT_ATLAS_EXECUTABLE` environment variable.

The program was built and used with eckit 1.28.3 and Atlas 0.46.0
(`e0af59ef4d895ffc647a9db01b48c0a3e13d16f6`), without external MPI, OpenMP or Fortran. The
`eckit_mpi` library then contains serial stubs. Atlas needs GRID, FIELD, FUNCTIONSPACE and
INTERPOLATION enabled; transforms, numerics, tessellation, GPU features, NetCDF and eckit codecs
can stay disabled. ecbuild 3.7.2 needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` with CMake 4.

## Protocol

Protocol 1 uses little-endian IEEE 754 binary64 coordinates and fixed-width unsigned integers.
One process handles one mesh and any number of batches.

1. Send `MPASAT01`, a uint64 cell count and the cell-centre XYZ triples in native cell order.
2. Read `MPASOK01` after tree construction.
3. Send a uint64 block count and `(x, y, z, enclosing_radius)` quadruples.
4. Read uint64 block and candidate counts, uint64 offsets and uint32 native cell indices.
5. Repeat from step 3, or send a zero block count to end.

Candidate lists use lexicographic XYZ order, with ties in native order. Requests and responses
have explicit size limits. Errors go to stderr with a nonzero exit status; stdout carries only
binary protocol data.
