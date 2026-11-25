"""NetCDF boundary for raw and prepared, closed-global MPAS meshes."""

from dataclasses import dataclass
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import torch
from torch import Tensor

from mpas_init.gwd import CLASSIC_FIELDS
from mpas_init.ugwp import UGWP_FIELDS
from mpas_init.profiles import NATIVE_GWD_FIELDS
from mpas_init.vertical_grid import GlobalConnectivity


MESH_FIELDS = (
    "latCell", "lonCell", "xCell", "yCell", "zCell", "indexToCellID",
    "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge", "indexToEdgeID",
    "latVertex", "lonVertex", "xVertex", "yVertex", "zVertex", "indexToVertexID",
    "cellsOnEdge", "nEdgesOnCell", "nEdgesOnEdge", "edgesOnCell", "edgesOnEdge",
    "weightsOnEdge", "dvEdge", "dcEdge", "angleEdge", "areaCell", "areaTriangle",
    "cellsOnCell", "verticesOnCell", "verticesOnEdge", "edgesOnVertex", "cellsOnVertex",
    "kiteAreasOnVertex", "meshDensity", "nominalMinDc", "bdyMaskCell", "bdyMaskEdge",
    "bdyMaskVertex", "fEdge", "fVertex", "ter", "deriv_two", "defc_a", "defc_b",
    "cell_gradient_coef_x", "cell_gradient_coef_y",
)
GENERATED_FIELDS = frozenset(("bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex", "fEdge", "fVertex",
                              "ter", "deriv_two", "defc_a", "defc_b",
                              "cell_gradient_coef_x", "cell_gradient_coef_y"))
RAW_MESH_FIELDS = tuple(name for name in MESH_FIELDS if name not in GENERATED_FIELDS and name != "nominalMinDc")
SURFACE_DIMENSIONS = {"lu_index": ("nCells",), "soilcat_top": ("nCells",), "landmask": ("nCells",),
                      "landusef": ("nCells", "nlcat"), "soilf": ("nCells", "nscat")}
SURFACE_NAMES = {"lu_index": "ivgtyp", "soilcat_top": "isltyp", "landmask": "landmask",
                 "landusef": "landusef", "soilf": "soilf"}
CLIMATOLOGY_DIMENSIONS = {name: ("nCells", "nMonths") if name in ("greenfrac", "albedo12m", "lai12m")
                          else ("nCells",) for name in ("snoalb", "greenfrac", "shdmin", "shdmax", "albedo12m", "lai12m")}
CLIMATOLOGY_UNITS = {"snoalb": "unitless", "greenfrac": "percent", "shdmin": "percent",
                     "shdmax": "percent", "albedo12m": "percent", "lai12m": "unitless"}
CLASSIC_GWD_UNITS = {name: "m" if name in ("var2d", "gwd_elvmax") else "rad" if name == "gwd_theta"
                     else "unitless" for name in CLASSIC_FIELDS}
UGWP_UNITS = {name: "m" if name.startswith("var2d") else "unitless" for name in UGWP_FIELDS}


@dataclass(frozen=True)
class GlobalMesh:
    fields: dict[str, Tensor]
    dimensions: dict[str, tuple[str, ...]]
    sizes: dict[str, int]
    sphere_radius: float
    connectivity: GlobalConnectivity


@dataclass(frozen=True)
class GlobalStatic(GlobalMesh):
    """Prepared geometry and terrain, ready for atmospheric interpolation."""


def _read_mesh(path: str | Path, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> GlobalMesh:
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        on_a_sphere = getattr(ds, "on_a_sphere", "NO")
        if not isinstance(on_a_sphere, str) or on_a_sphere.strip() != "YES":
            raise ValueError("only closed-global spherical MPAS meshes are supported")
        missing = set(required).difference(ds.variables)
        if missing:
            raise ValueError(f"mesh file lacks required fields: {sorted(missing)}")
        if "rho" in ds.variables:
            raise ValueError("expected a static mesh, not a previous atmospheric initial condition")
        fields, dimensions = {}, {}
        for name in required + tuple(name for name in optional if name in ds.variables):
            variable = ds[name]
            values = np.asarray(variable[:])
            if values.dtype.kind == "f":
                if not np.isfinite(values).all() or np.any(values == variable.get_fill_value()):
                    raise ValueError(f"incomplete static field: {name}")
                fields[name] = torch.tensor(values, dtype=torch.float64)
            else:
                fields[name] = torch.tensor(values.astype(np.int64))
            dimensions[name] = variable.dimensions
        sizes = {name: len(dim) for name, dim in ds.dimensions.items()}
        radius = float(ds.sphere_radius)
    connectivity = _global_connectivity(fields, sizes, radius)
    return GlobalMesh(fields, dimensions, sizes, radius, connectivity)


def _global_connectivity(fields: dict[str, Tensor], sizes: dict[str, int], radius: float) -> GlobalConnectivity:
    """Validate decoded input at the I/O boundary, before entering tensor numerics."""
    nc, ne, nv = (sizes[name] for name in ("nCells", "nEdges", "nVertices"))
    if not np.isfinite(radius) or radius <= 0 or nc - ne + nv != 2:
        raise ValueError("invalid closed-global spherical mesh geometry")
    if sizes["vertexDegree"] != 3:
        raise ValueError("MPAS atmosphere requires three cells at each vertex")
    if "R3" in sizes and sizes["R3"] != 3:
        raise ValueError("MPAS Cartesian vectors require R3=3")
    for name, shape in (("nEdgesOnCell", (nc,)), ("cellsOnEdge", (ne, 2)),
                        ("cellsOnCell", (nc, sizes["maxEdges"])), ("edgesOnCell", (nc, sizes["maxEdges"]))):
        if fields[name].shape != shape:
            raise ValueError(f"invalid global connectivity shape: {name}")
    for name in ("bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex"):
        if name in fields and torch.any(fields[name] != 0):
            raise ValueError("regional meshes and lateral boundaries are out of scope")
    counts, edge_cells = fields["nEdgesOnCell"], fields["cellsOnEdge"]
    slots = torch.arange(sizes["maxEdges"])[None, :] < counts[:, None]
    if torch.any(counts < 3) or torch.any(counts > sizes["maxEdges"]):
        raise ValueError("invalid cell edge counts")
    if torch.any(edge_cells < 1) or torch.any(edge_cells > nc) or torch.any(edge_cells[:, 0] == edge_cells[:, 1]):
        raise ValueError("global edges must join two distinct valid cells")
    for name, limit in (("cellsOnCell", nc), ("edgesOnCell", ne)):
        values = fields[name]
        if torch.any(values[slots] < 1) or torch.any(values < 0) or torch.any(values > limit):
            raise ValueError(f"invalid active global connectivity: {name}")
    for name in ("dcEdge", "dvEdge", "areaCell", "areaTriangle", "meshDensity"):
        if torch.any(fields[name] <= 0):
            raise ValueError(f"nonpositive mesh geometry: {name}")
    if "nominalMinDc" in fields and (fields["nominalMinDc"].shape != () or fields["nominalMinDc"] <= 0):
        raise ValueError("nominalMinDc must be a positive scalar in mesh coordinate units")
    for name, limit, shape in (("verticesOnCell", nv, (nc, sizes["maxEdges"])),
                               ("verticesOnEdge", nv, (ne, 2)),
                               ("cellsOnVertex", nc, (nv, sizes["vertexDegree"])),
                               ("edgesOnVertex", ne, (nv, sizes["vertexDegree"]))):
        value = fields[name]
        if value.shape != shape:
            raise ValueError(f"invalid global connectivity shape: {name}")
        selected = value[slots] if name == "verticesOnCell" else value
        if torch.any(selected < 1) or torch.any(value < 0) or torch.any(value > limit):
            raise ValueError(f"invalid global connectivity: {name}")
    # An active edge must contain this cell and the declared neighbor, not just valid IDs.
    active_cells = torch.arange(1, nc + 1)[:, None].expand_as(slots)[slots]
    active_edges = edge_cells[fields["edgesOnCell"][slots] - 1]
    neighbors = fields["cellsOnCell"][slots]
    agrees = ((active_edges[:, 0] == active_cells) & (active_edges[:, 1] == neighbors)
              | (active_edges[:, 1] == active_cells) & (active_edges[:, 0] == neighbors))
    if not agrees.all() or not torch.equal(torch.bincount(fields["edgesOnCell"][slots] - 1, minlength=ne),
                                           torch.full((ne,), 2, dtype=torch.int64)):
        raise ValueError("global cell/edge connectivity is not reciprocal")
    return GlobalConnectivity((fields["cellsOnCell"] - 1).clamp_min(0),
                                      (fields["edgesOnCell"] - 1).clamp_min(0), counts,
                                      fields["dcEdge"], fields["dvEdge"], edge_cells - 1)


def read_raw_mesh(path: str | Path) -> GlobalMesh:
    """Import raw global geometry without requiring terrain or derived coefficients."""
    return _read_mesh(path, RAW_MESH_FIELDS, ("nominalMinDc",))


def read_static(path: str | Path) -> GlobalStatic:
    """Read complete geometry and terrain. Never fill absent static fields with zeros."""
    mesh = _read_mesh(path, MESH_FIELDS)
    return GlobalStatic(mesh.fields, mesh.dimensions, mesh.sizes, mesh.sphere_radius, mesh.connectivity)


def _validate_surface(surface: dict[str, Tensor], n_cells: int, landuse_name: str) -> tuple[int, int]:
    if set(surface) != set(SURFACE_NAMES):
        raise ValueError("incomplete surface-category field inventory")
    if landuse_name not in ("USGS", "MODIFIED_IGBP_MODIS_NOAH"):
        raise ValueError("unsupported native land-use classification")
    n_land, n_soil = (24 if landuse_name == "USGS" else 20), 16
    for name, value in surface.items():
        shape = (n_land if name == "landusef" else n_soil, n_cells) if name.endswith("f") else (n_cells,)
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid surface-category field: {name}")
        if name.endswith("f"):
            if value.dtype not in (torch.float32, torch.float64) or torch.any(value < 0) or torch.any(value > 1):
                raise ValueError(f"invalid category fraction: {name}")
            if torch.any((value.sum(0) - 1).abs() > 1e-15 + 16 * torch.finfo(value.dtype).eps):
                raise ValueError(f"unnormalized category fractions: {name}")
        elif value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"category IDs and mask must be integer: {name}")
    land, soil, mask = (surface[name] for name in ("lu_index", "soilcat_top", "landmask"))
    water, ice = (16, 24) if landuse_name == "USGS" else (17, 15)
    if torch.any((land < 1) | (land > n_land)) or torch.any((soil < 1) | (soil > n_soil)):
        raise ValueError("surface category ID lies outside its classification")
    if (not torch.equal(mask, (land != water).to(mask.dtype))
            or not torch.equal(land == water, soil == 14) or torch.any(soil[land == ice] != 16)):
        raise ValueError("inconsistent native surface-category water/ice mask")
    return n_land, n_soil


def read_surface_categories(path: str | Path) -> tuple[dict[str, Tensor], str]:
    """Read native category tensor order and fraction precision, independently of dycore geometry."""
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        if not (set(SURFACE_NAMES.values()) | {"mminlu"}).issubset(ds.variables):
            raise ValueError("incomplete surface-category file inventory")
        if ds["mminlu"].dimensions != ("StrLen",) or len(ds.dimensions["StrLen"]) != 64:
            raise ValueError("native land-use classification requires a 64-character mminlu")
        landuse_name = np.asarray(ds["mminlu"][:]).tobytes().decode("ascii").rstrip(" \x00")
        category_ids = {"isice_lu", "iswater_lu"} & set(ds.variables)
        if category_ids:
            if category_ids != {"isice_lu", "iswater_lu"}:
                raise ValueError("incomplete native land-use category IDs")
            ice, water = (24, 16) if landuse_name == "USGS" else (15, 17)
            for name, expected in (("isice_lu", ice), ("iswater_lu", water)):
                variable = ds[name]
                if variable.dimensions or variable.dtype.kind not in ("i", "u") or variable.getValue() != expected:
                    raise ValueError("native land-use category IDs disagree with the classification")
        surface = {}
        for name, native_name in SURFACE_NAMES.items():
            variable = ds[native_name]
            if variable.dimensions != SURFACE_DIMENSIONS[name]:
                raise ValueError(f"invalid native surface-category dimensions: {native_name}")
            values = np.asarray(variable[:])
            if np.any(values == variable.get_fill_value()):
                raise ValueError(f"incomplete surface-category field: {native_name}")
            if name.endswith("f"):
                if values.dtype.kind != "f":
                    raise ValueError(f"category fraction must be floating point: {native_name}")
                surface[name] = torch.from_numpy(values.T.astype(values.dtype.newbyteorder("="), copy=True))
            else:
                if values.dtype.kind not in ("i", "u"):
                    raise ValueError(f"category IDs and mask must be integer: {native_name}")
                surface[name] = torch.tensor(values.astype(np.int64))
        _validate_surface(surface, len(ds.dimensions["nCells"]), landuse_name)
    return surface, landuse_name


def _validate_climatology(fields: dict[str, Tensor], surface: dict[str, Tensor], landuse_name: str) -> None:
    if set(fields) != set(CLIMATOLOGY_DIMENSIONS):
        raise ValueError("incomplete climatology field inventory")
    nc = surface["landmask"].shape[0]
    for name, value in fields.items():
        shape = (12, nc) if len(CLIMATOLOGY_DIMENSIONS[name]) == 2 else (nc,)
        if value.shape != shape or value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all():
            raise ValueError(f"invalid climatology field: {name}")
        upper = 1.0 if name == "snoalb" else float("inf") if name == "lai12m" else 100.0
        if torch.any(value < 0) or torch.any(value > upper):
            raise ValueError(f"climatology value outside native units: {name}")
        water = (surface["landmask"] == 0).to(value.device)
        if name == "albedo12m":
            ice = (surface["lu_index"] == (24 if landuse_name == "USGS" else 15)).to(value.device)
            if torch.any(value[:, water] != 8.0) or torch.any(value[:, ice] != 70.0):
                raise ValueError("climatology albedo violates native water/ice fills")
        elif torch.any(value[..., water] != 0):
            raise ValueError(f"climatology water value must be zero: {name}")
    if (not torch.equal(fields["shdmin"], fields["greenfrac"].amin(0))
            or not torch.equal(fields["shdmax"], fields["greenfrac"].amax(0))):
        raise ValueError("climatology vegetation extrema disagree with the monthly field")


def read_climatology(path: str | Path) -> dict[str, Tensor]:
    """Read six native climatology fields in tensor order, with their encoded precision."""
    surface, landuse_name = read_surface_categories(path)
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        if not set(CLIMATOLOGY_DIMENSIONS).issubset(ds.variables):
            raise ValueError("incomplete climatology file inventory")
        if "nMonths" not in ds.dimensions or len(ds.dimensions["nMonths"]) != 12:
            raise ValueError("native climatology requires twelve months")
        result = {}
        for name, dimensions in CLIMATOLOGY_DIMENSIONS.items():
            variable = ds[name]
            if variable.dimensions != dimensions or variable.dtype.kind != "f":
                raise ValueError(f"invalid native climatology dimensions or type: {name}")
            values = np.asarray(variable[:])
            if np.any(values == variable.get_fill_value()):
                raise ValueError(f"incomplete climatology field: {name}")
            result[name] = torch.from_numpy(values.T.astype(values.dtype.newbyteorder("="), copy=True))
    _validate_climatology(result, surface, landuse_name)
    return result


def _validate_soil_temperature(value: Tensor, surface: dict[str, Tensor]) -> None:
    if (value.shape != surface["landmask"].shape or value.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(value).all()):
        raise ValueError("invalid annual soil-temperature field")
    water = (surface["landmask"] == 0).to(value.device)
    if torch.any(value[water] != 0) or torch.any(value[~water] <= 0):
        raise ValueError("annual soil temperature requires water zero and positive land values")


def read_soil_temperature(path: str | Path) -> Tensor:
    """Read the native annual soil-temperature field and retain encoded precision."""
    surface, _ = read_surface_categories(path)
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        if "soiltemp" not in ds.variables:
            raise ValueError("missing annual soil-temperature field")
        variable = ds["soiltemp"]
        if variable.dimensions != ("nCells",) or variable.dtype.kind != "f":
            raise ValueError("invalid native soil-temperature dimensions or type")
        values = np.asarray(variable[:])
        if np.any(values == variable.get_fill_value()):
            raise ValueError("incomplete annual soil-temperature field")
        result = torch.from_numpy(values.astype(values.dtype.newbyteorder("="), copy=True))
    _validate_soil_temperature(result, surface)
    return result


def _validate_classic_gwd(fields: dict[str, Tensor], n_cells: int, *, required: tuple[str, ...] = CLASSIC_FIELDS) -> None:
    if set(fields) != set(required):
        raise ValueError("incomplete classic GWD field inventory")
    for name, value in fields.items():
        if value.shape != (n_cells,) or value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all():
            raise ValueError("invalid classic GWD field: " + name)
        if ((name in ("var2d", "con", "gwd_sigma") and torch.any(value < 0))
                or (name.startswith("oa") and torch.any(value.abs() > 1))
                or ((name.startswith("ol") or name == "gwd_gamma") and torch.any((value < 0) | (value > 1)))
                or (name == "gwd_theta" and torch.any(value.abs() > torch.pi * 0.5))):
            raise ValueError("classic GWD field outside native range: " + name)


def read_classic_gwd(path: str | Path, *, native_noah: bool = False) -> dict[str, Tensor]:
    """Read every native classic field, retaining its encoded floating precision."""
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        required = NATIVE_GWD_FIELDS if native_noah else CLASSIC_FIELDS
        if not set(required).issubset(ds.variables):
            raise ValueError("incomplete classic GWD field inventory")
        result = {}
        for name in required:
            variable = ds[name]
            if variable.dimensions != ("nCells",) or variable.dtype.kind != "f":
                raise ValueError("invalid native classic GWD dimensions or type: " + name)
            values = np.asarray(variable[:])
            if np.any(values == variable.get_fill_value()):
                raise ValueError("incomplete classic GWD field: " + name)
            result[name] = torch.from_numpy(values.astype(values.dtype.newbyteorder("="), copy=True))
        _validate_classic_gwd(result, len(ds.dimensions["nCells"]), required=required)
    return result


def read_native_noah_static(path: str | Path) -> tuple[dict[str, Tensor], str]:
    """Read the prepared native-profile additions in file/cell-major tensor order."""
    categories, landuse_name = read_surface_categories(path)
    climatology = read_climatology(path)
    fields = {name: value.T if name.endswith("f") else value for name, value in categories.items()
              if name in ("landmask", "landusef", "soilf")}
    fields.update({name: climatology[name].T for name in ("greenfrac", "albedo12m", "lai12m")})
    fields["soiltemp"] = read_soil_temperature(path)
    fields.update(read_classic_gwd(path, native_noah=True))
    return fields, landuse_name


def _validate_ugwp(fields: dict[str, Tensor], n_cells: int) -> None:
    if set(fields) != set(UGWP_FIELDS):
        raise ValueError("incomplete UGWP field inventory")
    for name,value in fields.items():
        if value.shape != (n_cells,) or value.dtype not in (torch.float32,torch.float64) or not torch.isfinite(value).all():
            raise ValueError("invalid UGWP field: "+name)
        if ((name.startswith("var2d") and torch.any(value < 0))
                or (name.startswith("con") and torch.any((value < 0) | (value > 10)))
                or (name.startswith("oa") and torch.any(value.abs() > 1))
                or (name.startswith("ol") and torch.any((value < 0) | (value > 1)))):
            raise ValueError("UGWP field outside native range: "+name)


def read_ugwp(path: str | Path) -> dict[str, Tensor]:
    """Read all twenty native fields, retaining their encoded floating precision."""
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        if not set(UGWP_FIELDS).issubset(ds.variables):
            raise ValueError("incomplete UGWP field inventory")
        result = {}
        for name in UGWP_FIELDS:
            variable = ds[name]
            if variable.dimensions != ("nCells",) or variable.dtype.kind != "f":
                raise ValueError("invalid native UGWP dimensions or type: "+name)
            values = np.asarray(variable[:])
            if np.any(values == variable.get_fill_value()):
                raise ValueError("incomplete UGWP field: "+name)
            result[name] = torch.from_numpy(values.astype(values.dtype.newbyteorder("="),copy=True))
        _validate_ugwp(result,len(ds.dimensions["nCells"]))
    return result


def read_ufs_static(path: str | Path) -> dict[str, Tensor]:
    """Read all fourteen classic and twenty UGWP fields from one prepared mesh."""
    return {**read_classic_gwd(path), **read_ugwp(path)}


def write_static(path: str | Path, static: GlobalStatic, *, surface: dict[str, Tensor] | None = None,
                  landuse_name: str | None = None, climatology: dict[str, Tensor] | None = None,
                  soil_temperature: Tensor | None = None, classic_gwd: dict[str, Tensor] | None = None,
                  ugwp: dict[str, Tensor] | None = None) -> None:
    """Write dycore statics with complete optional land-data stages, without placeholders."""
    if set(static.fields) != set(MESH_FIELDS) or set(static.dimensions) != set(MESH_FIELDS):
        raise ValueError("incomplete dycore-static field inventory")
    for name, value in static.fields.items():
        if value.shape != tuple(static.sizes[dim] for dim in static.dimensions[name]) or not torch.isfinite(value).all():
            raise ValueError(f"invalid or incomplete static field: {name}")
    _global_connectivity(static.fields, static.sizes, static.sphere_radius)
    if surface is not None:
        n_land, n_soil = _validate_surface(surface, static.sizes["nCells"], landuse_name)
    elif landuse_name is not None:
        raise ValueError("landuse_name requires complete surface categories")
    if climatology is not None:
        if surface is None:
            raise ValueError("climatology requires complete surface categories")
        _validate_climatology(climatology, surface, landuse_name)
    if soil_temperature is not None:
        if surface is None:
            raise ValueError("soil temperature requires complete surface categories")
        _validate_soil_temperature(soil_temperature, surface)
    if classic_gwd is not None:
        _validate_classic_gwd(classic_gwd, static.sizes["nCells"])
    if ugwp is not None:
        _validate_ugwp(ugwp,static.sizes["nCells"])
    path = Path(path)
    dataset = Dataset(path, "w", clobber=False, format="NETCDF3_64BIT_OFFSET")
    try:
        with dataset as ds:
            # Native input streams require R3 even when this static subset has no vector variable.
            for name in sorted({dim for dims in static.dimensions.values() for dim in dims} | {"R3"}):
                ds.createDimension(name, 3 if name == "R3" else static.sizes[name])
            ds.setncatts({"on_a_sphere": "YES", "sphere_radius": static.sphere_radius, "is_periodic": "NO",
                          "model_name": "mpas", "core_name": "init_atmosphere",
                          "mpas_init_profile": "global-dycore-category-static" if surface is not None else "global-dycore-static",
                          "history": "mpas-init PyTorch global mesh and native terrain preparation"})
            for name, value in static.fields.items():
                variable = ds.createVariable(name, "f8" if value.is_floating_point() else "i4", static.dimensions[name])
                variable[:] = value.detach().cpu().numpy()
            if surface is not None:
                ds.createDimension("nlcat", n_land)
                ds.createDimension("nscat", n_soil)
                ds.createDimension("StrLen", 64)
                for name, value in surface.items():
                    dtype = ("f4" if value.dtype == torch.float32 else "f8") if name.endswith("f") else "i4"
                    variable = ds.createVariable(SURFACE_NAMES[name], dtype, SURFACE_DIMENSIONS[name])
                    array = value.detach().cpu().numpy()
                    variable[:] = array.T if name.endswith("f") else array
                chars = np.full(64, b" ", dtype="S1")
                encoded = landuse_name.encode("ascii")
                chars[:len(encoded)] = np.frombuffer(encoded, dtype="S1")
                ds.createVariable("mminlu", "S1", ("StrLen",))[:] = chars
                ice, water = (24, 16) if landuse_name == "USGS" else (15, 17)
                for name, value in (("isice_lu", ice), ("iswater_lu", water)):
                    variable = ds.createVariable(name, "i4", ())
                    variable.units = "unitless"
                    variable.assignValue(value)
            if climatology is not None:
                ds.mpas_init_profile = "global-dycore-category-climatology-static"
                ds.createDimension("nMonths", 12)
                for name, value in climatology.items():
                    variable = ds.createVariable(name, "f4" if value.dtype == torch.float32 else "f8",
                                                  CLIMATOLOGY_DIMENSIONS[name])
                    variable.units = CLIMATOLOGY_UNITS[name]
                    variable[:] = value.detach().cpu().numpy().T
            if soil_temperature is not None:
                ds.mpas_init_profile = ("global-dycore-category-climatology-soil-static" if climatology is not None
                                        else "global-dycore-category-soil-static")
                variable = ds.createVariable("soiltemp", "f4" if soil_temperature.dtype == torch.float32 else "f8", ("nCells",))
                variable.units = "K"
                variable[:] = soil_temperature.detach().cpu().numpy()
            if classic_gwd is not None:
                ds.mpas_init_profile = ds.mpas_init_profile.removesuffix("-static") + "-classic-gwd-static"
                for name, value in classic_gwd.items():
                    variable = ds.createVariable(name, "f4" if value.dtype == torch.float32 else "f8", ("nCells",))
                    variable.units = CLASSIC_GWD_UNITS[name]
                    variable[:] = value.detach().cpu().numpy()
            if ugwp is not None:
                ds.mpas_init_profile = ds.mpas_init_profile.removesuffix("-static") + "-ugwp-static"
                for name,value in ugwp.items():
                    variable = ds.createVariable(name,"f4" if value.dtype == torch.float32 else "f8",("nCells",))
                    variable.units = UGWP_UNITS[name]
                    variable[:] = value.detach().cpu().numpy()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
