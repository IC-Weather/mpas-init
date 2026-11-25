"""The native four-layer Noah cold-start file contract, not a warm restart."""

import torch
from torch import Tensor

from mpas_init.gwd import CLASSIC_FIELDS
from mpas_init.ugwp import UGWP_FIELDS


NATIVE_GWD_FIELDS = ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4")
NATIVE_STATIC_DIMENSIONS = {
    **{name: ("nCells",) for name in ("landmask", "soiltemp", *NATIVE_GWD_FIELDS)},
    **{name: ("nCells", "nMonths") for name in ("greenfrac", "albedo12m", "lai12m")},
    "landusef": ("nCells", "nlcat"), "soilf": ("nCells", "nscat"),
}
NATIVE_ZERO_DIMENSIONS = {
    **{name: ("Time", "nCells", "nVertLevels") for name in
       ("nc", "ng", "ni", "nifa", "nr", "ns", "nwfa", "qh", "volg", "refl10cm")},
    "canwat": ("Time", "nCells"), "h_oml_initial": ("Time", "nCells"),
    "dz": ("Time", "nCells", "nSoilLevels"),
}
NATIVE_DIMENSIONS = {**NATIVE_STATIC_DIMENSIONS, **NATIVE_ZERO_DIMENSIONS, "isice_lu": (), "iswater_lu": ()}
UFS_STATIC_FIELDS = (*CLASSIC_FIELDS, *UGWP_FIELDS)
NATIVE_UNITS = {
    **{name: "unitless" for name in NATIVE_DIMENSIONS},
    **{name: "-" for name in ("indexToCellID", "indexToEdgeID", "indexToVertexID", "cellsOnEdge", "nEdgesOnCell",
                              "nEdgesOnEdge", "edgesOnCell", "edgesOnEdge", "weightsOnEdge", "dvEdge", "cellsOnCell",
                              "verticesOnCell", "verticesOnEdge", "edgesOnVertex", "cellsOnVertex",
                              "bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex")},
    **{name: "rad" for name in ("latCell", "lonCell", "latEdge", "lonEdge", "latVertex", "lonVertex", "angleEdge")},
    **{name: "m" for name in ("xCell", "yCell", "zCell", "xEdge", "yEdge", "zEdge", "xVertex", "yVertex", "zVertex",
                              "dcEdge", "nominalMinDc", "ter", "dzs", "zs", "snowh")},
    **{name: "m^2" for name in ("areaCell", "areaTriangle", "kiteAreasOnVertex")},
    **{name: "m^-1" for name in ("cell_gradient_coef_x", "cell_gradient_coef_y")},
    **{name: "m s^{-1}" for name in ("u_init", "v_init", "u", "w", "u10", "v10", "uReconstructZonal", "uReconstructMeridional")},
    **{name: "K" for name in ("theta", "theta_base", "t_init", "tslb", "tmn", "skintemp", "sst", "t2m")},
    **{name: "kg m^{-3}" for name in ("rho", "rho_base")},
    **{name: "kg kg^{-1}" for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qv_init", "q2")},
    **{name: "kg m^{-2}" for name in ("snow", "precipw")},
    **{name: "m3 m^{-3}" for name in ("smois", "sh2o")},
    **{name: "percent" for name in ("relhum", "rh2", "vegfra", "shdmin", "shdmax")},
    **{name: "nb kg^{-1}" for name in ("nc", "ng", "ni", "nifa", "nr", "ns", "nwfa")},
    "qh": "kg kg^{-1}", "volg": "m^3 kg^{-1}", "refl10cm": "dBZ", "canwat": "kg m^{-2}",
    "soiltemp": "K", "var2d": "m", "dz": "m", "h_oml_initial": "m",
    "greenfrac": "percent", "albedo12m": "percent",
    "surface_pressure": "Pa", "zgrid": "m MSL",
}


def initialize_native_noah(static_fields: dict[str, Tensor], dycore: dict[str, Tensor],
                           landuse_name: str) -> dict[str, Tensor]:
    """Keep prepared statics and the original explicitly declared no-input defaults.

    ``dz`` is the unused legacy zero field, not soil thickness or depth.
    Actual four-layer coordinates remain ``zs`` and ``dzs`` in the surface state.
    """
    if set(static_fields) != set(NATIVE_STATIC_DIMENSIONS):
        raise ValueError("incomplete native Noah static inventory")
    if landuse_name not in ("USGS", "MODIFIED_IGBP_MODIS_NOAH"):
        raise ValueError("unsupported native land-use classification")
    rho = dycore["rho"]
    if rho.ndim != 2:
        raise ValueError("native Noah requires cell-major atmospheric state")
    sizes = {"nCells": rho.shape[0], "nMonths": 12, "nlcat": 24 if landuse_name == "USGS" else 20, "nscat": 16}
    for name, value in static_fields.items():
        if value.shape != tuple(sizes[dim] for dim in NATIVE_STATIC_DIMENSIONS[name]):
            raise ValueError("invalid native Noah static shape: " + name)
        torch._assert_async(torch.isfinite(value).all(), "nonfinite native Noah statics")
    for name in ("qc", "qr", "qi", "qs", "qg"):
        torch._assert_async((dycore[name] == 0).all(), "native Noah profile requires its declared no-analyzed-cloud cold start")
    fields = dict(static_fields)
    for name, dimensions in NATIVE_ZERO_DIMENSIONS.items():
        shape = (rho.shape[0], 4) if name == "dz" else rho.shape if "nVertLevels" in dimensions else (rho.shape[0],)
        fields[name] = rho.new_zeros(shape)
    ice, water = (24, 16) if landuse_name == "USGS" else (15, 17)
    fields["isice_lu"] = rho.new_tensor(ice, dtype=torch.int64)
    fields["iswater_lu"] = rho.new_tensor(water, dtype=torch.int64)
    return fields


def initialize_ufs_static(static_fields: dict[str, Tensor], dycore: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keep complete prepared GWD statics and the original terrain-maximum rule.

    This does not prepare missing UGWP geography or replace its global polar gate.
    The sampled maximum and final model terrain remain differentiable away from ties.
    """
    terrain = dycore["ter"]
    if set(static_fields) != set(UFS_STATIC_FIELDS) or terrain.ndim != 1:
        raise ValueError("incomplete UFS GWD static inventory or target terrain")
    torch._assert_async(torch.isfinite(terrain).all(), "nonfinite UFS model terrain")
    for name, value in static_fields.items():
        if value.shape != terrain.shape:
            raise ValueError("invalid UFS GWD static shape: " + name)
        torch._assert_async(torch.isfinite(value).all(), "nonfinite UFS GWD statics")
    return {**static_fields, "gwd_elvmax": torch.maximum(static_fields["gwd_elvmax"], terrain)}
