from dataclasses import replace

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_ozone
from mpas_init.tracers import initialize_ufs_tracers
from test_ozone_grids import generated_grid
from test_tracers import write_source


def full_pressure(source):
    y, x = torch.meshgrid(source.latitude, source.longitude, indexing="ij")
    ap, b = source.pressure * .35, source.pressure * (.65 / 105000.)
    ps = 105000. + 7000.*torch.sin(y*torch.pi/180.) + 4000.*torch.cos(x*torch.pi/180.)
    pressure = ap[:, None, None] + b[:, None, None]*ps
    return replace(source, pressure=pressure), (ap, b, ps)


def write_full_pressure(path, source, coefficients, representation, *, dry=False):
    ap, b, ps = coefficients
    write_source(path, replace(source, pressure=source.pressure[:, 0, 0]), dry=dry)
    with Dataset(path, "a") as ds:
        level = ds["pressure"]
        level.units, level[:] = "1", np.arange(source.pressure.shape[0])
        if representation == "3d":
            level.delncattr("standard_name")
            var = ds.createVariable("full_pressure", "f8", ds["o3"].dimensions)
            var.units, var.standard_name, var[0] = "hPa", "air_pressure", source.pressure.numpy()/100.
            var.coordinates = "latitude longitude"
            return
        level.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
        aa, bb = (ds.createVariable(name, "f8", ("pressure",)) for name in ("aa", "bb"))
        bb.units, bb[:] = "1", b.numpy()
        surface = ds.createVariable("ps", "f8", ("time", "latitude", "longitude"))
        surface.units, surface.standard_name, surface[0] = "hPa", "surface_air_pressure", ps.numpy()/100.
        surface.coordinates = "latitude longitude"
        if representation == "a":
            aa.units, aa[:] = "1", ap.numpy()/100000.
            p0 = ds.createVariable("p0", "f8")
            p0.units, p0[:] = "hPa", 1000.
            level.formula_terms = "a: aa b: bb ps: ps p0: p0"
        else:
            aa.units, aa[:] = "hPa", ap.numpy()/100.
            level.formula_terms = "ap: aa b: bb ps: ps"


@pytest.mark.parametrize("defect", ["shape","negative","nonfinite","duplicate","crossed","mixed_scan","coverage"])
def test_invalid_full_pressure_tensor_rejects(defect):
    source, atmosphere, fields = generated_grid()
    source, _ = full_pressure(source)
    p = source.pressure.clone()
    if defect == "shape": p=p[:,0]
    elif defect == "negative": p[0,0,0]=-1.
    elif defect == "nonfinite": p[0,0,0]=torch.nan
    elif defect == "duplicate": p[1,0,0]=p[0,0,0]
    elif defect == "crossed": p[1,0,0]=p[2,0,0]+1.
    elif defect == "mixed_scan": p[:,0,0]=p[:,0,0].flip(0)
    else: p=p*100.
    with pytest.raises((ValueError,RuntimeError)):
        initialize_ufs_tracers(replace(source,pressure=p),atmosphere,fields)


@pytest.mark.parametrize("defect", ["p0","term","units","interfaces","pressure_grid","ps_grid","mean","duplicate","crossed"])
def test_invalid_full_pressure_codec_rejects(tmp_path, defect):
    source, _, _ = generated_grid()
    source, coefficients = full_pressure(source)
    path=tmp_path/"wrong.nc"
    write_full_pressure(path,source,coefficients,"3d" if defect in ("pressure_grid","mean","duplicate","crossed") else "a")
    with Dataset(path,"a") as ds:
        if defect=="p0": ds["pressure"].formula_terms="a: aa b: bb ps: ps"
        elif defect=="term": ds["pressure"].formula_terms+=" b: bb"
        elif defect=="units": ds["ps"].units="unknown"
        elif defect=="interfaces":
            ds.createDimension("interface",source.pressure.shape[0]+1)
            ds.createVariable("interface_a","f8",("interface",)).units="1"
            ds["pressure"].formula_terms="a: interface_a b: bb ps: ps p0: p0"
        elif defect=="mean": ds["full_pressure"].cell_methods="time: mean"
        elif defect=="duplicate": ds["full_pressure"][0,1,0,0]=ds["full_pressure"][0,0,0,0]
        elif defect=="crossed": ds["full_pressure"][0,1,0,0]=ds["full_pressure"][0,2,0,0]+1.
        else:
            var=ds.createVariable("other_latitude","f8",("latitude",))
            var[:],var.standard_name,var.units=ds["latitude"][:]+.1,"latitude","degrees_north"
            ds["full_pressure" if defect=="pressure_grid" else "ps"].coordinates="other_latitude longitude"
    with pytest.raises((ValueError,RuntimeError)):
        read_ozone(path)
