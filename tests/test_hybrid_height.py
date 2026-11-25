from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_atmosphere
from mpas_init.vertical import hybrid_pressure, ifs_hydrostatic_columns
from test_atmosphere import generated_source
from test_source_collection import copy_variables
from test_vertical_grid import assert_stage


def inputs(levels=33, rows=5, columns=8):
    eta = np.r_[0., np.geomspace(1.e-5, 1., levels)]
    ap, b = 40000. * eta * (1. - eta), eta ** 2
    latitude, longitude = np.linspace(-90., 90., rows), np.arange(columns) * 360. / columns
    lat, lon = np.meshgrid(np.deg2rad(latitude), np.deg2rad(longitude), indexing="ij")
    sp = 100000. + 1024. * np.cos(lat) ** 2 * np.cos(lon)
    fraction = np.linspace(0., 1., levels)[:, None, None]
    temperature = 220. + 60. * fraction + np.zeros_like(sp)[None]
    humidity = .012 * fraction ** 3 + np.zeros_like(sp)[None]
    phi = 9.80665 * (1500. * np.cos(lat) ** 2 + 100. * np.sin(lon))
    return tuple(torch.tensor(value, dtype=torch.float64) for value in (ap, b, sp, temperature, humidity, phi))


def test_isothermal_pressure_mass_mean_and_top_height():
    pi = torch.tensor([0., 100., 1000., 10000., 50000., 100000.], dtype=torch.float64)[:, None]
    temperature, humidity = torch.full((5, 1), 280., dtype=torch.float64), torch.full((5, 1), .007, dtype=torch.float64)
    phi = torch.tensor([1500. * 9.80665], dtype=torch.float64)
    pressure, height = ifs_hydrostatic_columns(pi, temperature, humidity, phi)
    expected = np.empty((5, 1))
    rt = 287.06 * 280. * (1. + .609133 * .007)
    p = pi[:, 0].numpy()
    expected[0] = 1500. + rt / 9.80665 * np.log(p[-1] / (p[1] / 2.))
    # Exact pressure-mass integral of an isothermal column, independent of the recurrence.
    expected[1:, 0] = 1500. + rt / 9.80665 * (
        1. - (p[2:] * np.log(p[2:] / p[-1]) - p[1:-1] * np.log(p[1:-1] / p[-1])) / np.diff(p[1:]))
    assert_stage(height, torch.tensor(expected), 64)
    assert torch.equal(pressure, (pi[:-1] + pi[1:]) / 2)


@pytest.mark.parametrize("defect", ["shape", "top", "order", "temperature", "humidity", "surface", "dtype"])
def test_invalid_columns_reject(defect):
    values = list(inputs(5))
    pi = hybrid_pressure(*values[:3], interfaces=True)
    if defect == "shape":
        pi = pi[:-1]
    elif defect == "top":
        pi[0] = .1
    elif defect == "order":
        pi = pi.flip(0)
    elif defect == "temperature":
        values[3][1, 0, 0] = 0.
    elif defect == "humidity":
        values[4][1, 0, 0] = 1.
    elif defect == "surface":
        values[5][0, 0] = torch.nan
    else:
        values[5] = values[5].float()
    with pytest.raises((ValueError, RuntimeError)):
        ifs_hydrostatic_columns(pi, *values[3:])


def write_generated(path, *, convention="ap", reverse=False):
    original = path.with_name("base.nc")
    generated_source(original, "3d")
    with Dataset(original) as ds:
        names = set(ds.variables) - {"pressure", "geopotential_height"}
    copy_variables(original, path, names)
    values = inputs(7)
    with Dataset(path, "a") as ds:
        ds.createDimension("interface", 8)
        lev = ds.createVariable("level", "f8", ("level",))
        lev[:], lev.units = np.arange(1, 8), "1"
        interface = ds.createVariable("interface", "f8", ("interface",))
        interface[:], interface.units, interface.standard_name = np.arange(8), "1", "atmosphere_hybrid_sigma_pressure_coordinate"
        interface.formula_terms = "ap: aa b: bb ps: ps" if convention == "ap" else "a: aa b: bb ps: ps p0: p0"
        for name, value, units in (("aa", values[0] if convention == "ap" else values[0] / 131072., "Pa" if convention == "ap" else "1"),
                                    ("bb", values[1], "1")):
            var = ds.createVariable(name, "f8", ("interface",))
            var[:], var.units = value.numpy(), units
        if convention == "a":
            var = ds.createVariable("p0", "f8")
            var[:], var.units = 131072., "Pa"
        for name, value, units, standard in (("ps", values[2], "Pa", "surface_air_pressure"),
                                              ("surface_phi", values[5], "m2 s-2", "surface_geopotential")):
            var = ds.createVariable(name, "f8", ("time", "latitude", "longitude"))
            var[0], var.units, var.standard_name = value.numpy(), units, standard
        for name, value in (("air_temperature", values[3]), ("specific_humidity", values[4])):
            ds[name][:] = value.numpy().transpose(2, 1, 0)[:, None]
        if reverse:
            for var in ds.variables.values():
                data = var[:]
                for axis, dimension in enumerate(var.dimensions):
                    if dimension in ("latitude", "longitude", "level", "interface"):
                        data = np.flip(data, axis=axis)
                var[:] = data
    original.unlink()
    return values


@pytest.mark.parametrize("defect", ["implicit", "labels", "layer_dimensions", "interfaces", "surface_pressure", "surface_phi", "units", "averaged", "supplied_height"])
def test_incomplete_or_ambiguous_reconstruction_rejects(tmp_path, defect):
    path = tmp_path / "source.nc"
    write_generated(path)
    option = "ifs-vapour-only"
    with Dataset(path, "a") as ds:
        if defect == "implicit":
            option = None
        elif defect == "labels":
            ds["level"][1] = 99
        elif defect == "layer_dimensions":
            ds.renameVariable("level", "actual_layer_ids")
            ds.createDimension("unrelated", 7)
            ids = ds.createVariable("level", "f8", ("unrelated",))
            ids[:], ids.units = np.arange(1, 8), "1"
        elif defect == "interfaces":
            ds["interface"][1] = 99
        elif defect == "surface_pressure":
            ds["bb"][-1] = .9
        elif defect == "surface_phi":
            ds["surface_phi"].delncattr("standard_name")
        elif defect == "units":
            ds["surface_phi"].units = "m"
        elif defect == "averaged":
            ds["surface_phi"].cell_methods = "time: mean"
        else:
            var = ds.createVariable("height", "f8", ds["air_temperature"].dimensions)
            var.standard_name, var.units = "geopotential_height", "m"
            var[:] = 1.
    with pytest.raises((ValueError, RuntimeError)):
        read_atmosphere(path, hybrid_convention=option)
