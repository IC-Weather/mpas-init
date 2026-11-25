from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import HorizontalGrid
from mpas_init.netcdf import read_near_surface
from test_atmosphere import generated_source
from test_source_collection import split_complete


NEAR_NAMES = {"air_temperature": "tt_near_surface", "relative_humidity": "rh_near_surface",
              "eastward_wind": "uu_near_surface", "northward_wind": "vv_near_surface"}


def generated_near(path, *, reverse=False):
    """Small nonconstant complete forecast with independently supplied source grids."""
    generated_source(path, "3d")
    with Dataset(path, "a") as ds:
        ds["temperature_2m"].delncattr("standard_name")
        for height in (2, 10):
            var = ds.createVariable(f"height_{height}m", "f8")
            var.assignValue(height)
            var.standard_name, var.units = "height", "m"
        for index, (name, varname) in enumerate(NEAR_NAMES.items()):
            lat = np.linspace(90., -90., 7 + 2 * index)
            if index == 2:
                lat = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(lat.size)[0][::-1]))
            lon = np.arange(12 + 4 * index) * 360. / (12 + 4 * index)
            yy, xx = np.meshgrid(lat, lon, indexing="ij")
            value = (280., 55., 5., -2.)[index] + .0001 * yy ** 2 + .001 * xx
            ydim, xdim = name + "_lat", name + "_lon"
            for dim, data, standard, units in ((ydim, lat, "latitude", "degrees_north"),
                                                 (xdim, lon, "longitude", "degrees_east")):
                ds.createDimension(dim, data.size)
                var = ds.createVariable(dim, "f8", (dim,))
                var[:], var.standard_name, var.units = data[::-1] if reverse else data, standard, units
                if reverse and standard == "longitude":
                    var[:] = var[:] - 360.
            var = ds.createVariable(varname, "f8", (xdim, "time", ydim))
            var[:] = (value[::-1, ::-1] if reverse else value).T[:, None]
            var.standard_name = name
            var.units = "K" if index == 0 else "%" if index == 1 else "m s-1"
            var.coordinates = f"height_{2 if index < 2 else 10}m {ydim} {xdim}"
            if index == 1:
                var.mpas_init_saturation_reference = "liquid_water"


@pytest.mark.parametrize("reverse", [False, True])
def test_cf_near_field_grids_and_split_files(tmp_path, reverse):
    path = tmp_path / "source.nc"
    generated_near(path, reverse=reverse)
    actual = read_near_surface(path)
    assert set(actual.field_grids) == set(NEAR_NAMES) - {"air_temperature"}
    assert actual.lead.total_seconds() == 21600
    for index, (name, value) in enumerate(actual.fields.items()):
        grid = actual.field_grids.get(name, HorizontalGrid(actual.latitude, actual.longitude))
        yy, xx = torch.meshgrid(grid.latitude, grid.longitude, indexing="ij")
        expected = (280., 55., 5., -2.)[index] + .0001 * yy ** 2 + .001 * xx
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert actual.field_grids["eastward_wind"].grid_type == "rectilinear_latlon"
    paths = split_complete(path, tmp_path / "split")
    split = read_near_surface(paths[::-1])
    for name in actual.fields:
        torch.testing.assert_close(split.fields[name], actual.fields[name], rtol=0, atol=0)
    for name in actual.field_grids:
        for axis in ("latitude", "longitude"):
            torch.testing.assert_close(getattr(split.field_grids[name], axis), getattr(actual.field_grids[name], axis), rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["missing_coordinate", "units", "gap", "height", "phase", "mean", "missing", "dimension", "duplicate"])
def test_invalid_near_field_grids_reject(tmp_path, defect):
    path = tmp_path / "source.nc"
    generated_near(path)
    with Dataset(path, "a") as ds:
        var = ds["uu_near_surface"]
        if defect == "missing_coordinate":
            ds["eastward_wind_lat"].delncattr("standard_name")
        elif defect == "units":
            ds["eastward_wind_lat"].units = "m"
        elif defect == "gap":
            ds["eastward_wind_lon"][:] *= .5
        elif defect == "height":
            var.coordinates = "height_2m eastward_wind_lat eastward_wind_lon"
        elif defect == "phase":
            ds["rh_near_surface"].mpas_init_saturation_reference = "ice"
        elif defect == "mean":
            var.cell_methods = "time: mean"
        elif defect == "missing":
            var[0, 0, 0] = np.ma.masked
        elif defect == "dimension":
            ds.renameVariable("uu_near_surface", "old_wind")
            ds["old_wind"].delncattr("standard_name")
            new = ds.createVariable("uu_near_surface", "f8", ("eastward_wind_lon", "eastward_wind_lat"))
            new.standard_name = "eastward_wind"
        else:
            new = ds.createVariable("duplicate", "f8", var.dimensions)
            new.standard_name = "eastward_wind"
    with pytest.raises(ValueError):
        read_near_surface(path)


def test_near_explicit_coordinate_mappings(tmp_path):
    path = tmp_path / "source.nc"
    generated_near(path)
    expected = read_near_surface(path)
    mapping = {}
    with Dataset(path, "a") as ds:
        for name in ("air_temperature", "eastward_wind"):
            for standard, suffix in (("latitude", "lat"), ("longitude", "lon")):
                varname = name + "_" + suffix
                ds[varname].delncattr("standard_name")
                mapping[name + "." + standard] = varname
    with pytest.raises(ValueError):
        read_near_surface(path)
    actual = read_near_surface(path, mapping)
    for name in expected.fields:
        torch.testing.assert_close(actual.fields[name], expected.fields[name], rtol=0, atol=0)
