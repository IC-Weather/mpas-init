from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import HorizontalGrid, remap_atmosphere
from mpas_init.netcdf import read_atmosphere
from test_atmosphere import generated_source
from test_source_collection import copy_variables, split_complete


STRIDES = {"air_temperature": 2, "specific_humidity": 3, "eastward_wind": 2,
           "northward_wind": 3, "relative_humidity": 2}


def prepare_multigrid(source, output, *, reverse=False):
    """Select complete global lattices independently for each atmospheric field."""
    with Dataset(source) as ds:
        variables = {getattr(var, "standard_name", ""): name for name, var in ds.variables.items() if var.ndim == 4}
        selected = {name: variables[name] for name in STRIDES}
        height = ds[variables["geopotential_height"]]
        latitude = next(var for var in ds.variables.values() if getattr(var, "standard_name", "") == "latitude")
        longitude = next(var for var in ds.variables.values() if getattr(var, "standard_name", "") == "longitude")
        lat, lon = np.asarray(latitude[:]), np.asarray(longitude[:])
        latitude_dim, longitude_dim = latitude.dimensions[0], longitude.dimensions[0]
        time_dim = next(var.dimensions[0] for var in ds.variables.values() if getattr(var, "standard_name", "") == "time")
        level_dim = next(dim for dim in height.dimensions if dim not in (time_dim, *latitude.dimensions, *longitude.dimensions))
        order = (time_dim, level_dim, latitude.dimensions[0], longitude.dimensions[0])
        values = {name: np.asarray(ds[varname][:]).transpose(tuple(ds[varname].dimensions.index(dim) for dim in order))
                  for name, varname in selected.items()}
        attributes = {name: {key: ds[varname].getncattr(key) for key in ds[varname].ncattrs() if key != "_FillValue"}
                      for name, varname in selected.items()}
        dtypes = {name: ds[varname].dtype for name, varname in selected.items()}
        names = set(ds.variables) - set(selected.values())
    copy_variables(source, output, names)
    strides = {}
    with Dataset(output, "a") as ds:
        for name in STRIDES:
            preferred = STRIDES[name]
            stride = preferred if lon.size % preferred == 0 and (lat.size - 1) % preferred == 0 else 2
            assert lon.size % stride == 0 and (lat.size - 1) % stride == 0
            strides[name] = stride
            ydim, xdim = name + "_lat", name + "_lon"
            for dim, data, standard, units in ((ydim, lat[::stride], "latitude", "degrees_north"),
                                                 (xdim, lon[::stride], "longitude", "degrees_east")):
                ds.createDimension(dim, data.size)
                var = ds.createVariable(dim, "f8", (dim,))
                var[:], var.standard_name, var.units = data, standard, units
            var = ds.createVariable(selected[name], dtypes[name], (xdim, time_dim, ydim, level_dim))
            var.setncatts(attributes[name])
            var.coordinates = ydim + " " + xdim
            var[:] = values[name][:, :, ::stride, ::stride].transpose(3, 0, 2, 1)
        if reverse:
            for var in ds.variables.values():
                data = var[:]
                for axis, dim in enumerate(var.dimensions):
                    if dim in (latitude_dim, longitude_dim, level_dim) or dim.endswith(("_lat", "_lon")):
                        data = np.flip(data, axis=axis)
                if getattr(var, "standard_name", "") == "longitude":
                    data = data - 360.
                var[:] = data
    return strides


@pytest.mark.parametrize("representation", ["3d", "a", "ap"])
@pytest.mark.parametrize("reverse", [False, True])
def test_cf_field_coordinates_levels_and_split_files(tmp_path, representation, reverse):
    original, path = tmp_path / "original.nc", tmp_path / "source.nc"
    generated_source(original, representation)
    strides = prepare_multigrid(original, path, reverse=reverse)
    baseline = read_atmosphere(original, include_relative_humidity=True)
    actual = read_atmosphere(path, include_relative_humidity=True)
    assert set(actual.field_grids) == set(STRIDES)
    torch.testing.assert_close(actual.latitude, baseline.latitude, rtol=0, atol=0)
    torch.testing.assert_close(actual.longitude, baseline.longitude, rtol=0, atol=0)
    for name, stride in strides.items():
        grid = actual.field_grids[name]
        expected = baseline.fields[name][:, ::stride, ::stride]
        if reverse:
            expected = expected.flip(0)
        torch.testing.assert_close(actual.fields[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(grid.latitude, baseline.latitude[::stride], rtol=0, atol=0)
        torch.testing.assert_close(grid.longitude, baseline.longitude[::stride], rtol=0, atol=0)
    split = split_complete(path, tmp_path / "split")
    decoded = read_atmosphere(split[::-1], include_relative_humidity=True)
    for name in actual.fields:
        torch.testing.assert_close(decoded.fields[name], actual.fields[name], rtol=0, atol=0)
    for name in actual.field_grids:
        for attribute in ("latitude", "longitude"):
            torch.testing.assert_close(getattr(decoded.field_grids[name], attribute), getattr(actual.field_grids[name], attribute), rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["latitude_units", "missing_coordinate", "nonperiodic", "wrong_vertical", "mean", "missing_value"])
def test_invalid_field_grid_rejects(tmp_path, defect):
    original, path = tmp_path / "original.nc", tmp_path / "source.nc"
    generated_source(original, "3d")
    prepare_multigrid(original, path)
    with Dataset(path, "a") as ds:
        if defect == "latitude_units":
            ds["eastward_wind_lat"].units = "m"
        elif defect == "missing_coordinate":
            ds["eastward_wind_lat"].delncattr("standard_name")
        elif defect == "nonperiodic":
            ds["eastward_wind_lon"][:] *= .5
        elif defect == "wrong_vertical":
            ds.renameVariable("eastward_wind", "original_wind")
            ds["original_wind"].delncattr("standard_name")
            ds.createDimension("unrelated_level", 7)
            var = ds.createVariable("eastward_wind", "f8", ("time", "unrelated_level", "eastward_wind_lat", "eastward_wind_lon"))
            var.standard_name, var.units = "eastward_wind", "m s-1"
            var[:] = 1.
        elif defect == "mean":
            ds["eastward_wind"].cell_methods = "time: mean"
        else:
            ds["eastward_wind"][0, 0, 0, 0] = np.ma.masked
    with pytest.raises(ValueError):
        read_atmosphere(path, include_relative_humidity=True)


def test_renamed_field_coordinates_require_explicit_mapping(tmp_path):
    original, path = tmp_path / "original.nc", tmp_path / "source.nc"
    generated_source(original, "3d")
    prepare_multigrid(original, path)
    expected = read_atmosphere(path)
    with Dataset(path, "a") as ds:
        for old, new in (("eastward_wind_lat", "yy"), ("eastward_wind_lon", "xx")):
            ds.renameVariable(old, new)
            ds[new].delncattr("standard_name")
        ds["eastward_wind"].coordinates = "yy xx"
    with pytest.raises(ValueError):
        read_atmosphere(path)
    actual = read_atmosphere(path, {"eastward_wind.latitude": "yy", "eastward_wind.longitude": "xx"})
    torch.testing.assert_close(actual.fields["eastward_wind"], expected.fields["eastward_wind"], rtol=0, atol=0)
    torch.testing.assert_close(actual.field_grids["eastward_wind"].latitude, expected.field_grids["eastward_wind"].latitude, rtol=0, atol=0)


def test_global_coordinate_mapping_still_applies_to_primary_fields(tmp_path):
    path = tmp_path / "source.nc"
    generated_source(path, "3d")
    expected = read_atmosphere(path)
    with Dataset(path, "a") as ds:
        ds["latitude"].delncattr("standard_name")
        ds["longitude"].delncattr("standard_name")
    actual = read_atmosphere(path, {"latitude": "latitude", "longitude": "longitude"})
    assert actual.field_grids is None
    for name in expected.fields:
        torch.testing.assert_close(actual.fields[name], expected.fields[name], rtol=0, atol=0)


def test_nonuniform_field_grid_and_primary_ownership(tmp_path):
    original, path = tmp_path / "original.nc", tmp_path / "source.nc"
    generated_source(original, "3d")
    prepare_multigrid(original, path)
    with Dataset(path, "a") as ds:
        ds["eastward_wind_lat"][:] = [80., 7., -80.]
    source = read_atmosphere(path)
    assert source.grid_type == "regular_latlon"
    assert source.field_grids["eastward_wind"].grid_type == "rectilinear_latlon"
    torch.testing.assert_close(source.field_grids["eastward_wind"].latitude, torch.tensor([80., 7., -80.], dtype=torch.float64), rtol=0, atol=0)
    with pytest.raises(ValueError, match="primary grid"):
        remap_atmosphere(source.fields, ("eastward_wind",), source.latitude, source.longitude,
                           torch.tensor([.2]), torch.tensor([1.]),
                           field_grids={"geopotential_height": HorizontalGrid(source.latitude, source.longitude)})


def test_height_reconstruction_rejects_noncollocated_humidity(tmp_path):
    from test_hybrid_height import write_generated
    path = tmp_path / "interfaces.nc"
    write_generated(path)
    with Dataset(path, "a") as ds:
        old = np.asarray(ds["specific_humidity"][:]).transpose(1, 3, 2, 0)
        ds.renameVariable("specific_humidity", "old_humidity")
        ds["old_humidity"].delncattr("standard_name")
        for dim, original, standard, units in (("qy", "latitude", "latitude", "degrees_north"),
                                                 ("qx", "longitude", "longitude", "degrees_east")):
            values = ds[original][::2]
            ds.createDimension(dim, len(values))
            var = ds.createVariable(dim, "f8", (dim,))
            var[:], var.standard_name, var.units = values, standard, units
        var = ds.createVariable("specific_humidity", "f8", ("time", "level", "qy", "qx"))
        var[:], var.standard_name, var.units = old[:, :, ::2, ::2], "specific_humidity", "kg kg-1"
    with pytest.raises(ValueError, match="collocated temperature and specific humidity"):
        read_atmosphere(path, hybrid_convention="ifs-vapour-only")


def test_direct_remap_analytic_values_and_coordinate_derivatives():
    y, x = torch.linspace(90., -90., 9, dtype=torch.float64), torch.arange(12, dtype=torch.float64) * 30.
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    height = torch.stack((1000. + .01 * yy, 5000. + .01 * yy))
    oy, ox = torch.linspace(90., -90., 13, dtype=torch.float64), torch.arange(24, dtype=torch.float64) * 15.
    gy, gx = torch.meshgrid(oy, ox, indexing="ij")
    value = torch.stack((20. + .0001 * gy * gy + .001 * gx, 40. + .0002 * gy * gy + .002 * gx))
    target_y, target_x = value.new_tensor([22.3, -12.7]), value.new_tensor([100.2, 146.3])
    def forward(value, target_y, target_x):
        return remap_atmosphere({"geopotential_height": height, "eastward_wind": value}, ("eastward_wind",),
                                 y, x, target_y * torch.pi / 180., target_x * torch.pi / 180.,
                                 field_grids={"eastward_wind": HorizontalGrid(oy, ox)})["eastward_wind"]
    expected = torch.stack((20. + .0001 * target_y ** 2 + .001 * target_x,
                            40. + .0002 * target_y ** 2 + .002 * target_x), dim=-1)
    torch.testing.assert_close(forward(value, target_y, target_x), expected, rtol=64 * torch.finfo(value.dtype).eps, atol=1.e-15)
    inputs = (value, target_y, target_x)
    directions = (value * .001, torch.ones_like(target_y) * .1, torch.ones_like(target_x) * .2)
    _, jvp = torch.func.jvp(forward, inputs, directions)
    analytic = expected * .001 + torch.stack((.0002 * target_y * .1 + .001 * .2,
                                              .0004 * target_y * .1 + .002 * .2), dim=-1)
    torch.testing.assert_close(jvp, analytic, rtol=64 * torch.finfo(value.dtype).eps, atol=1.e-15)
    _, pullback = torch.func.vjp(forward, *inputs)
    gradients = pullback(torch.ones_like(jvp))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in gradients)
    torch.testing.assert_close(sum((g * d).sum() for g, d in zip(gradients, directions)), jvp.sum(), rtol=64 * torch.finfo(value.dtype).eps, atol=1.e-15)
    for step in (1.e-2, 1.e-3, 1.e-4):
        yp, xp = target_y.numpy() + step * .1, target_x.numpy() + step * .2
        ym, xm = target_y.numpy() - step * .1, target_x.numpy() - step * .2
        def truth(yy, xx):
            return np.stack((20. + .0001 * yy ** 2 + .001 * xx, 40. + .0002 * yy ** 2 + .002 * xx), axis=-1)
        fd = torch.tensor(((1. + step * .001) * truth(yp, xp) - (1. - step * .001) * truth(ym, xm)) / (2 * step))
        torch.testing.assert_close(jvp, fd, rtol=2.e-6, atol=2.e-8)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(jvp * 0, fd, rtol=2.e-6, atol=2.e-8)


def test_nonuniform_source_axis_derivatives_are_independent():
    roots = np.polynomial.legendre.leggauss(12)[0][::-1].copy()
    y = torch.tensor(np.rad2deg(np.arcsin(roots)))
    x = torch.arange(24, dtype=torch.float64) * 15.
    row, column = torch.meshgrid(torch.arange(12, dtype=torch.float64), x, indexing="ij")
    value = torch.stack((20. + .01 * row ** 2 + .001 * column, 40. + .02 * row ** 2 + .002 * column))
    height = torch.stack((torch.zeros_like(row), torch.ones_like(row) * 5000.))
    lat, lon = y.new_tensor([22.3, -12.7]), y.new_tensor([100.2, 146.3])
    def forward(value, y, x):
        return remap_atmosphere({"geopotential_height": height, "eastward_wind": value}, ("eastward_wind",),
                                 torch.linspace(90., -90., 12, dtype=torch.float64), x,
                                 lat * torch.pi / 180., lon * torch.pi / 180.,
                                 field_grids={"eastward_wind": HorizontalGrid(y, x, "rectilinear_latlon")})["eastward_wind"]
    inputs, directions = (value, y, x), (value * .001, torch.ones_like(y) * .03, torch.ones_like(x) * .04)
    _, jvp = torch.func.jvp(forward, inputs, directions)
    _, pullback = torch.func.vjp(forward, *inputs)
    gradients = pullback(torch.ones_like(jvp))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in gradients)
    torch.testing.assert_close(sum((g * d).sum() for g, d in zip(gradients, directions)), jvp.sum(), rtol=64 * torch.finfo(value.dtype).eps, atol=1.e-15)
    def independent(step):
        indices = np.interp(lat.numpy(), (y.numpy() + step * .03)[::-1], np.arange(12)[::-1])
        xx = lon.numpy() - step * .04
        return (1. + step * .001) * np.stack((20. + .01 * indices ** 2 + .001 * xx,
                                               40. + .02 * indices ** 2 + .002 * xx), axis=-1)
    torch.testing.assert_close(forward(*inputs), torch.tensor(independent(0.)), rtol=64 * torch.finfo(value.dtype).eps, atol=1.e-15)
    for step in (1.e-2, 1.e-3, 1.e-4):
        fd = torch.tensor((independent(step) - independent(-step)) / (2 * step))
        torch.testing.assert_close(jvp, fd, rtol=2.e-6, atol=2.e-8)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(jvp * 0, fd, rtol=2.e-6, atol=2.e-8)
