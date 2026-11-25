from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import global_latlon_indices, sixteen_point
from mpas_init.netcdf import read_atmosphere
from test_atmosphere import generated_source
from test_vertical_grid import assert_stage


def gaussian_latitude(n):
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return np.rad2deg(np.arcsin(nodes[::-1]))


def make_cf_gaussian(path, representation="ap", *, reverse=False, latitude=None):
    generated_source(path, representation)
    latitude = gaussian_latitude(6) if latitude is None else latitude
    # The analytic generator uses five rows. Recreate only that dimension.
    original = path.with_suffix(".original.nc")
    path.rename(original)
    with Dataset(original) as src, Dataset(path, "w", clobber=False) as dst:
        dst.setncatts({key: src.getncattr(key) for key in src.ncattrs()})
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, latitude.size if name == "latitude" else len(dimension))
        for name, variable in src.variables.items():
            var = dst.createVariable(name, variable.dtype, variable.dimensions)
            var.setncatts({key: variable.getncattr(key) for key in variable.ncattrs() if key != "_FillValue"})
            values = np.asarray(variable[:])
            if name == "latitude":
                values = latitude
            elif "latitude" in variable.dimensions:
                indices = np.abs(np.asarray(src["latitude"][:])[:, None] - latitude).argmin(axis=0)
                values = np.take(values, indices, axis=variable.dimensions.index("latitude"))
            var[:] = values
    original.unlink()
    if reverse:
        with Dataset(path, "a") as ds:
            ds.set_auto_mask(False)
            for name, variable in ds.variables.items():
                values = np.asarray(variable[:])
                for axis, dimension in enumerate(variable.dimensions):
                    if dimension in ("latitude", "longitude", "level"):
                        values = np.flip(values, axis=axis)
                if name == "longitude":
                    values = values - 360.
                variable[:] = values
    return latitude


@pytest.mark.parametrize("representation", ["a", "ap", "3d"])
def test_gaussian_cf_preserves_nodes_values_and_axes(tmp_path, representation):
    first, second = tmp_path / "first.nc", tmp_path / "second.nc"
    latitude = make_cf_gaussian(first, representation)
    make_cf_gaussian(second, representation, reverse=True)
    a, b = read_atmosphere(first), read_atmosphere(second)
    assert a.grid_type == b.grid_type == "rectilinear_latlon"
    np.testing.assert_array_equal(a.latitude.numpy(), latitude)
    torch.testing.assert_close(a.pressure, b.pressure.flip(0), rtol=0, atol=0)
    for name, field in a.fields.items():
        torch.testing.assert_close(field, b.fields[name].flip(0), rtol=0, atol=0)


def test_float32_gaussian_coordinates_keep_supplied_precision(tmp_path):
    path = tmp_path / "source.nc"
    latitude = make_cf_gaussian(path)
    rounded = latitude.astype("f4").astype("f8")
    with Dataset(path, "a") as ds:
        ds["latitude"][:] = rounded
    source = read_atmosphere(path)
    assert source.grid_type == "rectilinear_latlon"
    np.testing.assert_array_equal(source.latitude.numpy(), rounded)
    assert not np.array_equal(rounded, latitude)


def test_nonuniform_supplied_coordinates_are_not_replaced_by_theoretical_roots(tmp_path):
    path = tmp_path / "source.nc"
    latitude = make_cf_gaussian(path)
    latitude[2] += .01
    with Dataset(path, "a") as ds:
        ds["latitude"][:] = latitude
    source = read_atmosphere(path)
    assert source.grid_type == "rectilinear_latlon"
    np.testing.assert_array_equal(source.latitude.numpy(), latitude)


@pytest.mark.parametrize("defect", ["missing_row", "duplicate_coordinate", "regional", "one_hemisphere", "repeated_seam"])
def test_incomplete_or_false_gaussian_grid_rejected(tmp_path, defect):
    path = tmp_path / "source.nc"
    latitude = make_cf_gaussian(path)
    with Dataset(path, "a") as ds:
        if defect == "missing_row":
            latitude = gaussian_latitude(8)[1:-1]
        elif defect == "duplicate_coordinate":
            latitude[2] = latitude[1]
        elif defect == "regional":
            latitude = np.linspace(70., -70., 6)
        elif defect == "one_hemisphere":
            latitude = gaussian_latitude(12)[:6]
        else:
            ds["longitude"][:] = np.linspace(0., 360., 8)
        ds["latitude"][:] = latitude
    with pytest.raises(ValueError, match="latitude|longitude|Gaussian"):
        read_atmosphere(path)


def test_gaussian_projection_and_field_derivatives():
    latitude = torch.tensor(gaussian_latitude(12))
    longitude = torch.arange(24, dtype=torch.float64) * 15.
    lat = torch.deg2rad(latitude.new_tensor([89., 63., 17., -31., -89.]))
    lon = torch.deg2rad(latitude.new_tensor([359., 31., 77., 156., 359.]))
    x, y = global_latlon_indices(latitude, longitude, lat, lon, grid_type="rectilinear_latlon")
    expected = []
    for target in np.rad2deg(lat.numpy()):
        if target > latitude[0]:
            expected.append(0.)
        elif target < latitude[-1]:
            expected.append(11.)
        else:
            j = next(j for j in range(11) if latitude[j] >= target >= latitude[j + 1])
            expected.append(j + (latitude[j].item() - target) / (latitude[j].item() - latitude[j + 1].item()))
    assert_stage(y, torch.tensor(expected), 16)
    rows, columns = torch.meshgrid(torch.arange(12, dtype=torch.float64),
                                    torch.arange(24, dtype=torch.float64), indexing="ij")
    field = 8. + columns * columns + .5 * rows * rows + 2. * rows * columns
    interior = slice(1, 4)
    torch.testing.assert_close(sixteen_point(field, x[interior], y[interior]),
                               8. + x[interior] ** 2 + .5 * y[interior] ** 2 + 2. * x[interior] * y[interior],
                               rtol=2.e-14, atol=2.e-14)
    def remap(f, a, b):
        xx, yy = global_latlon_indices(latitude, longitude, a, b, grid_type="rectilinear_latlon")
        return sixteen_point(f, xx, yy)
    assert torch.autograd.gradcheck(remap, (field.requires_grad_(), lat.requires_grad_(), lon.requires_grad_()))
    tangents = (torch.cos(field), torch.full_like(lat, .03), torch.full_like(lon, -.02))
    value, jvp = torch.func.jvp(remap, (field, lat, lon), tangents)
    weight = torch.arange(1., 6., dtype=torch.float64)
    _, pullback = torch.func.vjp(remap, field, lat, lon)
    vjp = pullback(weight)
    torch.testing.assert_close(torch.dot(weight, jvp), sum((a * b).sum() for a, b in zip(vjp, tangents)),
                               rtol=1.e-13, atol=1.e-13)
    assert torch.count_nonzero(jvp) == value.numel()
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(jvp), jvp, rtol=2.e-6, atol=2.e-8)
