from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import HorizontalGrid, global_latlon_indices, remap_atmosphere, sixteen_point
from mpas_init.netcdf import read_atmosphere
from test_gaussian import gaussian_latitude, make_cf_gaussian
from test_source_collection import split_complete


def gather_cf(source, output, counts, *, shuffle=False):
    """Select complete periodic subsets from each original row; no resampling."""
    with Dataset(source) as src, Dataset(output, "w", clobber=False) as dst:
        latitude = next(v for v in src.variables.values() if getattr(v, "standard_name", "") == "latitude")
        longitude = next(v for v in src.variables.values() if getattr(v, "standard_name", "") == "longitude")
        yd, xd = latitude.dimensions[0], longitude.dimensions[0]
        width = longitude.size
        assert len(counts) == latitude.size and all(width % n == 0 for n in counts)
        rows = np.repeat(np.arange(latitude.size), counts)
        columns = np.concatenate([np.arange(0, width, width//n) for n in counts])
        indices = rows*width + columns
        order = np.random.default_rng(891).permutation(indices.size) if shuffle else np.arange(indices.size)
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim))
        dst.createDimension("rgrid", indices.size)
        index = dst.createVariable("rgrid", "i8", ("rgrid",))
        index.compress, index[:] = yd + " " + xd, indices[order]
        for name, var in src.variables.items():
            data = np.asarray(var[:])
            dimensions = var.dimensions
            if var.name == latitude.name:
                dimensions, data = ("rgrid",), data[rows][order]
            elif var.name == longitude.name:
                dimensions, data = ("rgrid",), data[columns][order]
            elif yd in dimensions and xd in dimensions:
                remaining = tuple(d for d in dimensions if d not in (yd, xd))
                axes = tuple(dimensions.index(d) for d in (*remaining, yd, xd))
                data = data.transpose(axes)[..., rows, columns][..., order]
                dimensions = (*remaining, "rgrid")
            target = dst.createVariable(name, var.dtype, dimensions)
            target.setncatts({key: var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
            target[:] = data
            if "rgrid" in dimensions and name != "rgrid":
                target.coordinates = latitude.name + " " + longitude.name
        dst.setncatts({key: src.getncattr(key) for key in src.ncattrs()})
    return rows, columns


@pytest.mark.parametrize("representation", ["3d", "a", "ap"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_cf_gathered_pressure_fields_and_selection(tmp_path, representation, shuffle):
    original, path = tmp_path / "original.nc", tmp_path / "source.nc"
    make_cf_gaussian(original, representation)
    counts = [4, 4, 8, 8, 4, 4]
    rows, columns = gather_cf(original, path, counts, shuffle=shuffle)
    regular = read_atmosphere(original, include_relative_humidity=True)
    actual = read_atmosphere(path, include_relative_humidity=True, valid_time=regular.valid_time)
    assert actual.grid_type == "reduced_gaussian" and actual.field_grids is None
    torch.testing.assert_close(actual.row_counts, torch.tensor(counts), rtol=0, atol=0)
    torch.testing.assert_close(actual.latitude, regular.latitude, rtol=0, atol=0)
    torch.testing.assert_close(actual.longitude, regular.longitude[columns], rtol=0, atol=0)
    torch.testing.assert_close(actual.pressure, regular.pressure[:, rows, columns], rtol=0, atol=0)
    for name in regular.fields:
        torch.testing.assert_close(actual.fields[name], regular.fields[name][:, rows, columns], rtol=0, atol=0)
    split = split_complete(path, tmp_path / "split")
    decoded = read_atmosphere(split, include_relative_humidity=True)
    torch.testing.assert_close(decoded.pressure, actual.pressure, rtol=0, atol=0)
    for name in actual.fields:
        torch.testing.assert_close(decoded.fields[name], actual.fields[name], rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["compress", "duplicate_index", "outside_index", "missing_row", "row_latitude",
                                   "longitude_gap", "duplicate_longitude", "regional_latitude", "missing_value"])
def test_cf_invalid_reduced_grids_reject(tmp_path, defect):
    original, path = tmp_path / "original.nc", tmp_path / "source.nc"
    make_cf_gaussian(original)
    gather_cf(original, path, [4, 4, 8, 8, 4, 4])
    with Dataset(path, "a") as ds:
        if defect == "compress":
            ds["rgrid"].compress = "latitude absent"
        elif defect == "duplicate_index":
            ds["rgrid"][1] = ds["rgrid"][0]
        elif defect == "outside_index":
            ds["rgrid"][1] = 10000
        elif defect == "missing_row":
            ds["rgrid"][:4] = np.arange(4)+12
        elif defect == "row_latitude":
            ds["latitude"][1] += .01
        elif defect == "longitude_gap":
            ds["longitude"][1] += 1.
        elif defect == "duplicate_longitude":
            ds["longitude"][1] = ds["longitude"][0]
        elif defect == "regional_latitude":
            ds["latitude"][:] *= .5
        else:
            ds["geopotential_height"][0, 0, 0] = np.ma.masked
    with pytest.raises(ValueError):
        read_atmosphere(path)


def test_float32_gathered_coordinates_preserve_storage_precision(tmp_path):
    original,expanded,path = tmp_path/"original.nc",tmp_path/"expanded.nc",tmp_path/"source.nc"
    make_cf_gaussian(original)
    with Dataset(original) as src, Dataset(expanded,"w",clobber=False) as dst:
        for name,dim in src.dimensions.items():
            dst.createDimension(name,56 if name == "longitude" else len(dim))
        for name,var in src.variables.items():
            coordinate = name in ("latitude","longitude")
            target = dst.createVariable(name,"f4" if coordinate else var.dtype,var.dimensions)
            target.setncatts({key:var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
            values = np.asarray(var[:])
            if name == "longitude":
                values = np.arange(56)*360./56
            elif "longitude" in var.dimensions:
                values = np.take(values,np.arange(56)%8,axis=var.dimensions.index("longitude"))
            target[:] = values
    counts = [4,7,14,14,7,4]
    rows,columns = gather_cf(expanded,path,counts,shuffle=True)
    actual = read_atmosphere(path)
    with Dataset(expanded) as ds:
        expected_lon = np.asarray(ds["longitude"][:],dtype="f8")[columns]
        var = ds["air_temperature"]
        axes = tuple(var.dimensions.index(d) for d in ("time","level","latitude","longitude"))
        expected = np.asarray(var[:]).transpose(axes)[0][:,rows,columns]
    np.testing.assert_array_equal(actual.longitude.numpy(),expected_lon)
    np.testing.assert_array_equal(actual.fields["air_temperature"].numpy(),expected)
    torch.testing.assert_close(actual.row_counts,torch.tensor(counts),rtol=0,atol=0)
    with Dataset(path,"a") as ds:
        ds["longitude"][3] += .01
    with pytest.raises(ValueError,match="periodic"):
        read_atmosphere(path)


@pytest.mark.parametrize("matrix", [False,True])
def test_reduced_forecast_selection_ignores_unselected_missing_values(tmp_path,matrix):
    from test_forecast_selection import MEMBER,expand_forecast
    original,reduced,forecast = tmp_path/"original.nc",tmp_path/"reduced.nc",tmp_path/"forecast.nc"
    make_cf_gaussian(original)
    gather_cf(original,reduced,[4,4,8,8,4,4],shuffle=True)
    expand_forecast(reduced,forecast,matrix=matrix,invalid_unselected=True)
    expected = read_atmosphere(reduced,include_relative_humidity=True)
    actual = read_atmosphere(forecast,valid_time=expected.valid_time,reference_time=expected.reference_time,
                             member=MEMBER,include_relative_humidity=True)
    assert actual.member == MEMBER
    torch.testing.assert_close(actual.pressure,expected.pressure,rtol=0,atol=0)
    for name in expected.fields:
        torch.testing.assert_close(actual.fields[name],expected.fields[name],rtol=0,atol=0)


def grid_fixture():
    latitude = torch.tensor(gaussian_latitude(12))
    counts = torch.tensor([20, 24, 28, 32, 36, 40, 40, 36, 32, 28, 24, 20])
    longitude = torch.cat([torch.arange(int(n), dtype=torch.float64)*(360./int(n)) for n in counts])
    return HorizontalGrid(latitude, longitude, "reduced_gaussian", counts)


def test_reduced_rectangular_limit_and_zero_handling():
    latitude = torch.tensor(gaussian_latitude(12))
    longitude = torch.arange(24, dtype=torch.float64)*15.
    counts = torch.full((12,), 24, dtype=torch.int64)
    lat = torch.deg2rad(latitude.new_tensor([90., 73.1, 2.41, -61.7, -90.]))
    lon = torch.deg2rad(latitude.new_tensor([0., 359.1, 51.3, 178.2, 360.]))
    field = torch.rand((3, 12, 24), dtype=torch.float64, generator=torch.Generator().manual_seed(872))
    field[:, ::3, ::3] = 0
    x, y = global_latlon_indices(latitude, longitude, lat, lon, grid_type="rectilinear_latlon")
    xx, yy = global_latlon_indices(latitude, longitude.repeat(12), lat, lon,
                                   grid_type="reduced_gaussian", row_counts=counts)
    actual = sixteen_point(field.flatten(-2), xx, yy, row_counts=counts)
    torch.testing.assert_close(actual, sixteen_point(field, x, y), rtol=0, atol=0)


def test_reduced_fields_and_coordinate_derivatives():
    grid = grid_fixture()
    field = (1. + torch.arange(grid.longitude.numel(), dtype=torch.float64)*.001).requires_grad_()
    lat = torch.deg2rad(grid.latitude.new_tensor([61.21, 17.31, -31.23])).requires_grad_()
    lon = torch.deg2rad(grid.latitude.new_tensor([31.23, 77.12, 156.23])).requires_grad_()
    def evaluate(f, a, b):
        x, y = global_latlon_indices(grid.latitude, grid.longitude, a, b,
                                     grid_type=grid.grid_type, row_counts=grid.row_counts)
        return sixteen_point(f, x, y, row_counts=grid.row_counts)
    field_only = lambda f: evaluate(f, lat, lon)
    assert torch.autograd.gradcheck(field_only, (field,))
    # Enumerate every field basis in one batch, with gradcheck's unchanged step
    # and bounds. Coordinate and undefined-tangent checks still use gradcheck.
    with torch.no_grad():
        perturbations = torch.eye(field.numel(), dtype=field.dtype)*1e-6
        field_fd = ((field_only(field + perturbations) - field_only(field - perturbations))/(2e-6)).T
    field_jacobian = torch.func.jacfwd(field_only)(field)
    torch.testing.assert_close(field_jacobian, field_fd, rtol=1e-3, atol=1e-5)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(field_jacobian), field_fd, rtol=1e-3, atol=1e-5)
    assert torch.autograd.gradcheck(lambda a, b: evaluate(field, a, b), (lat, lon), check_forward_ad=True)
    offset = field.new_zeros((), requires_grad=True)
    assert torch.autograd.gradcheck(lambda delta: field_only(field + delta), (offset,), check_forward_ad=True)
    directions = (torch.cos(field), torch.full_like(lat, .01), torch.full_like(lon, -.02))
    _, tangent = torch.func.jvp(evaluate, (field, lat, lon), directions)
    for step in (1e-3, 5e-4, 2.5e-4):
        fd = (evaluate(*(v+step*d for v,d in zip((field,lat,lon),directions)))
              - evaluate(*(v-step*d for v,d in zip((field,lat,lon),directions))))/(2*step)
        torch.testing.assert_close(tangent, fd, rtol=2e-6, atol=2e-8)
    weights = tangent.new_tensor([.3, -.7, 1.1])
    gradients = torch.autograd.grad((evaluate(field, lat, lon)*weights).sum(), (field, lat, lon))
    torch.testing.assert_close(sum((g*d).sum() for g, d in zip(gradients, directions)),
                               (tangent*weights).sum(), rtol=2e-6, atol=2e-8)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(tangent), fd, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("convention", ["a", "ap"])
@pytest.mark.parametrize("reverse", [False, True])
def test_reduced_complete_interface_reconstruction(tmp_path, convention, reverse):
    from test_hybrid_height import write_generated
    original, path = tmp_path/"original.nc", tmp_path/"source.nc"
    write_generated(original,convention=convention,reverse=reverse)
    rows,columns = gather_cf(original,path,[4,4,8,4,4],shuffle=True)
    with Dataset(original) as ds:
        latitude,longitude = np.asarray(ds["latitude"][:]),np.remainder(np.asarray(ds["longitude"][:]),360.)
    order = np.lexsort((longitude[columns],-latitude[rows]))
    rows = np.argsort(np.argsort(-latitude))[rows][order]
    columns = np.argsort(np.argsort(longitude))[columns][order]
    regular = read_atmosphere(original,hybrid_convention="ifs-vapour-only")
    actual = read_atmosphere(path,hybrid_convention="ifs-vapour-only")
    assert actual.pressure.ndim == 2 and actual.vertical_convention == "ifs-vapour-only"
    torch.testing.assert_close(actual.pressure,regular.pressure[:,rows,columns],rtol=0,atol=0)
    for name in regular.fields:
        torch.testing.assert_close(actual.fields[name],regular.fields[name][:,rows,columns],rtol=0,atol=0)
    split = split_complete(path,tmp_path/"split")
    decoded = read_atmosphere(split,hybrid_convention="ifs-vapour-only")
    torch.testing.assert_close(decoded.pressure,actual.pressure,rtol=0,atol=0)
    torch.testing.assert_close(decoded.fields["geopotential_height"],actual.fields["geopotential_height"],rtol=0,atol=0)


@pytest.mark.parametrize("reduced_primary", [False,True])
def test_mixed_regular_and_reduced_field_grids(tmp_path, reduced_primary):
    from test_source_collection import copy_variables
    original,reduced,path = tmp_path/"original.nc",tmp_path/"reduced.nc",tmp_path/"mixed.nc"
    make_cf_gaussian(original)
    gather_cf(original,reduced,[4,4,8,8,4,4],shuffle=True)
    primary,secondary = (reduced,original) if reduced_primary else (original,reduced)
    with Dataset(primary) as ds:
        copy_variables(primary,path,set(ds.variables)-{"eastward_wind"})
    with Dataset(secondary) as src, Dataset(path,"a") as ds:
        wind = src["eastward_wind"]
        dimensions = set(wind.dimensions)-{"time","level"}
        if not reduced_primary:
            dimensions.update(src["rgrid"].compress.split())
        for dim in dimensions:
            ds.createDimension("wind_"+dim,len(src.dimensions[dim]))
        for name in ("latitude","longitude","rgrid") if not reduced_primary else ("latitude","longitude"):
            var = src[name]
            target = ds.createVariable("wind_"+name,var.dtype,tuple("wind_"+d for d in var.dimensions))
            target.setncatts({key:var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
            if name == "rgrid":
                target.compress = " ".join("wind_"+d for d in var.compress.split())
            target[:] = var[:]
        var = ds.createVariable("eastward_wind",wind.dtype,tuple("wind_"+d if d in dimensions else d for d in wind.dimensions))
        var.setncatts({key:wind.getncattr(key) for key in wind.ncattrs() if key != "_FillValue"})
        var.coordinates = "wind_latitude wind_longitude"
        var[:] = wind[:]
    actual,own = read_atmosphere(path),read_atmosphere(secondary)
    assert set(actual.field_grids) == {"eastward_wind"}
    torch.testing.assert_close(actual.fields["eastward_wind"],own.fields["eastward_wind"],rtol=0,atol=0)
    targets = torch.deg2rad(actual.latitude.new_tensor([61.21,17.31,-31.23]))
    longitude = torch.deg2rad(actual.latitude.new_tensor([31.23,77.12,156.23]))
    mapped = remap_atmosphere(actual.fields,("eastward_wind",),actual.latitude,actual.longitude,targets,longitude,
                              grid_type=actual.grid_type,row_counts=actual.row_counts,field_grids=actual.field_grids)
    x,y = global_latlon_indices(own.latitude,own.longitude,targets,longitude,grid_type=own.grid_type,row_counts=own.row_counts)
    torch.testing.assert_close(mapped["eastward_wind"],sixteen_point(own.fields["eastward_wind"],x,y,row_counts=own.row_counts).T,
                               rtol=0,atol=0)
