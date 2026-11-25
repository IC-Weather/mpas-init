from dataclasses import replace

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import HorizontalGrid
from mpas_init.netcdf import UFS_SURFACE_VARIABLES, read_ufs_surface
from mpas_init.ufs_surface import initialize_ufs_surface
from test_forecast_selection import MEMBER, expand_forecast
from test_source_collection import split_complete
from test_ufs_surface import generated_inputs, write_source


def mixed_inputs():
    source, atmosphere, target, native, surface = generated_inputs()
    latitude = torch.linspace(-90., 90., 9, dtype=torch.float64)
    own = ("friction_velocity", "specific_humidity_2m", "air_temperature_2m",
           "canopy_water_amount", "deep_soil_temperature", "visible_strong_cosine_albedo")
    grids, fields = {}, {}
    for index, name in enumerate(source.fields):
        lat, lon = latitude, source.longitude
        if name in own:
            size = 7 + 2 * (index % 3)
            lat = torch.tensor(np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(size)[0])))
            lon = torch.arange(8 + 4 * (index % 3), dtype=torch.float64) * (360. / (8 + 4 * (index % 3))) + 7.5
            grids[name] = HorizontalGrid(lat, lon, "rectilinear_latlon")
        pattern = lat[:, None] / 90. + torch.sin(lon[None] * (torch.pi / 180.))
        scale = 280. if UFS_SURFACE_VARIABLES[name] == "K" else .01 if name == "specific_humidity_2m" else .2
        fields[name] = scale * (1. + .1 * pattern)
    return replace(source, latitude=latitude, fields=fields, field_grids=grids), atmosphere, target, native, surface


def assert_decoded(actual, expected):
    assert set(actual.fields) == set(expected.fields)
    assert set(actual.field_grids) == set(expected.field_grids)
    for name in actual.fields:
        torch.testing.assert_close(actual.fields[name], expected.fields[name], rtol=0, atol=0)
    for name in actual.field_grids:
        a, b = actual.field_grids[name], expected.field_grids[name]
        assert a.grid_type == b.grid_type and a.row_counts is b.row_counts is None
        for axis in ("latitude", "longitude"):
            torch.testing.assert_close(getattr(a, axis), getattr(b, axis), rtol=0, atol=0)
    for axis in ("latitude", "longitude"):
        torch.testing.assert_close(getattr(actual, axis), getattr(expected, axis), rtol=0, atol=0)
    assert (actual.valid_time, actual.reference_time, actual.lead, actual.member) == (
        expected.valid_time, expected.reference_time, expected.lead, expected.member)


def test_explicit_per_field_coordinate_and_quantity_mapping(tmp_path):
    source = mixed_inputs()[0]
    path = tmp_path / "source.nc"
    write_source(path, source)
    mapping = {"latitude": "latitude", "longitude": "longitude"}
    with Dataset(path, "a") as ds:
        for axis in ("latitude", "longitude"):
            alias = "primary_" + axis
            ds.renameVariable(axis, alias)
            ds[alias].delncattr("standard_name")
            mapping[axis] = alias
            mapping["aerodynamic_roughness_length." + axis] = alias
        for index, name in enumerate(source.fields):
            alias = f"field_{index}"
            ds.renameVariable(name, alias)
            ds[alias].delncattr("mpas_init_quantity")
            mapping[name] = alias
            if name in source.field_grids:
                for axis, suffix in (("latitude", "lat"), ("longitude", "lon")):
                    old, new = name + "_" + suffix, f"axis_{index}_{suffix}"
                    ds.renameVariable(old, new)
                    ds[new].delncattr("standard_name")
                    mapping[name + "." + axis] = new
                ds[alias].coordinates = "height_2m" if name.endswith("_2m") else ""
    with pytest.raises(ValueError):
        read_ufs_surface(path)
    assert_decoded(read_ufs_surface(path, mapping), source)


def test_multigrid_forecast_selection_and_split_files(tmp_path):
    source = mixed_inputs()[0]
    base, forecast = tmp_path / "base.nc", tmp_path / "forecast.nc"
    write_source(base, source)
    expand_forecast(base, forecast, matrix=True, invalid_unselected=True)
    selection = {"valid_time": source.valid_time, "reference_time": source.reference_time, "member": MEMBER}
    expected = replace(source, member=MEMBER)
    assert_decoded(read_ufs_surface(forecast, **selection), expected)
    paths = split_complete(forecast, tmp_path / "split")
    assert_decoded(read_ufs_surface(paths[::-1], **selection), expected)
    with pytest.raises(ValueError):
        read_ufs_surface(paths, **{**selection, "member": MEMBER-1})


@pytest.mark.parametrize("defect", ["unknown", "shape", "reduced", "type", "nan", "regional", "duplicate", "seam"])
def test_invalid_tensor_field_grid_rejects(defect):
    source, atmosphere, target, native, surface = mixed_inputs()
    name = "friction_velocity"
    grid = source.field_grids[name]
    if defect == "unknown":
        source.field_grids["not_a_quantity"] = grid
    elif defect == "shape":
        source.fields[name] = source.fields[name][:, :-1]
    else:
        if defect == "reduced":
            grid = replace(grid, row_counts=torch.full((grid.latitude.numel(),), grid.longitude.numel()))
        elif defect == "type":
            grid = replace(grid, grid_type="curvilinear")
        elif defect == "nan":
            grid = replace(grid, latitude=grid.latitude * torch.nan)
        elif defect == "regional":
            grid = replace(grid, latitude=grid.latitude * .5)
        elif defect == "duplicate":
            lat = grid.latitude.clone()
            lat[1] = lat[0]
            grid = replace(grid, latitude=lat)
        else:
            grid = replace(grid, longitude=grid.longitude * .5)
        source.field_grids[name] = grid
    with pytest.raises((ValueError, RuntimeError)):
        initialize_ufs_surface(source, atmosphere, target, native, surface)


@pytest.mark.parametrize("defect", ["coordinate", "units", "mean", "height", "missing", "regional", "duplicate", "clock"])
def test_invalid_cf_field_grid_rejects(tmp_path, defect):
    path = tmp_path / "source.nc"
    write_source(path, mixed_inputs()[0])
    name = "air_temperature_2m"
    with Dataset(path, "a") as ds:
        if defect == "coordinate":
            ds[name + "_lat"].delncattr("standard_name")
        elif defect == "units":
            ds[name + "_lon"].units = "m"
        elif defect == "mean":
            ds[name].cell_methods = "time: mean"
        elif defect == "height":
            ds["height_2m"].assignValue(10.)
        elif defect == "missing":
            ds[name][0, 0, 0] = np.nan
        elif defect == "regional":
            ds[name + "_lat"][:] *= .5
        elif defect == "duplicate":
            ds[name + "_lat"][1] = ds[name + "_lat"][0]
        else:
            ds["forecast_period"].assignValue(3600.)
    with pytest.raises(ValueError):
        read_ufs_surface(path)
