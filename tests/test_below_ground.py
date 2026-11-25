from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import MISSING, remap_atmosphere
from mpas_init.netcdf import read_atmosphere, read_pressure_levels
from mpas_init.thermodynamics import validate_atmospheric_support
from test_atmosphere import generated_source
from test_multigrid import prepare_multigrid
from test_source_collection import copy_variables


def masked_source(path, *, reverse=False, units="Pa"):
    p, fields = generated_source(path, "3d")
    ps = p[0]
    levels = np.array([98000., 90000., 70000., 45000., 25000., 5000., 200.])
    height = 287. * 280. * (1. + (461.6 / 287. - 1.) * .004) / 9.80616 * np.log(ps / levels[:, None, None])
    fields["geopotential_height"] = height
    fields["eastward_wind"] = 10. + .0001 * height
    fields["northward_wind"] = -3. + .00005 * height
    missing = levels[:, None, None] > ps
    with Dataset(path, "a") as ds:
        ds["pressure"].delncattr("standard_name")
        pressure = ds.createVariable("level", "f8", ("level",))
        pressure[:], pressure.standard_name, pressure.units = levels, "air_pressure", "Pa"
        surface = ds.createVariable("ps", "f8", ("longitude", "time", "latitude"))
        surface[:], surface.standard_name, surface.units = (ps / (100. if units == "hPa" else 1.)).T[:, None], "surface_air_pressure", units
        for name, value in fields.items():
            ds[name][:] = np.ma.array(value, mask=missing).transpose(2, 1, 0)[:, None]
        if reverse:
            for var in ds.variables.values():
                value = var[:]
                for axis, dim in enumerate(var.dimensions):
                    if dim in ("latitude", "longitude", "level"):
                        value = np.flip(value, axis)
                if getattr(var, "standard_name", "") == "longitude":
                    value = value - 360.
                var[:] = value
    return levels, ps, missing


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("units", ["Pa", "hPa"])
def test_cf_below_ground_mask_units_scans_and_split(tmp_path, reverse, units):
    path = tmp_path / "source.nc"
    levels, ps, missing = masked_source(path, reverse=reverse, units=units)
    actual = read_atmosphere(path, include_relative_humidity=True)
    assert set(actual.field_surface_pressure) == set(actual.fields)
    torch.testing.assert_close(actual.pressure, torch.tensor(levels[::-1].copy() if reverse else levels), rtol=0, atol=0)
    expected_mask = missing[::-1, ::-1] if reverse else missing[:, ::-1]
    for name, value in actual.fields.items():
        torch.testing.assert_close(actual.field_surface_pressure[name], torch.tensor(ps[::-1].copy()), rtol=0, atol=0)
        assert torch.equal(value == MISSING, torch.tensor(expected_mask.copy()))
    with pytest.raises(ValueError, match="missing or nonfinite"):
        read_pressure_levels(path, include_relative_humidity=True)
    with Dataset(path) as ds:
        names = set(ds.variables)
    parts = [copy_variables(path, tmp_path / "fields.nc", names - {"ps"}),
             copy_variables(path, tmp_path / "ps.nc", {"ps", "latitude", "longitude"})]
    split = read_atmosphere(parts[::-1], include_relative_humidity=True)
    for name in actual.fields:
        torch.testing.assert_close(split.fields[name], actual.fields[name], rtol=0, atol=0)
        torch.testing.assert_close(split.field_surface_pressure[name], actual.field_surface_pressure[name], rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["absent", "ambiguous", "units", "mean", "nonpositive", "masked", "above_ground", "coverage"])
def test_cf_below_ground_invalid_support_rejects(tmp_path, defect):
    path = tmp_path / "source.nc"
    masked_source(path)
    with Dataset(path, "a") as ds:
        if defect == "absent":
            ds["ps"].delncattr("standard_name")
        elif defect == "ambiguous":
            other = ds.createVariable("ps2", "f8", ds["ps"].dimensions)
            other[:], other.standard_name, other.units = ds["ps"][:], "surface_air_pressure", "Pa"
        elif defect == "units":
            ds["ps"].units = "1"
        elif defect == "mean":
            ds["ps"].cell_methods = "time: mean"
        elif defect == "nonpositive":
            ds["ps"][0, 0, 0] = 0.
        elif defect == "masked":
            ds["ps"][0, 0, 0] = np.ma.masked
        elif defect == "above_ground":
            ds["air_temperature"][0, 0, 0, -1] = np.ma.masked
        else:
            ds["ps"][0, 0, 0] = 250.
    with pytest.raises(ValueError):
        read_atmosphere(path, include_relative_humidity=True)


def test_renamed_surface_pressure_explicit_mapping(tmp_path):
    path = tmp_path / "source.nc"
    masked_source(path)
    expected = read_atmosphere(path)
    with Dataset(path, "a") as ds:
        ds.renameVariable("ps", "renamed_surface")
        ds["renamed_surface"].delncattr("standard_name")
    mapping = {name + ".surface_air_pressure": "renamed_surface" for name in expected.fields}
    actual = read_atmosphere(path, mapping)
    for name in actual.fields:
        torch.testing.assert_close(actual.field_surface_pressure[name], expected.field_surface_pressure[name], rtol=0, atol=0)


def test_below_ground_fields_own_collocated_surface_pressure(tmp_path):
    original, path = tmp_path / "original.nc", tmp_path / "multigrid.nc"
    _, ps, _ = masked_source(original)
    strides = prepare_multigrid(original, path)
    with Dataset(path, "a") as ds:
        for name, stride in strides.items():
            var = ds.createVariable("ps_" + name, "f8", ("time", name + "_lat", name + "_lon"))
            var[:], var.standard_name, var.units = ps[None, ::stride, ::stride], "surface_air_pressure", "Pa"
    source = read_atmosphere(path, include_relative_humidity=True)
    assert set(source.field_surface_pressure) == set(source.fields)
    for name, stride in strides.items():
        expected = torch.tensor(ps[::stride, ::stride][::-1].copy())
        torch.testing.assert_close(source.field_surface_pressure[name], expected, rtol=0, atol=0)
    validate_atmospheric_support(source.fields, source.pressure, source.field_surface_pressure)


def test_fractional_relative_humidity_preserves_missing_sentinel(tmp_path):
    path = tmp_path / "source.nc"
    masked_source(path)
    expected = read_atmosphere(path, include_relative_humidity=True)
    with Dataset(path, "a") as ds:
        ds["relative_humidity"][:] = ds["relative_humidity"][:] / 100.
        ds["relative_humidity"].units = "1"
    actual = read_atmosphere(path, include_relative_humidity=True)
    torch.testing.assert_close(actual.fields["relative_humidity"], expected.fields["relative_humidity"], rtol=0, atol=0)
    validate_atmospheric_support(actual.fields, actual.pressure, actual.field_surface_pressure)


@pytest.mark.parametrize("defect", ["undeclared", "unknown", "shape", "precision", "above_ground", "pressure", "all_missing_level"])
def test_tensor_api_missing_support_validation(tmp_path, defect):
    path = tmp_path / "source.nc"
    masked_source(path)
    source = read_atmosphere(path)
    fields, surfaces, pressure = dict(source.fields), dict(source.field_surface_pressure), source.pressure
    if defect == "undeclared":
        surfaces.pop("air_temperature")
    elif defect == "unknown":
        surfaces["unknown"] = surfaces["air_temperature"]
    elif defect == "shape":
        surfaces["air_temperature"] = surfaces["air_temperature"][:, :-1]
    elif defect == "precision":
        surfaces["air_temperature"] = surfaces["air_temperature"].float()
    elif defect == "above_ground":
        fields["air_temperature"] = fields["air_temperature"].clone()
        fields["air_temperature"][-1, 0, 0] = MISSING
    elif defect == "pressure":
        pressure = pressure.clone()
        pressure[-1] = 0.
    else:
        surfaces = {name: torch.full_like(ps, 95000.) for name, ps in surfaces.items()}
        fields = {name: torch.cat((torch.full_like(value[:1], MISSING), value[1:])) for name, value in fields.items()}
    with pytest.raises((ValueError, RuntimeError)):
        names = validate_atmospheric_support(fields, pressure, surfaces)
        remap_atmosphere(fields, ("air_temperature",), source.latitude, source.longitude,
                        torch.tensor([0.]), torch.tensor([0.]), missing_fields=names)
