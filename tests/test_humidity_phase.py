"""Explicit source-grid RH normalization against an independent scalar formula."""

import math

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import MISSING
from mpas_init.netcdf import read_atmosphere, read_near_surface
from mpas_init.thermodynamics import wps_ice_blend_to_liquid
from test_atmosphere import generated_source
from test_below_ground import masked_source
from test_compiler_backend import compiler, observed_regions
from test_grib import codes, write_source, rewrite
from test_multigrid import prepare_multigrid
from test_near_multigrid import generated_near
from test_vertical_grid import assert_stage


PHASE = {"relative_humidity.saturation_reference": "wps_ice_blend"}


def scalar_phase(temperature, humidity):
    def point(t, rh):
        if rh == float(np.asarray(MISSING, dtype=humidity.dtype)) or t > 273.15:
            return rh
        ice = .01 * math.exp(9.550426 - 5723.265 / t + 3.53068 * math.log(t) - .00728332 * t)
        water = 6.112 * math.exp(17.67 * (t - 273.15) / (t - 273.15 + 243.5))
        weight = (273.15 - t) / 20.
        mixed = weight * ice + (1. - weight) * water if t > 253.15 else ice
        return rh * (mixed / water)
    return np.array([point(float(t), float(rh)) for t, rh in zip(temperature.flat, humidity.flat)]).reshape(temperature.shape)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_phase_values_boundaries_masks_and_controls(dtype):
    temperature = torch.tensor([180., 233., 253.15, 260., 273.15, 280., 310., float("nan")], dtype=dtype)
    humidity = torch.tensor([0., 18., 35., 57., 100., 108., 78., MISSING], dtype=dtype)
    actual = wps_ice_blend_to_liquid(temperature, humidity)
    expected = torch.tensor(scalar_phase(temperature.numpy(), humidity.numpy()), dtype=dtype)
    assert_stage(actual, expected, 64)
    assert actual[-1] == MISSING
    inverse = humidity.square() / torch.where(actual == 0, torch.ones_like(actual), actual)
    inverse = torch.where(humidity == MISSING, humidity, inverse)
    assert torch.isfinite(inverse).all()
    for wrong in (humidity, inverse):
        with pytest.raises(AssertionError):
            assert_stage(wrong, expected, 64)


@pytest.mark.parametrize("defect", ("shape", "missing_temperature", "zero_temperature", "nan_rh", "singular_temperature"))
def test_phase_invalid_support_rejects(defect):
    temperature, humidity = torch.tensor([260., 280.]), torch.tensor([60., 80.])
    if defect == "shape":
        temperature = temperature[:1]
    elif defect == "missing_temperature":
        temperature[0] = MISSING
    elif defect == "zero_temperature":
        temperature[0] = 0.
    elif defect == "nan_rh":
        humidity[0] = float("nan")
    else:
        temperature[0] = 29.65
    with pytest.raises((ValueError, RuntimeError)):
        wps_ice_blend_to_liquid(temperature, humidity)


@pytest.mark.parametrize("representation", ("3d", "a", "ap"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("units", ("%", "1"))
def test_cf_phase_on_source_grid_before_interpolation(tmp_path, representation, reverse, units):
    path = tmp_path / "source.nc"
    generated_source(path, representation, reverse=reverse)
    with Dataset(path, "a") as ds:
        t = ds["air_temperature"]
        t[:] = 225. + np.arange(t.size).reshape(t.shape) % 72
        ds["relative_humidity"].mpas_init_saturation_reference = "wps_ice_blend"
        if units == "1":
            ds["relative_humidity"][:] = ds["relative_humidity"][:] / 100.
            ds["relative_humidity"].units = units
    source = read_atmosphere(path, include_relative_humidity=True)
    expected = scalar_phase(source.fields["air_temperature"].numpy(), np.full((7, 5, 8), 45.))
    assert_stage(source.fields["relative_humidity"], torch.tensor(expected), 64)
    with Dataset(path, "a") as ds:
        ds["relative_humidity"].delncattr("mpas_init_saturation_reference")
    explicit = read_atmosphere(path, PHASE, include_relative_humidity=True)
    assert torch.equal(source.fields["relative_humidity"], explicit.fields["relative_humidity"])
    with pytest.raises(ValueError, match="saturation reference"):
        read_atmosphere(path, include_relative_humidity=True)


def test_cf_phase_conflicts_and_independent_grids(tmp_path):
    source, multigrid = tmp_path / "source.nc", tmp_path / "multigrid.nc"
    generated_source(source, "3d")
    with pytest.raises(ValueError, match="conflicting"):
        read_atmosphere(source, PHASE, include_relative_humidity=True)
    prepare_multigrid(source, multigrid)
    with Dataset(multigrid, "a") as ds:
        ds["relative_humidity"].mpas_init_saturation_reference = "wps_ice_blend"
        ds["air_temperature_lon"][:] = ds["air_temperature_lon"][:] + .1
    with pytest.raises(ValueError, match="collocated"):
        read_atmosphere(multigrid, include_relative_humidity=True)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("units", ("%", "1"))
def test_cf_phase_preserves_declared_below_ground_masks(tmp_path, reverse, units):
    path = tmp_path / "source.nc"
    masked_source(path, reverse=reverse)
    with Dataset(path, "a") as ds:
        ds["air_temperature"][:] = ds["air_temperature"][:] * 0. + 240.
        rh = ds["relative_humidity"]
        rh.mpas_init_saturation_reference = "wps_ice_blend"
        if units == "1":
            rh[:] = rh[:] / 100.
            rh.units = units
    source = read_atmosphere(path, include_relative_humidity=True)
    t, rh = source.fields["air_temperature"], source.fields["relative_humidity"]
    assert torch.equal(t == MISSING, rh == MISSING)
    raw = torch.where(rh == MISSING, torch.full_like(rh, MISSING), torch.full_like(rh, 45.)).numpy()
    assert_stage(rh, torch.tensor(scalar_phase(t.numpy(), raw)), 64)


def test_near_surface_phase_requires_collocation(tmp_path):
    path = tmp_path / "source.nc"
    generated_near(path)
    with Dataset(path, "a") as ds:
        ds["rh_near_surface"].mpas_init_saturation_reference = "wps_ice_blend"
    with pytest.raises(ValueError, match="collocated"):
        read_near_surface(path)


@pytest.mark.parametrize("units", ("%", "1"))
def test_near_surface_phase_values_and_units(tmp_path, units):
    path = tmp_path / "source.nc"
    generated_source(path, "3d")
    temperature = 235. + np.arange(40).reshape(5, 8) * 1.4
    humidity = 35. + np.arange(40).reshape(5, 8)
    with Dataset(path, "a") as ds:
        ds["temperature_2m"].delncattr("standard_name")
        for height in (2, 10):
            coordinate = ds.createVariable(f"height_{height}m", "f8")
            coordinate.assignValue(height)
            coordinate.standard_name, coordinate.units = "height", "m"
        for name, values, unit, height in (("air_temperature", temperature, "K", 2),
                                           ("relative_humidity", humidity / (100. if units == "1" else 1.), units, 2),
                                           ("eastward_wind", np.full((5, 8), 5.), "m s-1", 10),
                                           ("northward_wind", np.full((5, 8), -2.), "m s-1", 10)):
            variable = ds.createVariable("near_" + name, "f8", ("time", "latitude", "longitude"))
            variable[:], variable.units, variable.standard_name = values[None], unit, name
            variable.coordinates = f"height_{height}m latitude longitude"
            if name == "relative_humidity":
                variable.mpas_init_saturation_reference = "wps_ice_blend"
    source = read_near_surface(path)
    expected = scalar_phase(temperature, humidity)[::-1].copy()
    assert_stage(source.fields["relative_humidity"], torch.tensor(expected), 64)


@pytest.mark.parametrize("scan", ({}, {"reverse_lat": True, "reverse_lon": True, "column_scan": True}))
def test_grib_phase_source_values_and_collocation(tmp_path, codes, scan):
    path = tmp_path / "source.grib2"
    expected, _, _ = write_source(codes, path, **scan)
    def warm(handle):
        if codes.codes_get(handle, "shortName") == "t":
            codes.codes_set_values(handle, codes.codes_get_values(handle) + 220.)
    rewrite(codes, path, warm)
    expected["air_temperature"] += 220.
    source = read_atmosphere(path, PHASE, include_relative_humidity=True)
    expected["relative_humidity"] = scalar_phase(expected["air_temperature"], expected["relative_humidity"])
    for name, value in expected.items():
        assert_stage(source.fields[name], torch.tensor(value), 64)
    def change(handle):
        if codes.codes_get(handle, "shortName") == "r":
            for key in ("longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees"):
                codes.codes_set(handle, key, (codes.codes_get(handle, key) + .1) % 360.)
    rewrite(codes, path, change)
    with pytest.raises(ValueError, match="collocated"):
        read_atmosphere(path, PHASE, include_relative_humidity=True)


@pytest.mark.compiler
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_compiled_phase_values_and_both_ad_modes(compiler, observed_regions, dtype):
    inputs = tuple(torch.tensor(value, dtype=dtype, requires_grad=True) for value in
                   ([195., 233., 251., 255., 266., 272., 279., 310.], [3., 22., 44., 68., 96., 105., 55., 0.]))
    compiled = torch.compile(wps_ice_blend_to_liquid, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    actual = compiled(*inputs)
    expected = wps_ice_blend_to_liquid(*inputs)
    assert_stage(actual, expected, 64)
    weights = torch.cos(torch.arange(8, dtype=dtype))
    gradients = torch.autograd.grad((actual * weights).sum(), inputs)
    for actual_gradient, expected_gradient in zip(gradients,
                                                  torch.autograd.grad((expected * weights).sum(), inputs)):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2.e-4 if dtype == torch.float32 else 2.e-6, atol=0.)
    def transformed(t, rh, dt, drh):
        return torch.func.jvp(wps_ice_blend_to_liquid, (t, rh), (dt, drh))
    compiled_jvp = torch.compile(transformed, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    directions = (torch.sin(weights), torch.cos(weights))
    _, tangent = compiled_jvp(*inputs, *directions)
    _, expected_tangent = transformed(*inputs, *directions)
    torch.testing.assert_close(tangent, expected_tangent, rtol=2.e-4 if dtype == torch.float32 else 2.e-6, atol=0.)
    assert observed_regions and all(region["calls"] > 0 for region in observed_regions)
    with pytest.raises(RuntimeError, match="positive collocated"):
        compiled(-inputs[0], inputs[1])
