from dataclasses import replace
from datetime import datetime, timedelta, timezone

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import OzoneSource, PressureLevelSource, read_ozone
from mpas_init.tracers import UFS_TRACER_FIELDS, initialize_ufs_tracers, mpas_layer_pressure, periodic_bilinear, vertical_log_pressure
from test_vertical_grid import assert_stage


def generated_inputs():
    date = datetime(2024, 5, 21, tzinfo=timezone.utc)
    latitude, longitude = torch.linspace(-90., 90., 7, dtype=torch.float64), torch.arange(12, dtype=torch.float64) * 30
    pressure = torch.tensor([100., 1000., 5000., 20000., 60000., 105000.], dtype=torch.float64)
    p, lat, lon = torch.meshgrid(pressure, latitude, longitude, indexing="ij")
    ozone = 1.e-7 * (2 + .15 * torch.log(p) + .001 * lat + .2 * torch.sin(lon * torch.pi / 180))
    humidity = .003 + 2.e-8 * p + 1.e-6 * (lat + 90) + 1.e-4 * torch.cos(lon * torch.pi / 180)
    source = OzoneSource(pressure, latitude, longitude, ozone, "moist_air", humidity, date, date, timedelta(0))
    atmosphere = PressureLevelSource(pressure, latitude, longitude, {}, date, date, timedelta(0))
    rho = torch.tensor([[.9, .3, .02], [.95, .32, .021], [.85, .29, .019], [.92, .31, .018]], dtype=torch.float64)
    fields = {"rho": rho, "theta": torch.full_like(rho, 300.), "qv": torch.full_like(rho, .01),
              "latCell": torch.tensor([-1.2, -.4, .4, 1.2], dtype=torch.float64),
              "lonCell": torch.tensor([-2.4, -.6, .6, 2.4], dtype=torch.float64),
              **{name: torch.zeros_like(rho) for name in ("qc", "qr", "qi", "qs", "qg")}}
    return source, atmosphere, fields


def assert_tracers(actual, expected):
    assert set(actual) == set(expected) == UFS_TRACER_FIELDS
    for name in UFS_TRACER_FIELDS:
        if name == "o3mr":
            assert_stage(torch.as_tensor(actual[name]), torch.as_tensor(expected[name]), 256)
        else:
            np.testing.assert_array_equal(np.asarray(actual[name]), np.asarray(expected[name]), err_msg=name)


def test_analytic_bilinear_log_pressure_and_basis():
    source, atmosphere, fields = generated_inputs()
    lat, lon = source.latitude, source.longitude
    slab = lat[:, None] * .2 + lon[None, :] / 360
    queries_lat, queries_lon = torch.tensor([45., -45., 90., -90.]), torch.tensor([45., 315., -1., 359.])
    actual = periodic_bilinear(slab, lat, lon, queries_lat, queries_lon)
    torch.testing.assert_close(actual[:2], torch.tensor([9.125, -8.125], dtype=actual.dtype), rtol=0, atol=1.e-14)
    assert actual[2] - actual[3] == 36
    values = torch.log(source.pressure)[:, None].expand(-1, 4)
    targets = torch.tensor([[200., 2000., 40000.]] * 4, dtype=torch.float64)
    torch.testing.assert_close(vertical_log_pressure(source.pressure, values, targets), torch.log(targets), rtol=0, atol=2.e-15)
    moist = initialize_ufs_tracers(source, atmosphere, fields)
    dry = replace(source, ozone=source.ozone / (1. - source.specific_humidity), mass_basis="dry_air", specific_humidity=None)
    assert_tracers(moist, initialize_ufs_tracers(dry, atmosphere, fields))
    assert torch.count_nonzero(moist["o3mr"]) == moist["o3mr"].numel()
    assert torch.all(moist["sgs_tke"] == torch.tensor(1.e-4, dtype=torch.float32).double())
    for water_mass in (0., 1.):
        air_mass, ozone_mass = 10., 3.e-5
        sample = replace(source, ozone=torch.full_like(source.ozone, ozone_mass / air_mass),
                         specific_humidity=torch.full_like(source.specific_humidity, water_mass / air_mass))
        state = {**fields, "qv": torch.full_like(fields["qv"], water_mass / (air_mass - water_mass))}
        expected_ratio = torch.full_like(fields["qv"], ozone_mass / (air_mass - water_mass))
        assert_stage(initialize_ufs_tracers(sample, atmosphere, state)["o3mr"], expected_ratio, 64)


def test_local_forward_reverse_gradchecks():
    dtype = torch.float64
    rho, theta, qv = (x.requires_grad_() for x in (torch.tensor([[1., .1]], dtype=dtype),
                                                  torch.tensor([[290., 320.]], dtype=dtype),
                                                  torch.tensor([[.01, .001]], dtype=dtype)))
    assert torch.autograd.gradcheck(mpas_layer_pressure, (rho, theta, qv), check_forward_ad=True)
    lat, lon = torch.tensor([-90., 0., 90.], dtype=dtype), torch.tensor([0., 90., 180., 270.], dtype=dtype)
    field = torch.arange(12, dtype=dtype).reshape(3, 4).requires_grad_()
    target_lat, target_lon = torch.tensor([-45., 30.], dtype=dtype).requires_grad_(), torch.tensor([35., 320.], dtype=dtype).requires_grad_()
    assert torch.autograd.gradcheck(lambda f, a, b: periodic_bilinear(f, lat, lon, a, b),
                                    (field, target_lat, target_lon), check_forward_ad=True)
    p = torch.tensor([100., 1000., 10000.], dtype=dtype).requires_grad_()
    profiles = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype).requires_grad_()
    target = torch.tensor([[300., 3000.], [400., 4000.]], dtype=dtype).requires_grad_()
    assert torch.autograd.gradcheck(vertical_log_pressure, (p, profiles, target), check_forward_ad=True)


@pytest.mark.parametrize("defect", ["basis", "humidity", "bad_humidity", "negative", "nonfinite", "time", "partial_grid", "cloud"])
def test_invalid_tracer_sources_reject(defect):
    source, atmosphere, fields = generated_inputs()
    if defect == "basis":
        source = replace(source, mass_basis="unknown")
    elif defect == "humidity":
        source = replace(source, specific_humidity=None)
    elif defect == "bad_humidity":
        source.specific_humidity[1, 2, 3] = 1.
    elif defect == "negative":
        source.ozone[1, 2, 3] = -1.e-7
    elif defect == "nonfinite":
        source.ozone[1, 2, 3] = torch.nan
    elif defect == "time":
        source = replace(source, valid_time=source.valid_time + timedelta(hours=1))
    elif defect == "partial_grid":
        source = replace(source, longitude=source.longitude * .5)
    else:
        fields["qc"][1, 2] = .001
    with pytest.raises((ValueError, RuntimeError)):
        initialize_ufs_tracers(source, atmosphere, fields)


def write_source(path, source, *, dry=False):
    with Dataset(path, "w") as ds:
        for name, value, units, standard in (("pressure", source.pressure, "Pa", "air_pressure"),
                                            ("latitude", source.latitude, "degrees_north", "latitude"),
                                            ("longitude", source.longitude, "degrees_east", "longitude")):
            ds.createDimension(name, value.shape[0])
            var = ds.createVariable(name, "f8", (name,))
            var[:], var.units, var.standard_name = value.numpy(), units, standard
        ds.createDimension("time", 1)
        for name, value, units, dims in (("time", [source.valid_time.timestamp()], "seconds since 1970-01-01", ("time",)),
                                        ("forecast_reference_time", source.reference_time.timestamp(), "seconds since 1970-01-01", ()),
                                        ("forecast_period", source.lead.total_seconds(), "seconds", ())):
            var = ds.createVariable(name, "f8", dims)
            var[:], var.units, var.standard_name = value, units, name
        var = ds.createVariable("o3", "f8", ("time", "pressure", "latitude", "longitude"))
        var[0], var.units = source.ozone.numpy(), "kg kg-1"
        if dry:
            var.mpas_init_mass_basis = "dry_air"
        else:
            var.standard_name = "mass_fraction_of_ozone_in_air"
            q = ds.createVariable("specific_humidity", "f8", var.dimensions)
            q[0], q.units, q.standard_name = source.specific_humidity.numpy(), "kg kg-1", "specific_humidity"


def test_ozone_codec_basis_mapping_and_rejections(tmp_path):
    source, _, _ = generated_inputs()
    path = tmp_path / "source.nc"
    write_source(path, source)
    actual = read_ozone(path)
    assert actual.mass_basis == "moist_air"
    assert torch.equal(actual.ozone, source.ozone)
    assert torch.equal(actual.specific_humidity, source.specific_humidity)
    for name in ("o3", "specific_humidity"):
        with Dataset(path, "a") as ds:
            ds[name].cell_methods = "time: mean"
        with pytest.raises(ValueError, match="instantaneous point"):
            read_ozone(path)
        with Dataset(path, "a") as ds:
            ds[name].delncattr("cell_methods")
    with Dataset(path, "a") as ds:
        ds["o3"].mpas_init_mass_basis = "dry_air"
    with pytest.raises(ValueError, match="mass basis"):
        read_ozone(path)
    with pytest.raises(ValueError, match="quantity mappings"):
        read_ozone(path, {"ozone_mass_mixing_ratio": "o3", "mass_fraction_of_ozone_in_air": "o3"})
    with Dataset(path, "a") as ds:
        ds["o3"].delncattr("standard_name")
    actual = read_ozone(path, {"ozone_mass_mixing_ratio": "o3"})
    assert actual.mass_basis == "dry_air" and actual.specific_humidity is None
    with Dataset(path, "a") as ds:
        ds["o3"].units = "mol mol-1"
    with pytest.raises(ValueError, match="mass units"):
        read_ozone(path, {"ozone_mass_mixing_ratio": "o3"})
    with Dataset(path, "a") as ds:
        ds["o3"].units = "kg kg-1"
        ds["o3"].delncattr("mpas_init_mass_basis")
    with pytest.raises(ValueError, match="mass basis"):
        read_ozone(path, {"ozone_mass_mixing_ratio": "o3"})
    missing = tmp_path / "missing.nc"
    write_source(missing, source)
    with Dataset(missing, "a") as ds:
        ds["specific_humidity"].delncattr("standard_name")
    with pytest.raises(ValueError, match="specific_humidity"):
        read_ozone(missing)
