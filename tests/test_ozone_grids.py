from dataclasses import replace
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_ozone
from mpas_init.tracers import initialize_ufs_tracers, periodic_bilinear
from test_forecast_selection import MEMBER, expand_forecast
from test_reduced import gather_cf
from test_source_collection import copy_variables, split_complete
from test_tracers import generated_inputs, write_source


def generated_grid():
    source, atmosphere, fields = generated_inputs()
    lat = torch.tensor(np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(8)[0][::-1].copy())))
    lon = torch.arange(24, dtype=torch.float64) * 15. + 3.75
    p, y, x = torch.meshgrid(source.pressure, lat, lon, indexing="ij")
    ozone = 1.e-7 * (2. + .15 * torch.log(p) + .001 * y + .2 * torch.sin(x * torch.pi / 180.))
    humidity = .003 + 2.e-8 * p + 1.e-6 * (y + 90.) + 1.e-4 * torch.cos(x * torch.pi / 180.)
    return replace(source, latitude=lat, longitude=lon, ozone=ozone, specific_humidity=humidity,
                   grid_type="rectilinear_latlon"), atmosphere, fields


def gather_source(source, counts):
    columns = [torch.arange(0, source.longitude.numel(), source.longitude.numel() // n) for n in counts]
    return replace(source, longitude=torch.cat([source.longitude[c] for c in columns]),
                   pressure=source.pressure if source.pressure.ndim == 1 else torch.cat(
                       [source.pressure[:, i, c] for i, c in enumerate(columns)], dim=-1),
                   ozone=torch.cat([source.ozone[:, i, c] for i, c in enumerate(columns)], dim=-1),
                   specific_humidity=None if source.specific_humidity is None else torch.cat(
                       [source.specific_humidity[:, i, c] for i, c in enumerate(columns)], dim=-1),
                   grid_type="reduced_gaussian", row_counts=torch.tensor(counts))


def test_cli_ozone_polar_policy_requires_source(tmp_path):
    command = [sys.executable,"-B","-m","mpas_init","--source","absent.nc","--static","absent.nc",
               "--ozone-polar-extension","clamp","--output",str(tmp_path/"wrong.nc")]
    result = subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode != 0 and "requires --ozone-source" in result.stderr
    assert not (tmp_path/"wrong.nc").exists()


@pytest.mark.parametrize("shuffle", [False, True])
def test_reduced_cf_split_selection_mapping_and_stored_precision(tmp_path, shuffle):
    source, _, _ = generated_grid()
    base, path = tmp_path / "base.nc", tmp_path / "gathered.nc"
    write_source(base, source)
    counts = [4, 8, 12, 24, 24, 12, 8, 4]
    gather_cf(base, path, counts, shuffle=shuffle)
    expected = gather_source(source, counts)
    for paths in (path, split_complete(path, tmp_path / "split")):
        actual = read_ozone(paths)
        assert actual.grid_type == "reduced_gaussian"
        for name in ("ozone", "specific_humidity", "latitude", "longitude", "pressure", "row_counts"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    forecast = tmp_path / "forecast.nc"
    expand_forecast(path, forecast, matrix=True, invalid_unselected=True)
    selected = read_ozone(forecast, valid_time=source.valid_time, reference_time=source.reference_time, member=MEMBER)
    assert selected.member == MEMBER
    torch.testing.assert_close(selected.ozone, expected.ozone, rtol=0, atol=0)
    with pytest.raises(ValueError):
        read_ozone(forecast, valid_time=source.valid_time, reference_time=source.reference_time, member=MEMBER-1)


@pytest.mark.parametrize("defect", ["counts", "count_sum", "count_type", "gap", "row_order", "regional", "latitude", "basis", "humidity", "policy"])
def test_invalid_reduced_ozone_rejects(defect):
    source, atmosphere, fields = generated_grid()
    source = gather_source(source, [4, 8, 12, 24, 24, 12, 8, 4])
    options = {}
    if defect == "counts":
        source = replace(source, row_counts=source.row_counts[:-1])
    elif defect == "count_sum":
        source.row_counts[0] += 1
    elif defect == "count_type":
        source = replace(source, row_counts=source.row_counts.double())
    elif defect == "gap":
        source.longitude[1] += .1
    elif defect == "row_order":
        source.longitude[:4] = source.longitude[:4].flip(0)
    elif defect == "regional":
        source = replace(source, latitude=source.latitude * .5)
    elif defect == "latitude":
        source.latitude[1] = source.latitude[0]
    elif defect == "basis":
        source = replace(source, grid_type="regular_latlon")
    elif defect == "humidity":
        source = replace(source, specific_humidity=source.specific_humidity[:, :-1])
    else:
        options["polar_extension"] = "extrapolate"
    with pytest.raises((ValueError, RuntimeError)):
        initialize_ufs_tracers(source, atmosphere, fields, **options)


def test_reduced_rounded_coordinates_use_actual_nodes():
    source, atmosphere, fields = generated_grid()
    counts = torch.tensor([7, 9, 13, 17, 17, 13, 9, 7])
    longitude = torch.cat([torch.arange(n, dtype=torch.float32) * (360. / n) for n in counts.tolist()])
    latitude = torch.repeat_interleave(source.latitude, counts)
    p, y = source.pressure[:, None], latitude[None]
    ozone = 1.e-7 * (2. + .15 * torch.log(p) + .001 * y + .2 * torch.sin(longitude.double()[None] * torch.pi / 180.))
    source = replace(source, longitude=longitude, ozone=ozone, specific_humidity=torch.full_like(ozone, .01),
                     row_counts=counts, grid_type="reduced_gaussian")
    # Exact stored nodes remain exact, including rounded longitudes next to nominal cell boundaries.
    for index in (8, 15, 20, 37):
        value = periodic_bilinear(ozone, source.latitude, longitude, latitude[index:index+1], longitude[index:index+1].double(), row_counts=counts)
        torch.testing.assert_close(value[:, 0], ozone[:, index], rtol=0, atol=0)
    assert torch.isfinite(initialize_ufs_tracers(source, atmosphere, fields)["o3mr"]).all()


def test_float32_reduced_cf_coordinates_preserve_storage_precision(tmp_path):
    source, atmosphere, fields = generated_grid()
    counts = [7, 9, 13, 17, 17, 13, 9, 7]
    longitude = np.concatenate([np.arange(n, dtype="f4") * np.float32(360./n) for n in counts])
    latitude = np.repeat(source.latitude.numpy(), counts)
    p = source.pressure.numpy()[:, None]
    ozone = 1.e-7 * (2. + .15*np.log(p) + .001*latitude + .2*np.sin(longitude.astype('f8')*np.pi/180.))
    base, path = tmp_path / "base.nc", tmp_path / "rounded.nc"
    write_source(base, source)
    copy_variables(base, path, {"pressure", "time", "forecast_reference_time", "forecast_period"})
    with Dataset(path, "a") as ds:
        ds.createDimension("row", len(counts))
        ds.createDimension("width", max(counts))
        ds.createDimension("point", sum(counts))
        index = ds.createVariable("point", "i8", ("point",))
        index.compress, index[:] = "row width", np.concatenate([i*max(counts)+np.arange(n) for i,n in enumerate(counts)])
        for name, values, units, dtype in (("latitude", latitude, "degrees_north", "f8"),
                                          ("longitude", longitude, "degrees_east", "f4")):
            var = ds.createVariable(name, dtype, ("point",))
            var[:], var.units, var.standard_name = values, units, name
        for name, values in (("mass_fraction_of_ozone_in_air", ozone), ("specific_humidity", np.full_like(ozone,.01))):
            var = ds.createVariable(name, "f8", ("time", "pressure", "point"))
            var[0], var.units, var.standard_name, var.coordinates = values, "kg kg-1", name, "latitude longitude"
    decoded = read_ozone(path)
    assert decoded.longitude.dtype == torch.float32 and decoded.ozone.dtype == torch.float64
    np.testing.assert_array_equal(decoded.longitude.numpy(), longitude)
    assert torch.isfinite(initialize_ufs_tracers(decoded, atmosphere, fields)["o3mr"]).all()


@pytest.mark.parametrize("defect", ["index", "compress", "units", "gap", "missing", "mean", "humidity_coordinates", "clock"])
def test_invalid_cf_ozone_layout_rejects(tmp_path, defect):
    source = generated_grid()[0]
    base, path = tmp_path / "base.nc", tmp_path / "gathered.nc"
    write_source(base, source)
    gather_cf(base, path, [4,8,12,24,24,12,8,4])
    with Dataset(path, "a") as ds:
        if defect == "index":
            ds["rgrid"][1] = ds["rgrid"][0]
        elif defect == "compress":
            ds["rgrid"].compress = "missing latitude"
        elif defect == "units":
            ds["longitude"].units = "m"
        elif defect == "gap":
            ds["longitude"][1] += .1
        elif defect == "missing":
            ds["specific_humidity"][0,0,0] = np.nan
        elif defect == "mean":
            ds["specific_humidity"].cell_methods = "time: mean"
        elif defect == "humidity_coordinates":
            var = ds.createVariable("other_latitude", "f8", ("rgrid",))
            var[:], var.units, var.standard_name = ds["latitude"][:] + .1, "degrees_north", "latitude"
            ds["specific_humidity"].coordinates = "other_latitude longitude"
        else:
            ds["forecast_period"].assignValue(3600.)
    with pytest.raises(ValueError):
        read_ozone(path)
