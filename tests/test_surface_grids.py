from dataclasses import replace

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.horizontal import MISSING
from mpas_init.netcdf import SURFACE_VARIABLES, read_surface
from mpas_init.surface import initialize_surface
from test_source_collection import split_complete
from test_surface import small_surface_inputs


MASKED = frozenset(("soil_temperature", "soil_moisture", "sea_ice_fraction"))


def generated_surface(path, *, reverse=False):
    """Small independent analytic fields, with a different coastal mask per field."""
    source, _ = small_surface_inputs()
    with Dataset(path, "w", clobber=False) as ds:
        for name, size in (("time", 1), ("soil_depth", 4), ("bounds", 2)):
            ds.createDimension(name, size)
        t = ds.createVariable("time", "f8", ("time",))
        t[:], t.units, t.standard_name = [21600.], "seconds since 2024-05-21 00:00:00", "time"
        for name, value, units, standard in (("reference", 0., t.units, "forecast_reference_time"),
                                              ("lead", 6., "hours", "forecast_period")):
            var = ds.createVariable(name, "f8")
            var.assignValue(value)
            var.units, var.standard_name = units, standard
        depth = ds.createVariable("soil_depth", "f8", ("soil_depth",))
        depth[:], depth.units, depth.standard_name = source.soil_bounds.mean(-1).numpy(), "m", "depth"
        depth.bounds, depth.positive = "soil_bounds", "down"
        ds.createVariable("soil_bounds", "f8", ("soil_depth", "bounds"))[:] = source.soil_bounds.numpy()
        for index, (name, (standard, units)) in enumerate(SURFACE_VARIABLES.items()):
            ny, nx = (7, 12) if name == "landmask" else ((5, 8) if index % 2 else (4, 6))
            latitude = np.linspace(90., -90., ny) if ny != 4 else np.array([80., 23., -17., -80.])
            longitude = np.arange(nx) * (360. / nx)
            ydim, xdim = name + "_lat", name + "_lon"
            for dim, value, coord, unit in ((ydim, latitude, "latitude", "degrees_north"),
                                             (xdim, longitude, "longitude", "degrees_east")):
                ds.createDimension(dim, value.size)
                var = ds.createVariable(dim, "f8", (dim,))
                var[:], var.standard_name, var.units = value, coord, unit
            yy, xx = np.meshgrid(latitude, longitude, indexing="ij")
            wave = np.sin(np.deg2rad(xx)) + np.deg2rad(yy) / 5.
            mask = (xx < (135. if name == "soil_temperature" else 225. if name == "soil_moisture" else 180.)).astype(float)
            if name == "landmask":
                value = mask
            elif name.startswith("soil_"):
                value = np.stack([(275. + k + wave) if name == "soil_temperature" else (.2 + k / 100 + wave / 100) for k in range(4)])
            else:
                value = {"surface_altitude": 200. + 10. * wave, "skin_temperature": 280. + wave,
                         "snow_water_equivalent": 20. + wave, "snow_depth": .2 + wave / 100,
                         "sea_ice_fraction": .1 + wave / 100}[name]
            dims = ("time", "soil_depth", ydim, xdim) if name.startswith("soil_") else ("time", ydim, xdim)
            var = ds.createVariable(standard, "f8", dims)
            var[:], var.standard_name, var.units = value[None], standard, units
            var.coordinates = ydim + " " + xdim
            if name in MASKED:
                coast = ds.createVariable("coast_" + name, "f8", ("time", ydim, xdim))
                coast[:], coast.standard_name, coast.units = mask[None], "land_binary_mask", "1"
                coast.coordinates = var.coordinates
        if reverse:
            for var in ds.variables.values():
                value = var[:]
                for axis, dim in enumerate(var.dimensions):
                    if dim == "soil_depth" or dim.endswith(("_lat", "_lon")):
                        value = np.flip(value, axis)
                if getattr(var, "standard_name", "") == "longitude":
                    value = value - 360.
                var[:] = value
    return {"land_binary_mask": "land_binary_mask"}


@pytest.mark.parametrize("reverse", [False, True])
def test_cf_surface_grids_masks_scans_and_split(tmp_path, reverse):
    original, path = tmp_path / "original.nc", tmp_path / "source.nc"
    mapping = generated_surface(original)
    generated_surface(path, reverse=reverse)
    actual, expected = read_surface(path, mapping), read_surface(original, mapping)
    assert set(actual.field_grids) == set(SURFACE_VARIABLES) - {"landmask"}
    assert set(actual.field_landmasks) == MASKED
    assert actual.field_grids["soil_temperature"].grid_type == "rectilinear_latlon"
    parts = split_complete(path, tmp_path / "split")
    split = read_surface(parts[::-1], mapping)
    for source in (actual, split):
        for name in expected.fields:
            torch.testing.assert_close(source.fields[name], expected.fields[name], rtol=0, atol=0)
        for name in MASKED:
            torch.testing.assert_close(source.field_landmasks[name], expected.field_landmasks[name], rtol=0, atol=0)
        for name in source.field_grids:
            for axis in ("latitude", "longitude"):
                torch.testing.assert_close(getattr(source.field_grids[name], axis), getattr(expected.field_grids[name], axis), rtol=0, atol=0)
    with pytest.raises(ValueError, match="land_binary_mask"):
        read_surface(path)  # Multiple primary masks require an explicit choice.


@pytest.mark.parametrize("defect", ["absent_mask", "ambiguous_mask", "mask_units", "fractional_mask", "masked_mask", "mask_mean",
                                    "mask_coordinates", "field_mean", "field_longitude", "soil_dimension"])
def test_cf_surface_grid_or_mask_errors_reject(tmp_path, defect):
    path = tmp_path / "source.nc"
    mapping = generated_surface(path)
    with Dataset(path, "a") as ds:
        coast = ds["coast_soil_temperature"]
        if defect == "absent_mask":
            coast.delncattr("standard_name")
        elif defect == "ambiguous_mask":
            var = ds.createVariable("another_coast", "f8", coast.dimensions)
            var[:], var.standard_name, var.units = coast[:], "land_binary_mask", "1"
        elif defect == "mask_units":
            coast.units = "%"
        elif defect == "fractional_mask":
            coast[0, 0, 0] = .5
        elif defect == "masked_mask":
            coast[0, 0, 0] = np.ma.masked
        elif defect == "mask_mean":
            coast.cell_methods = "time: mean"
        elif defect == "mask_coordinates":
            var = ds.createVariable("other_latitude", "f8", ("soil_temperature_lat",))
            var[:], var.standard_name, var.units = [80., 25., -15., -80.], "latitude", "degrees_north"
            coast.coordinates = "other_latitude soil_temperature_lon"
        elif defect == "field_mean":
            ds["surface_temperature"].cell_methods = "time: mean"
        elif defect == "field_longitude":
            ds["skin_temperature_lon"][:] *= .5
        else:
            ds.renameVariable("soil_temperature", "old_soil")
            ds["old_soil"].delncattr("standard_name")
            ds.createDimension("wrong_depth", 4)
            var = ds.createVariable("soil_temperature", "f8", ("time", "wrong_depth", "soil_temperature_lat", "soil_temperature_lon"))
            var[:], var.standard_name, var.units = ds["old_soil"][:], "soil_temperature", "K"
    with pytest.raises(ValueError):
        read_surface(path, mapping)


def test_cf_surface_renamed_coordinates_and_masks(tmp_path):
    path = tmp_path / "source.nc"
    mapping = generated_surface(path)
    expected = read_surface(path, mapping)
    with Dataset(path, "a") as ds:
        for name in ("landmask_lat", "landmask_lon", "soil_temperature_lat", "soil_temperature_lon"):
            ds[name].delncattr("standard_name")
        ds["coast_soil_temperature"].delncattr("standard_name")
    mapping.update({"land_binary_mask.latitude": "landmask_lat", "land_binary_mask.longitude": "landmask_lon",
                    "soil_temperature.latitude": "soil_temperature_lat", "soil_temperature.longitude": "soil_temperature_lon",
                    "soil_temperature.land_binary_mask": "coast_soil_temperature"})
    actual = read_surface(path, mapping)
    for name in actual.fields:
        torch.testing.assert_close(actual.fields[name], expected.fields[name], rtol=0, atol=0)
    torch.testing.assert_close(actual.field_landmasks["soil_temperature"], expected.field_landmasks["soil_temperature"], rtol=0, atol=0)


@pytest.mark.parametrize("defect", ["unknown_grid", "primary_grid", "unknown_mask", "missing_mask", "mask_shape", "fractional", "reduced", "missing_support"])
def test_surface_grid_tensor_api_rejects(tmp_path, defect):
    path = tmp_path / "source.nc"
    mapping = generated_surface(path)
    source = read_surface(path, mapping)
    grids, masks, fields = dict(source.field_grids), dict(source.field_landmasks), dict(source.fields)
    if defect in ("unknown_grid", "primary_grid"):
        grids["unknown" if defect == "unknown_grid" else "landmask"] = grids["soil_temperature"]
    elif defect == "unknown_mask":
        masks["skin_temperature"] = masks["soil_temperature"]
    elif defect == "missing_mask":
        masks.pop("soil_temperature")
    elif defect == "mask_shape":
        masks["soil_temperature"] = masks["soil_temperature"][:, :-1]
    elif defect == "fractional":
        masks["soil_temperature"] = torch.full_like(masks["soil_temperature"], .5)
    elif defect == "reduced":
        grids["soil_temperature"] = replace(grids["soil_temperature"], row_counts=torch.ones(4, dtype=torch.int64))
    else:
        fields["soil_temperature"] = torch.full_like(fields["soil_temperature"], MISSING)
    _, static = small_surface_inputs()
    with pytest.raises((RuntimeError, ValueError)):
        initialize_surface(replace(source, fields=fields, field_grids=grids, field_landmasks=masks), static)
