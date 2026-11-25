from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import PressureLevelSource, UfsSurfaceSource, UFS_SURFACE_VARIABLES, read_ufs_surface
from mpas_init.ufs_surface import initialize_ufs_surface


QUANTITIES = {
    "sfcr": "aerodynamic_roughness_length", "alvsf": "visible_strong_cosine_albedo",
    "alnsf": "near_ir_strong_cosine_albedo", "alvwf": "visible_weak_cosine_albedo",
    "alnwf": "near_ir_weak_cosine_albedo", "cnwat": "canopy_water_amount",
    "f10m": "wind_speed_ratio_10m_to_lowest_model_level", "ffhh": "surface_heat_transfer_function",
    "ffmm": "surface_momentum_transfer_function", "fricv": "friction_velocity",
    "icetk": "sea_ice_thickness", "spfh2m": "specific_humidity_2m", "tmp2m": "air_temperature_2m",
    "tg3": "deep_soil_temperature", "tisfc": "sea_ice_surface_temperature",
}


def generated_inputs():
    date = datetime(2024, 5, 21, tzinfo=timezone.utc)
    lat = torch.tensor(np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(8)[0])))
    lon = torch.arange(12, dtype=torch.float64) * 30.
    pattern = lat[:, None] / 90. + torch.sin(lon[None] * (torch.pi / 180.))
    fields = {}
    for name, units in UFS_SURFACE_VARIABLES.items():
        scale = 280. if units == "K" else .01 if name == "specific_humidity_2m" else .2
        fields[name] = scale * (1. + .1 * pattern)
    source = UfsSurfaceSource(lat, lon, fields, date, date, timedelta(0))
    atmosphere = PressureLevelSource(torch.tensor([100., 100000.]), lat, lon, {}, date, date, timedelta(0))
    targets = {"latCell": torch.tensor([-70., -20., 25., 70.], dtype=torch.float64) * (torch.pi / 180.),
               "lonCell": torch.tensor([-7., 48., 139., 302.], dtype=torch.float64) * (torch.pi / 180.)}
    native = {"landmask": torch.tensor([1., 1., 0., 1.]), "isice_lu": torch.tensor(15)}
    surface = {"ivgtyp": torch.tensor([2, 15, 17, 3])}
    return source, atmosphere, targets, native, surface


@pytest.mark.parametrize("defect", ["inventory", "shape", "nan", "albedo", "humidity", "roughness", "time", "regional", "seam", "mask", "target", "policy"])
def test_invalid_analyzed_surface_rejects(defect):
    source, atmosphere, target, native, surface = generated_inputs()
    options = {}
    if defect == "inventory":
        source.fields.pop("friction_velocity")
    elif defect == "shape":
        source.fields["friction_velocity"] = source.fields["friction_velocity"][:, :-1]
    elif defect in ("nan", "albedo", "humidity", "roughness"):
        name, value = {"nan": ("friction_velocity", torch.nan), "albedo": ("visible_strong_cosine_albedo", 1.1),
                       "humidity": ("specific_humidity_2m", 1.), "roughness": ("aerodynamic_roughness_length", 0.)}[defect]
        source.fields[name][0, 0] = value
    elif defect == "time":
        source = replace(source, reference_time=source.reference_time - timedelta(hours=1))
    elif defect == "regional":
        source = replace(source, latitude=source.latitude * .5)
    elif defect == "seam":
        source = replace(source, longitude=source.longitude * (360. / 330.))
    elif defect == "mask":
        native["landmask"][0] = .5
    elif defect == "target":
        target["latCell"][0] = 92. * torch.pi / 180.
    else:
        options["polar_extension"] = "extrapolate"
    with pytest.raises((ValueError, RuntimeError)):
        initialize_ufs_surface(source, atmosphere, target, native, surface, **options)


def write_source(path, source):
    with Dataset(path, "w") as ds:
        for name, value, units in (("latitude", source.latitude, "degrees_north"), ("longitude", source.longitude, "degrees_east")):
            ds.createDimension(name, value.shape[0])
            var = ds.createVariable(name, "f8", (name,))
            var[:], var.units, var.standard_name = value.numpy(), units, name
        ds.createDimension("time", 1)
        for name, value, units, dims in (("time", [source.valid_time.timestamp()], "seconds since 1970-01-01", ("time",)),
                                        ("forecast_reference_time", source.reference_time.timestamp(), "seconds since 1970-01-01", ()),
                                        ("forecast_period", source.lead.total_seconds(), "seconds", ())):
            var = ds.createVariable(name, "f8", dims)
            var[:], var.units, var.standard_name = value, units, name
        height = ds.createVariable("height_2m", "f8")
        height.assignValue(2.)
        height.standard_name, height.units, height.positive = "height", "m", "up"
        for name, values in source.fields.items():
            grid = (source.field_grids or {}).get(name)
            ydim, xdim = "latitude", "longitude"
            if grid is not None:
                ydim, xdim = name + "_lat", name + "_lon"
                for dim, data, standard, units in ((ydim, grid.latitude, "latitude", "degrees_north"),
                                                  (xdim, grid.longitude, "longitude", "degrees_east")):
                    ds.createDimension(dim, data.numel())
                    axis = ds.createVariable(dim, "f8", (dim,))
                    axis[:], axis.standard_name, axis.units = data.numpy(), standard, units
            dimensions = ("time", ydim, xdim) if grid is None else (xdim, "time", ydim)
            var = ds.createVariable(name, "f8", dimensions)
            var[:] = values.numpy()[None] if grid is None else values.numpy().T[:, None]
            var.units, var.mpas_init_quantity = UFS_SURFACE_VARIABLES[name], name
            var.coordinates = f"{ydim} {xdim}"
            if name in ("air_temperature_2m", "specific_humidity_2m"):
                var.coordinates += " height_2m"


def test_codec_explicit_quantities_mapping_and_rejections(tmp_path):
    source = generated_inputs()[0]
    path = tmp_path / "source.nc"
    write_source(path, source)
    decoded = read_ufs_surface(path)
    for name in source.fields:
        torch.testing.assert_close(decoded.fields[name], source.fields[name], rtol=0, atol=0)
    mapping = {}
    with Dataset(path, "a") as ds:
        for index, name in enumerate(source.fields):
            alias = f"field_{index}"
            ds.renameVariable(name, alias)
            ds[alias].delncattr("mpas_init_quantity")
            mapping[name] = alias
    with pytest.raises(ValueError):
        read_ufs_surface(path)
    renamed = read_ufs_surface(path, mapping)
    for name in source.fields:
        torch.testing.assert_close(renamed.fields[name], source.fields[name], rtol=0, atol=0)
    for attribute, wrong in (("units", "%"), ("mpas_init_quantity", "sea_ice_thickness"), ("cell_methods", "time: mean")):
        bad = tmp_path / (attribute + ".nc")
        shutil.copyfile(path, bad)
        with Dataset(bad, "a") as ds:
            ds[mapping["visible_strong_cosine_albedo"]].setncattr(attribute, wrong)
        with pytest.raises(ValueError):
            read_ufs_surface(bad, mapping)

    for defect in ("height", "calendar", "lead", "missing"):
        bad = tmp_path / (defect + ".nc")
        shutil.copyfile(path, bad)
        with Dataset(bad, "a") as ds:
            if defect == "height":
                ds["height_2m"].assignValue(10.)
            elif defect == "calendar":
                ds["time"].calendar = "julian"
            elif defect == "lead":
                ds["forecast_period"].assignValue(3600.)
            else:
                ds[mapping["friction_velocity"]][0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            read_ufs_surface(bad, mapping)


ROOT = Path(__file__).resolve().parents[1]
GFS_MAPPING = ROOT / "examples/gfs/ufs-surface.json"
# Unit labels exactly as the public GFS sfcanl.nc writes them; values below are fractions/SI like that file.
FV3_LABELS = {"sfcr": "m", "alvsf": "%", "alnsf": "%", "alvwf": "%", "alnwf": "%", "cnwat": "XXX", "f10m": "N/A",
              "ffhh": "XXX", "ffmm": "XXX", "fricv": "XXX", "icetk": "XXX", "spfh2m": "kg/kg", "tmp2m": "K",
              "tg3": "K", "tisfc": "K"}


def write_fv3_source(path, source, *, calendar="JULIAN", units="hours since 2024-05-21 00:00:00", value=0.):
    """Small synthetic file with the FV3 Gaussian sfcanl.nc metadata layout, not its data."""
    with Dataset(path, "w") as ds:
        for name, data, label in (("grid_xt", source.longitude, "degrees_E"), ("grid_yt", source.latitude.flip(0), "degrees_N")):
            ds.createDimension(name, data.numel())
            var = ds.createVariable(name, "f8", (name,))
            var[:], var.units, var.cartesian_axis = data.numpy(), label, name[-2].upper()
        ds.createDimension("time", 1)
        time = ds.createVariable("time", "f8", ("time",))
        time[:], time.units, time.calendar_type, time.calendar = value, units, calendar, calendar
        for name, quantity in QUANTITIES.items():
            var = ds.createVariable(name, "f4", ("time", "grid_yt", "grid_xt"))
            var[0] = source.fields[quantity].flip(0).numpy()
            var.units, var.missing, var.cell_methods = FV3_LABELS[name], np.float32(9.99e20), "time: point"


def gfs_mapping(**changes):
    mapping = json.loads(GFS_MAPPING.read_text())
    for key, value in changes.items():
        key = key.replace("__", ".")
        if value is None:
            mapping.pop(key)
        else:
            mapping[key] = value
    return mapping


def test_declared_fv3_conventions_decode_and_record(tmp_path):
    source = generated_inputs()[0]
    path = tmp_path / "sfcanl.nc"
    write_fv3_source(path, source)
    decoded = read_ufs_surface(path, gfs_mapping())
    assert (decoded.valid_time, decoded.reference_time, decoded.lead) == (source.valid_time, source.valid_time, timedelta(0))
    torch.testing.assert_close(decoded.latitude, source.latitude.flip(0), rtol=0, atol=0)
    for name, value in source.fields.items():
        torch.testing.assert_close(decoded.fields[name], value.flip(0).float().double(), rtol=0, atol=0)
    declarations = decoded.declarations
    assert declarations["time.calendar"] == {"declared": "standard", "source": "JULIAN"}
    assert declarations["visible_strong_cosine_albedo.units"] == {"declared": "1", "source": "%"}
    assert declarations["canopy_water_amount.units"] == {"declared": "kg m-2", "source": "XXX"}
    assert declarations["specific_humidity_2m.units"] == {"declared": "kg kg-1", "source": "kg/kg"}
    assert declarations["air_temperature_2m.height"] == {"declared": "2", "source": None}
    # Labels that already conform need no declaration and record none.
    assert {key for key in declarations if key.endswith(".units")} == {
        quantity + ".units" for name, quantity in QUANTITIES.items() if FV3_LABELS[name] != UFS_SURFACE_VARIABLES[quantity]}
    assert len(declarations) == 1 + 11 + 2


@pytest.mark.parametrize("key", ["time__calendar", "visible_strong_cosine_albedo__units", "friction_velocity__units",
                                 "specific_humidity_2m__units", "air_temperature_2m__height", "forecast_reference_time",
                                 "latitude"])
def test_undeclared_fv3_conventions_still_reject(tmp_path, key):
    path = tmp_path / "sfcanl.nc"
    write_fv3_source(path, generated_inputs()[0])
    read_ufs_surface(path, gfs_mapping())
    with pytest.raises(ValueError):
        read_ufs_surface(path, gfs_mapping(**{key: None}))


@pytest.mark.parametrize("changes", [
    {"visible_strong_cosine_albedo__units": "%"}, {"friction_velocity__units": "cm s-1"},
    {"aerodynamic_roughness_length__units": "cm"}, {"deep_soil_temperature__units": "degC"},
    {"time__calendar": "julian"}, {"time__calendar": "noleap"},
    {"air_temperature_2m__height": "10"}, {"specific_humidity_2m__height": "two"}])
def test_conflicting_declarations_reject(tmp_path, changes):
    path = tmp_path / "sfcanl.nc"
    write_fv3_source(path, generated_inputs()[0])
    with pytest.raises(ValueError, match="declared"):
        read_ufs_surface(path, gfs_mapping(**changes))
