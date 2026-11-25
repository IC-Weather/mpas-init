"""Supplied CF geopotential is a representation, not a producer-specific path."""


from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_atmosphere
from mpas_init.surface import MISSING
from test_atmosphere import generated_source
from test_below_ground import masked_source
from test_gaussian import make_cf_gaussian
from test_multigrid import prepare_multigrid
from test_forecast_selection import MEMBER, expand_forecast
from test_reduced import gather_cf
from test_source_collection import split_complete


def encode_geopotential(path, *, units="m2 s-2", keep_height=False):
    """Offline encoding of an existing physical height; no initializer supplies truth."""
    with Dataset(path, "a") as ds:
        height = next(var for var in ds.variables.values()
                      if getattr(var, "standard_name", "") == "geopotential_height")
        phi = ds.createVariable("phi", "f8", height.dimensions)
        phi[:] = height[:] * 9.80665
        phi.units, phi.standard_name = units, "geopotential"
        if "coordinates" in height.ncattrs():
            phi.coordinates = height.coordinates
        if not keep_height:
            height.delncattr("standard_name")


def assert_decoded(actual, expected):
    assert set(actual.fields) == set(expected.fields)
    assert (actual.valid_time, actual.reference_time, actual.lead, actual.member) == (
        expected.valid_time, expected.reference_time, expected.lead, expected.member)
    for name in ("pressure", "latitude", "longitude"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    for name, value in expected.fields.items():
        torch.testing.assert_close(actual.fields[name], value,
                                   rtol=torch.finfo(value.dtype).eps if name == "geopotential_height" else 0, atol=0)


@pytest.mark.parametrize("representation", ("3d", "a", "ap"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("units", ("m2 s-2", "m**2 s**-2"))
def test_supplied_geopotential_full_levels(tmp_path, representation, reverse, units):
    path = tmp_path / "source.nc"
    generated_source(path, representation, reverse=reverse)
    expected = read_atmosphere(path, include_relative_humidity=True)
    encode_geopotential(path, units=units)
    actual = read_atmosphere(path, include_relative_humidity=True)
    assert_decoded(actual, expected)
    assert actual.field_grids is None


@pytest.mark.parametrize("layout", ("gaussian", "reduced", "multigrid", "masked"))
def test_geopotential_grids_masks_and_split_files(tmp_path, layout):
    path, original = tmp_path / "source.nc", tmp_path / "original.nc"
    if layout == "masked":
        masked_source(path, reverse=True)
    elif layout == "multigrid":
        generated_source(original, "ap")
        prepare_multigrid(original, path, reverse=True)
    elif layout == "reduced":
        make_cf_gaussian(original, "3d")
        gather_cf(original, path, [4, 4, 8, 8, 4, 4], shuffle=True)
    else:
        make_cf_gaussian(path, "ap")
    expected = read_atmosphere(path, include_relative_humidity=True)
    encode_geopotential(path)
    actual = read_atmosphere(path, include_relative_humidity=True)
    assert_decoded(actual, expected)
    if layout == "masked":
        assert (actual.fields["geopotential_height"] == MISSING).any()
        assert set(actual.field_surface_pressure) == set(expected.field_surface_pressure)
        for name, value in expected.field_surface_pressure.items():
            torch.testing.assert_close(actual.field_surface_pressure[name], value, rtol=0, atol=0)
    if layout == "reduced":
        torch.testing.assert_close(actual.row_counts, expected.row_counts, rtol=0, atol=0)
    if layout == "multigrid":
        assert set(actual.field_grids) == set(expected.field_grids)
        for name, grid in expected.field_grids.items():
            for coordinate in ("latitude", "longitude"):
                torch.testing.assert_close(getattr(actual.field_grids[name], coordinate),
                                           getattr(grid, coordinate), rtol=0, atol=0)
    split = split_complete(path, tmp_path / "split")
    assert_decoded(read_atmosphere(split[::-1], include_relative_humidity=True), actual)


def test_geopotential_explicit_mapping_resolves_quantity_ambiguity(tmp_path):
    path = tmp_path / "source.nc"
    generated_source(path, "ap")
    expected = read_atmosphere(path)
    encode_geopotential(path, keep_height=True)
    with pytest.raises(ValueError, match="conflicting.*geopotential"):
        read_atmosphere(path)
    for quantity, variable in (("geopotential", "phi"), ("geopotential_height", "geopotential_height")):
        assert_decoded(read_atmosphere(path, {quantity: variable}), expected)
    with pytest.raises(ValueError, match="conflicting.*geopotential"):
        read_atmosphere(path, {"geopotential": "phi", "geopotential_height": "geopotential_height"})
    with Dataset(path, "a") as ds:
        ds["geopotential_height"].delncattr("standard_name")
        ds["phi"].delncattr("standard_name")
    with pytest.raises(ValueError, match="geopotential_height"):
        read_atmosphere(path)
    assert_decoded(read_atmosphere(path, {"geopotential": "phi"}), expected)


@pytest.mark.parametrize("supplied_geopotential", (False, True))
def test_surface_geopotential_is_not_an_atmospheric_height_candidate(tmp_path, supplied_geopotential):
    path = tmp_path / "source.nc"
    generated_source(path, "ap")
    expected = read_atmosphere(path)
    if supplied_geopotential:
        encode_geopotential(path)
    with Dataset(path, "a") as ds:
        var = ds.createVariable("surface_phi", "f8", ("time", "latitude", "longitude"))
        var[:], var.standard_name, var.units = 1234., "geopotential", "m2 s-2"
    assert_decoded(read_atmosphere(path), expected)


def test_geopotential_mapping_retains_mask_and_anchor_coordinates(tmp_path):
    path = tmp_path / "source.nc"
    masked_source(path)
    expected = read_atmosphere(path)
    encode_geopotential(path)
    mapping = {"geopotential.latitude": "latitude", "geopotential.longitude": "longitude"}
    with Dataset(path, "a") as ds:
        for coordinate in ("latitude", "longitude"):
            ds[coordinate].delncattr("standard_name")
        ds["ps"].delncattr("standard_name")
    # Other fields still use their existing global coordinate/pressure mappings.
    mapping.update(latitude="latitude", longitude="longitude")
    mapping.update({("geopotential" if name == "geopotential_height" else name) + ".surface_air_pressure": "ps"
                    for name in expected.fields})
    actual = read_atmosphere(path, mapping)
    assert_decoded(actual, expected)
    for name, value in expected.field_surface_pressure.items():
        torch.testing.assert_close(actual.field_surface_pressure[name], value, rtol=0, atol=0)


def test_geopotential_selected_forecast_ignores_invalid_unselected_records(tmp_path):
    path, forecast = tmp_path / "source.nc", tmp_path / "forecast.nc"
    generated_source(path, "ap", reverse=True)
    encode_geopotential(path)
    expected = read_atmosphere(path, include_relative_humidity=True)
    expand_forecast(path, forecast, matrix=True, invalid_unselected=True)
    selection = dict(valid_time=expected.valid_time, reference_time=expected.reference_time, member=MEMBER)
    actual = read_atmosphere(forecast, include_relative_humidity=True, **selection)
    assert actual.member == MEMBER
    for name in expected.fields:
        torch.testing.assert_close(actual.fields[name], expected.fields[name], rtol=0, atol=0)
    parts = split_complete(forecast, tmp_path / "split")
    assert_decoded(read_atmosphere(parts, include_relative_humidity=True, **selection), actual)


@pytest.mark.parametrize("defect,error", (("units", "units"), ("altitude", "geopotential_height"),
                                         ("mean", "point samples"), ("missing", "missing"),
                                         ("duplicate", "found 2"), ("reconstruction", "conflicts")))
def test_invalid_or_ambiguous_geopotential_rejects(tmp_path, defect, error):
    path = tmp_path / "source.nc"
    generated_source(path, "ap")
    encode_geopotential(path)
    with Dataset(path, "a") as ds:
        if defect == "units":
            ds["phi"].units = "m"
        elif defect == "altitude":
            ds["phi"].standard_name, ds["phi"].units = "altitude", "m"
        elif defect == "mean":
            ds["phi"].cell_methods = "time: mean"
        elif defect == "missing":
            ds["phi"][0, 0, 0, -1] = np.ma.masked
        elif defect == "duplicate":
            var = ds.createVariable("phi2", "f8", ds["phi"].dimensions)
            var[:], var.standard_name, var.units = ds["phi"][:], "geopotential", "m2 s-2"
    with pytest.raises(ValueError, match=error):
        read_atmosphere(path, hybrid_convention="ifs-vapour-only" if defect == "reconstruction" else None)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_geopotential_conversion_forward_reverse(tmp_path, dtype):
    from mpas_init.vertical import geopotential_to_height

    phi = torch.tensor([-1961.33, 0., 49033.25, 294199.5], dtype=dtype, requires_grad=True)
    direction = phi.new_tensor([5., -3., 11., 29.])
    weights = phi.new_tensor([.25, -.5, .75, 1.25])
    value, jvp = torch.func.jvp(geopotential_to_height, (phi,), (direction,))
    gradient, = torch.autograd.grad((value * weights).sum(), phi)
    torch.testing.assert_close(value, phi / 9.80665, rtol=0, atol=0)
    torch.testing.assert_close(jvp, direction / 9.80665, rtol=0, atol=0)
    torch.testing.assert_close(gradient, weights / 9.80665, rtol=0, atol=0)
    np.savez(tmp_path / "conversion-ad.npz", geopotential=phi.detach().numpy(), value=value.detach().numpy(),
             direction=direction.numpy(), weights=weights.numpy(), jvp=jvp.detach().numpy(), vjp=gradient.numpy())
    for wrong in (phi / 9.80616, phi):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(value, wrong, rtol=2.e-6, atol=2.e-8)
