"""Surface/soil decoding without inferred producer preprocessing."""

from datetime import timedelta

import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_surface, SURFACE_VARIABLES
from mpas_init.surface import MISSING, snow_ocean_zero, soil_water_maximum
from test_grib import codes, message, rewrite


BOUNDS = ((0., .1), (.1, .4), (.4, 1.), (1., 2.))
PARAMETERS = {"landmask": (2, 0, 0, 81), "surface_altitude": (0, 3, 5, 7),
              "skin_temperature": (0, 0, 0, 11), "snow_water_equivalent": (0, 1, 13, 65),
              "snow_depth": (0, 1, 11, 66), "sea_ice_fraction": (10, 2, 0, 91),
              "soil_temperature": (2, 0, 2, 85), "soil_moisture": (2, 0, 192, 144)}
ZERO = {"surface_snow_amount.missing_ocean": "zero", "surface_snow_thickness.missing_ocean": "zero"}


def write_surface(codes, path, *, edition=2, masked=False, **scan):
    base, _, _, _ = message(codes, edition=edition, lead=21600, **scan)
    payloads, expected, mapping = [], {}, {}
    for name, parameters in PARAMETERS.items():
        layers = (2, 0, 3, 1) if name.startswith("soil_") else (None,)
        field = []
        for layer in layers:
            h = codes.codes_new_from_message(base)
            try:
                codes.codes_set(h, "centre", "kwbc")
                if edition == 1:
                    codes.codes_set(h, "table2Version", 2)
                    codes.codes_set(h, "indicatorOfParameter", parameters[3])
                    if name == "soil_moisture":
                        codes.codes_set(h, "centre", "ecmf")
                        codes.codes_set(h, "table2Version", 128)
                        codes.codes_set(h, "indicatorOfParameter", 39)
                else:
                    for key, value in zip(("discipline", "parameterCategory", "parameterNumber"), parameters[:3]):
                        codes.codes_set(h, key, value)
                codes.codes_set(h, "typeOfLevel", "depthBelowLandLayer" if layer is not None else "surface")
                if layer is not None:
                    if edition == 1:
                        codes.codes_set(h, "topLevel", round(BOUNDS[layer][0] * 100))
                        codes.codes_set(h, "bottomLevel", round(BOUNDS[layer][1] * 100))
                    else:
                        for side, depth in zip(("First", "Second"), BOUNDS[layer]):
                            codes.codes_set(h, "scaleFactorOf" + side + "FixedSurface", 2)
                            codes.codes_set(h, "scaledValueOf" + side + "FixedSurface", round(depth * 100))
                latitude = codes.codes_get_array(h, "latitudes")
                longitude = np.remainder(codes.codes_get_array(h, "longitudes"), 360.)
                ny, nx = np.unique(latitude).size, np.unique(longitude).size
                index = np.arange(nx * ny, dtype=float).reshape(ny, nx)
                values = ((index % 3) == 0).astype(float) if name == "landmask" else (
                    index * 8 if name == "surface_altitude" else index * .25 + 273 if name == "skin_temperature" else
                    index * .5 if name == "snow_water_equivalent" else index * .125 if name == "snow_depth" else
                    (index % 5) * .125 if name == "sea_ice_fraction" else
                    index * .25 + 250 + layer * 10 if name == "soil_temperature" else
                    index / 128 + .125 + layer / 16)
                if masked and name != "landmask":
                    values.flat[1] = MISSING
                raw = np.empty(nx * ny)
                order = np.lexsort((longitude, -latitude))
                raw[order] = values.reshape(-1)
                if masked and name != "landmask":
                    codes.codes_set(h, "bitmapPresent", 1)
                    codes.codes_set(h, "missingValue", 9999.)
                    raw[raw == MISSING] = 9999.
                codes.codes_set_values(h, raw)
                mapping[SURFACE_VARIABLES[name][0]] = codes.codes_get(h, "shortName")
                payloads.append(codes.codes_get_message(h))
                field.append((layer, values))
            finally:
                codes.codes_release(h)
        expected[name] = np.stack([v for _, v in sorted(field)]) if layer is not None else field[0][1]
    path.write_bytes(b"".join(payloads))
    return expected, mapping


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("scan", ({}, {"reverse_lat": True, "reverse_lon": True, "column_scan": True}, {"grid": "regular_gg"}))
def test_surface_exact_analytic_values_bounds_and_scans(tmp_path, codes, edition, scan):
    path = tmp_path / "surface.nc"
    expected, mapping = write_surface(codes, path, edition=edition, **scan)
    state = read_surface(path, mapping)
    assert state.lead == timedelta(hours=6)
    np.testing.assert_array_equal(state.soil_bounds, BOUNDS)
    assert state.field_grids is None and set(state.fields) == set(expected)
    for name, value in expected.items():
        assert state.fields[name].dtype == torch.float64
        np.testing.assert_array_equal(state.fields[name], value)


@pytest.mark.parametrize("edition", (1, 2))
def test_surface_bitmap_preserves_missing_and_valid_zeros(tmp_path, codes, edition):
    path = tmp_path / "masked.grib"
    expected, mapping = write_surface(codes, path, edition=edition, masked=True)
    state = read_surface(path, mapping)
    for name, value in expected.items():
        np.testing.assert_array_equal(state.fields[name], value)
    zero = read_surface(path, {**mapping, **ZERO})
    for name in ("snow_water_equivalent", "snow_depth"):
        assert state.fields[name].flatten()[1] == MISSING
        assert zero.fields[name].flatten()[1] == 0 and zero.fields[name].flatten()[0] == 0
    assert torch.equal(zero.fields["soil_moisture"], state.fields["soil_moisture"])


@pytest.mark.parametrize("defect", ("gap", "overlap", "negative_depth", "mismatched_layers", "duplicate", "nonbinary", "units", "wrong_units", "missing_land_snow", "masked_grid"))
def test_surface_invalid_metadata_and_masks_reject(tmp_path, codes, defect):
    path = tmp_path / "bad.grib2"
    _, mapping = write_surface(codes, path)
    def change(h):
        short = codes.codes_get(h, "shortName")
        if short == mapping["soil_temperature"]:
            top = codes.codes_get(h, "scaledValueOfFirstFixedSurface", int)
            if top == 10 and defect in ("gap", "overlap", "mismatched_layers"):
                codes.codes_set(h, "scaledValueOfFirstFixedSurface", 20 if defect != "overlap" else 5)
            if top == 0 and defect == "negative_depth":
                codes.codes_set(h, "scaledValueOfSecondFixedSurface", 0)
            if defect == "masked_grid":
                codes.codes_set(h, "longitudeOfFirstGridPointInDegrees", 1.)
                codes.codes_set(h, "longitudeOfLastGridPointInDegrees", 316.)
        if short == mapping["volume_fraction_of_condensed_water_in_soil"] and defect in ("gap", "overlap"):
            if codes.codes_get(h, "scaledValueOfFirstFixedSurface", int) == 10:
                codes.codes_set(h, "scaledValueOfFirstFixedSurface", 20 if defect == "gap" else 5)
        if short == mapping["volume_fraction_of_condensed_water_in_soil"] and defect == "negative_depth":
            if codes.codes_get(h, "scaledValueOfFirstFixedSurface", int) == 0:
                codes.codes_set(h, "scaledValueOfSecondFixedSurface", 0)
        if short == mapping["land_binary_mask"] and defect == "nonbinary":
            values = codes.codes_get_values(h); values[0] = .5; codes.codes_set_values(h, values)
        if short == mapping["surface_snow_amount"] and defect == "missing_land_snow":
            codes.codes_set(h, "bitmapPresent", 1); codes.codes_set(h, "missingValue", 9999.)
            values = codes.codes_get_values(h); values[0] = 9999.; codes.codes_set_values(h, values)
        if short == mapping["surface_temperature"] and defect == "units":
            codes.codes_set(h, "parameterCategory", 1); codes.codes_set(h, "parameterNumber", 13)
        if short == mapping['surface_snow_amount'] and defect == 'wrong_units':
            codes.codes_set(h, 'parameterCategory', 3); codes.codes_set(h, 'parameterNumber', 0)
    rewrite(codes, path, change)
    if defect == "duplicate": path.write_bytes(path.read_bytes() * 2)
    if defect == "units": mapping["surface_temperature"] = mapping["surface_snow_amount"]
    if defect == 'wrong_units': mapping['surface_snow_amount'] = 'sp'
    match = {"gap": "contiguous", "overlap": "contiguous", "negative_depth": "positive bounds", "mismatched_layers": "matching complete", "duplicate": "duplicate", "nonbinary": "binary", "units": "more than one", "wrong_units": "unsupported GRIB units", "missing_land_snow": "missing land", "masked_grid": "collocated"}[defect]
    with pytest.raises((ValueError, RuntimeError), match=match): read_surface(path, {**mapping, **ZERO})


@pytest.mark.parametrize("defect", ("shape", "land", "mask", "nonfinite"))
def test_snow_policy_guards(defect):
    values, land = torch.tensor([[MISSING,0.]]),torch.tensor([[0.,1.]])
    if defect=='shape': land=land.flatten()
    elif defect=='land': land[0,0]=1
    elif defect=='mask': land[0,0]=.5
    else: values[0,0]=float('nan')
    with pytest.raises((ValueError,RuntimeError)): snow_ocean_zero(values,land)


@pytest.mark.parametrize('maximum', (-1., 0., 1.01, float('nan'), float('inf'), 'invalid', True))
def test_invalid_soil_maximum_rejects(maximum):
    with pytest.raises(ValueError, match='soil maximum'): soil_water_maximum(torch.tensor([.2]), maximum)


@pytest.mark.parametrize('value', (-.01, 1.0003175048828126, float('nan'), float('inf')))
def test_unphysical_uncapped_soil_rejects(value):
    with pytest.raises(RuntimeError, match='soil volumetric water'): soil_water_maximum(torch.tensor([value]))


GFS_SNOW = {"surface_snow_amount.convention": "wps_ncep_gfs"}
DECLARED = {"surface_snow_amount.convention": {"declared": "wps_ncep_gfs", "source": None}}


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("zero", (False, True))
def test_declared_wps_ncep_gfs_snow_doubles_swe_and_derives_depth(tmp_path, codes, edition, zero):
    path = tmp_path / "masked.grib"
    expected, mapping = write_surface(codes, path, edition=edition, masked=True)
    undeclared = read_surface(path, mapping)
    assert undeclared.declarations is None
    for name, value in expected.items():
        np.testing.assert_array_equal(undeclared.fields[name], value)
    gfs = {key: value for key, value in mapping.items() if key != "surface_snow_thickness"}
    gfs.update(GFS_SNOW, **({"surface_snow_amount.missing_ocean": "zero"} if zero else {}))
    state = read_surface(path, gfs)
    assert state.declarations == DECLARED
    swe = torch.tensor(expected["snow_water_equivalent"])
    missing = swe == MISSING
    filled = torch.where(missing, 0. if zero else MISSING, swe * 2.)
    # rrpr.F order: double SWE, then SNOWH = 0.005 * SNOW; the missing-ocean fill precedes both.
    assert torch.equal(state.fields["snow_water_equivalent"], filled)
    assert torch.equal(state.fields["snow_depth"], torch.where(missing, filled, filled * .005))
    assert state.fields["snow_depth"].flatten()[1] == (0. if zero else MISSING)
    for name in set(expected) - {"snow_water_equivalent", "snow_depth"}:
        assert torch.equal(state.fields[name], undeclared.fields[name])


@pytest.mark.parametrize("change", ("mapped_depth", "depth_policy", "invalid"))
def test_declared_snow_convention_conflicts_reject(tmp_path, codes, change):
    path = tmp_path / "surface.grib"
    _, mapping = write_surface(codes, path)
    if change == "mapped_depth":
        gfs = {**mapping, **GFS_SNOW}
    else:
        gfs = {key: value for key, value in mapping.items() if key != "surface_snow_thickness"}
        gfs.update({"surface_snow_amount.convention": "doubled"} if change == "invalid"
                   else {**GFS_SNOW, "surface_snow_thickness.missing_ocean": "zero"})
    with pytest.raises(ValueError, match="surface_snow_amount.convention"):
        read_surface(path, gfs)
