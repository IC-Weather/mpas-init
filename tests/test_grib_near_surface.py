"""Real ecCodes 2m/10m decoding and unchanged complete-file acceptance."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_near_surface
from grib_fixture import LIQUID_WATER, NAMES
from test_grib import codes, message, rewrite
from test_humidity_phase import scalar_phase


HEIGHTS = {"air_temperature": 2, "relative_humidity": 2, "eastward_wind": 10, "northward_wind": 10}


def write_near(codes, path, *, omit=None, **settings):
    expected, payloads = {}, []
    for name, height in HEIGHTS.items():
        payload, values, latitude, longitude = message(codes, short_name=NAMES[name],
                                                       typeOfLevel="heightAboveGround", level=height, **settings)
        if name != omit:
            payloads.append(payload)
        expected[name] = values
    path.write_bytes(b"".join(payloads))
    return expected, latitude, longitude


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("scan", ({}, {"reverse_lat": True}, {"reverse_lon": True}, {"column_scan": True},
                                  {"origin": 180.}, {"reverse_lat": True, "reverse_lon": True, "column_scan": True}))
def test_near_scans_exact_values_and_positive_forecast(tmp_path, codes, edition, scan):
    path = tmp_path / "not-grib.nc"
    expected, latitude, longitude = write_near(codes, path, edition=edition, lead=21600, **scan)
    state = read_near_surface(path, LIQUID_WATER)
    assert state.valid_time == datetime(2024, 5, 21, 6, tzinfo=timezone.utc)
    assert state.lead == timedelta(hours=6) and state.member is None
    assert state.field_grids is None
    np.testing.assert_array_equal(state.latitude, latitude)
    np.testing.assert_array_equal(state.longitude, longitude)
    assert set(state.fields) == set(HEIGHTS)
    for name, value in expected.items():
        assert state.fields[name].dtype == torch.float64
        np.testing.assert_array_equal(state.fields[name], value)


@pytest.mark.parametrize("edition", (1, 2))
def test_near_gaussian_and_independent_field_grids(tmp_path, codes, edition):
    path = tmp_path / "gaussian.grib"
    expected, latitude, _ = write_near(codes, path, edition=edition, grid="regular_gg")

    def shift(h):
        if codes.codes_get(h, "cfName") == "relative_humidity":
            codes.codes_set(h, "longitudeOfFirstGridPointInDegrees", 1.)
            codes.codes_set(h, "longitudeOfLastGridPointInDegrees", 316.)

    rewrite(codes, path, shift)
    state = read_near_surface(path, LIQUID_WATER)
    assert state.grid_type == "rectilinear_latlon"
    np.testing.assert_allclose(state.latitude, latitude, rtol=0, atol=3.e-14)
    with path.open("rb") as stream:
        h = codes.codes_grib_new_from_file(stream)
        try:
            np.testing.assert_array_equal(state.latitude, np.unique(codes.codes_get_array(h, "latitudes"))[::-1])
        finally:
            codes.codes_release(h)
    assert set(state.field_grids) == {"relative_humidity"}
    np.testing.assert_array_equal(state.field_grids["relative_humidity"].longitude, np.arange(8) * 45. + 1.)
    for name, value in expected.items():
        np.testing.assert_array_equal(state.fields[name], value)
    with pytest.raises(ValueError, match="collocated"):
        read_near_surface(path, {"relative_humidity.saturation_reference": "wps_ice_blend"})


def test_near_phase_and_unrelated_isobaric_surface_or_height_records(tmp_path, codes):
    path = tmp_path / "combined.grib"
    expected, _, _ = write_near(codes, path)
    payloads = [path.read_bytes()]
    for name in HEIGHTS:
        payloads += [message(codes, short_name=NAMES[name])[0],
                     message(codes, short_name=NAMES[name], typeOfLevel="heightAboveGround", level=50)[0],
                     message(codes, short_name=NAMES[name], typeOfLevel="surface", level=0)[0]]
    path.write_bytes(b"".join(payloads))
    # Exercise all three thermodynamic branches with analytic independent values.
    def cold(h):
        if codes.codes_get(h, "cfName") == "air_temperature" and codes.codes_get(h, "typeOfLevel") == "heightAboveGround" and codes.codes_get(h, "level") == 2:
            codes.codes_set_values(h, np.resize([233., 260., 273.15, 280.], 40))
    rewrite(codes, path, cold)
    state = read_near_surface(path, {"relative_humidity.saturation_reference": "wps_ice_blend"})
    target = scalar_phase(np.resize([233., 260., 273.15, 280.], (5, 8)), expected["relative_humidity"])
    torch.testing.assert_close(state.fields["relative_humidity"], torch.tensor(target), rtol=64 * torch.finfo(torch.float64).eps, atol=1.e-15)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(state.fields["relative_humidity"], expected["relative_humidity"])


@pytest.mark.parametrize("defect", ("missing", "wrong_height", "fractional_height", "layer", "duplicate", "mask", "regional", "wind", "average", "units", "ambiguous"))
def test_near_invalid_metadata_and_support_reject(tmp_path, codes, defect):
    path = tmp_path / "bad.grib"
    write_near(codes, path, omit="relative_humidity" if defect == "missing" else None)

    def change(h):
        name = codes.codes_get(h, "cfName")
        if name == "air_temperature":
            if defect == "wrong_height":
                codes.codes_set(h, "level", 3)
            elif defect == "fractional_height":
                codes.codes_set(h, "scaleFactorOfFirstFixedSurface", 1)
                codes.codes_set(h, "scaledValueOfFirstFixedSurface", 21)
            elif defect == "layer":
                codes.codes_set(h, "typeOfLevel", "heightAboveGroundLayer")
            elif defect == "mask":
                codes.codes_set(h, "bitmapPresent", 1)
                codes.codes_set(h, "missingValue", 9999.)
                values = codes.codes_get_values(h)
                values[0] = 9999.
                codes.codes_set_values(h, values)
            elif defect == "regional":
                codes.codes_set(h, "longitudeOfLastGridPointInDegrees", 280.)
                codes.codes_set(h, "iDirectionIncrementInDegrees", 40.)
            elif defect == "average":
                codes.codes_set(h, "stepType", "avg")
                codes.codes_set(h, "endStep", 6)
            elif defect in ("units", "ambiguous"):
                codes.codes_set(h, "shortName", "2r")
        if defect == "wind" and name == "eastward_wind":
            codes.codes_set(h, "uvRelativeToGrid", 1)
    rewrite(codes, path, change)
    mapping = {**LIQUID_WATER, "air_temperature": "2r"} if defect in ("units", "ambiguous") else LIQUID_WATER
    if defect == "units":
        mapping = {**mapping, "relative_humidity": "r"}  # Do not select this temperature record twice.
    if defect == "duplicate":
        path.write_bytes(path.read_bytes() * 2)
    match = {"missing": "all four", "wrong_height": "all four", "fractional_height": "all four", "layer": "all four", "duplicate": "duplicate",
             "mask": "missing GRIB", "regional": "longitude", "wind": "earth-relative", "average": "instantaneous",
             "units": "units", "ambiguous": "more than one"}[defect]
    with pytest.raises(ValueError, match=match):
        read_near_surface(path, mapping)


def test_near_multi_field_messages_reject(tmp_path, codes):
    path = tmp_path / "multi.grib2"
    multi = codes.codes_grib_multi_new()
    try:
        for name, height in HEIGHTS.items():
            h = codes.codes_new_from_message(message(codes, short_name=NAMES[name], typeOfLevel="heightAboveGround", level=height)[0])
            try:
                codes.codes_grib_multi_append(h, 4, multi)
            finally:
                codes.codes_release(h)
        with path.open("wb") as stream:
            codes.codes_grib_multi_write(multi, stream)
    finally:
        codes.codes_grib_multi_release(multi)
    with pytest.raises(ValueError, match="multi-field"):
        read_near_surface(path, LIQUID_WATER)
    valid = tmp_path / "after-rejection.grib2"
    expected, _, _ = write_near(codes, valid)
    state = read_near_surface(valid, LIQUID_WATER)
    for name, value in expected.items():
        np.testing.assert_array_equal(state.fields[name], value)
