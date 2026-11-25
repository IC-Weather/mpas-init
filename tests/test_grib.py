"""Real ecCodes execution and independent values; no mocked GRIB decoder."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import importlib.util

import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_atmosphere, read_pressure_levels
from grib_fixture import LIQUID_WATER, NAMES


@pytest.fixture(scope="module")
def codes():
    if importlib.util.find_spec("eccodes") is None:
        pytest.skip("optional ecCodes Python dependency is absent")
    try:
        import eccodes
    except RuntimeError as exc:
        if "Cannot find the ecCodes library" not in str(exc):
            raise
        pytest.skip("optional ecCodes native library is unavailable")
    return eccodes


def message(codes, **settings):
    payload, values, latitude, longitude = _message(codes, **settings)
    return payload, values.copy(), latitude.copy(), longitude.copy()


@lru_cache(maxsize=256)
def _message(codes, *, edition=2, short_name="t", pressure=100000, lead=0, member=None,
            reverse_lat=False, reverse_lon=False, column_scan=False, alternate_scan=False,
            origin=0., grid="regular_ll", **changes):
    handle = codes.codes_grib_new_from_samples(f"{grid}_pl_grib{edition}")
    latitude, longitude = np.linspace(90., -90., 5), np.arange(8) * 45.
    if grid == "regular_gg":
        latitude = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(4)[0]))[::-1]
    values = (np.arange(latitude.size * 8).reshape(-1, 8) * .25 + pressure / 1024.)
    if short_name == "q":
        values /= 65536.
    elif short_name in ("gh", "z"):
        values += (100000. - pressure) / 8.
    lat_order = np.arange(latitude.size)[::-1] if reverse_lat else np.arange(latitude.size)
    lon_order = np.argsort(np.remainder(longitude - origin, 360.))
    if reverse_lon:
        lon_order = lon_order[::-1]
    raw_lon = np.remainder(longitude[lon_order], 360.)
    try:
        settings = {"Ni": 8, "Nj": latitude.size,
                    "latitudeOfFirstGridPointInDegrees": float(latitude[lat_order[0]]),
                    "latitudeOfLastGridPointInDegrees": float(latitude[lat_order[-1]]),
                    "longitudeOfFirstGridPointInDegrees": float(raw_lon[0]),
                    "longitudeOfLastGridPointInDegrees": float(raw_lon[-1]),
                    "iDirectionIncrementInDegrees": 45., "jScansPositively": int(reverse_lat),
                    "iScansNegatively": int(reverse_lon), "jPointsAreConsecutive": int(column_scan),
                    "dataDate": 20240521, "dataTime": 0,
                    "shortName": short_name, "typeOfLevel": "isobaricInhPa", "level": pressure / 100.,
                    "stepUnits": "s", "endStep": lead}
        if grid == "regular_ll":
            settings["jDirectionIncrementInDegrees"] = 45.
        else:
            settings["N"] = 2
        if member is not None:
            settings.update(productDefinitionTemplateNumber=1, perturbationNumber=member,
                            numberOfForecastsInEnsemble=10, typeOfEnsembleForecast=3)
        if edition == 2:
            settings.update(packingType="grid_ieee", precision=2, alternativeRowScanning=int(alternate_scan))
        else:
            settings.update(packingType="grid_simple", bitsPerValue=24)
        settings.update(changes)
        for key, value in settings.items():
            codes.codes_set(handle, key, value)
        stored = values[np.ix_(lat_order, lon_order)]
        raw = (stored.T if column_scan else stored).copy()
        if alternate_scan:
            raw[1::2] = raw[1::2, ::-1]
        codes.codes_set_values(handle, raw.reshape(-1))
        return codes.codes_get_message(handle), values, latitude, longitude
    finally:
        codes.codes_release(handle)


@pytest.mark.parametrize("edition", (1, 2))
def test_fixture_cache_returns_independent_values(codes, edition):
    settings = {"edition": edition, "short_name": "q", "reverse_lat": True, "origin": 180.}
    expected = _message.__wrapped__(codes, **settings)
    actual = message(codes, **settings)
    assert actual[0] == expected[0]
    for value, truth in zip(actual[1:], expected[1:]):
        np.testing.assert_array_equal(value, truth)
        value[:] = -999.
    repeated = message(codes, **settings)
    assert repeated[0] == expected[0]
    for value, truth in zip(repeated[1:], expected[1:]):
        np.testing.assert_array_equal(value, truth)
    assert _message.cache_info().maxsize == 256


def write_source(codes, path, *, omit=None, **settings):
    expected, payloads = {}, []
    for field, short_name in NAMES.items():
        expected[field] = []
        for pressure in (100000., 50000., 100.):
            payload, values, latitude, longitude = message(codes, short_name=short_name, pressure=pressure, **settings)
            expected[field].append(values)
            if omit != (field, pressure):
                payloads.append(payload)
    path.write_bytes(b"".join(payloads))
    return {name: np.stack(values) for name, values in expected.items()}, latitude, longitude


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("scan", ({}, {"reverse_lat": True}, {"reverse_lon": True}, {"column_scan": True},
                                  {"origin": 180.}, {"reverse_lon": True, "reverse_lat": True, "column_scan": True}))
def test_actual_codec_roundtrip_all_fields_scans_and_forecast(tmp_path, codes, edition, scan):
    path = tmp_path / "not-an-extension.nc"
    expected, latitude, longitude = write_source(codes, path, edition=edition, lead=21600, **scan)
    source = read_atmosphere(path, LIQUID_WATER, include_relative_humidity=True)
    assert set(source.fields) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(source.fields[name], values)
        assert source.fields[name].dtype == torch.float64
    np.testing.assert_array_equal(source.latitude, latitude)
    np.testing.assert_array_equal(source.longitude, longitude)
    np.testing.assert_array_equal(source.pressure, [100000., 50000., 100.])
    assert source.reference_time == datetime(2024, 5, 21, tzinfo=timezone.utc)
    assert source.lead == timedelta(hours=6) and source.valid_time == source.reference_time + source.lead
    assert source.member is None
    other = read_pressure_levels(path)
    assert set(other.fields) == set(expected) - {"relative_humidity"}


def test_regular_gaussian_coordinates_against_independent_roots(tmp_path, codes):
    path = tmp_path / "gaussian.grib2"
    expected, latitude, longitude = write_source(codes, path, grid="regular_gg")
    source = read_atmosphere(path)
    np.testing.assert_allclose(source.latitude, latitude, rtol=0, atol=3.e-14)
    np.testing.assert_array_equal(source.longitude, longitude)
    assert source.grid_type == "rectilinear_latlon"
    for name, value in source.fields.items():
        np.testing.assert_array_equal(value, expected[name])


@pytest.mark.parametrize("grid", ("regular_ll", "regular_gg"))
@pytest.mark.parametrize("column_scan", (False, True))
@pytest.mark.parametrize("reverse_lat,reverse_lon", ((False, False), (True, False), (False, True), (True, True)))
def test_rectangular_alternating_scan_uses_encoded_geometry(tmp_path, codes, grid, column_scan, reverse_lat, reverse_lon):
    path = tmp_path / "alternating.grib2"
    expected, latitude, longitude = write_source(codes, path, grid=grid, column_scan=column_scan,
                                                reverse_lat=reverse_lat, reverse_lon=reverse_lon,
                                                alternate_scan=True, origin=180.)
    source = read_atmosphere(path, LIQUID_WATER, include_relative_humidity=True)
    np.testing.assert_allclose(source.latitude, latitude, rtol=0, atol=3.e-14)
    np.testing.assert_array_equal(source.longitude, longitude)
    for name, value in expected.items():
        np.testing.assert_array_equal(source.fields[name], value)


def rewrite(codes, path, change):
    messages = []
    with path.open("rb") as stream:
        while (handle := codes.codes_grib_new_from_file(stream)) is not None:
            try:
                change(handle)
                messages.append(codes.codes_get_message(handle))
            finally:
                codes.codes_release(handle)
    path.write_bytes(b"".join(messages))


def test_pa_levels_geopotential_and_separate_field_grid(tmp_path, codes):
    path = tmp_path / "explicit.grib2"
    expected, _, _ = write_source(codes, path)

    def change(handle):
        level = codes.codes_get(handle, "level")
        codes.codes_set(handle, "typeOfLevel", "isobaricInPa")
        codes.codes_set(handle, "level", level * 100.)
        name = codes.codes_get(handle, "shortName")
        if name == "gh":
            codes.codes_set(handle, "shortName", "z")
        elif name == "t":
            codes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", 1.)
            codes.codes_set(handle, "longitudeOfLastGridPointInDegrees", 316.)

    rewrite(codes, path, change)
    source = read_atmosphere(path, {"geopotential": "z"})
    np.testing.assert_array_equal(source.pressure, [100000., 50000., 100.])
    np.testing.assert_array_equal(source.fields["geopotential_height"], expected["geopotential_height"] / 9.80665)
    np.testing.assert_array_equal(source.fields["air_temperature"], expected["air_temperature"])
    assert set(source.field_grids) == {"air_temperature"}
    np.testing.assert_array_equal(source.field_grids["air_temperature"].longitude, np.arange(8) * 45. + 1.)


def test_repeated_valid_time_requires_cycle_and_multi_field_rejects(tmp_path, codes):
    first, second = tmp_path / "first.grib", tmp_path / "second.grib"
    write_source(codes, first, lead=21600)
    write_source(codes, second, lead=10800, dataTime=300)
    valid = datetime(2024, 5, 21, 6, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="exactly one"):
        read_atmosphere([first, second], valid_time=valid)
    selected = read_atmosphere([first, second], valid_time=valid, reference_time=valid - timedelta(hours=3))
    assert selected.reference_time.hour == 3 and selected.lead == timedelta(hours=3)
    multi = codes.codes_grib_multi_new()
    try:
        for short_name in ("t", "q"):
            handle = codes.codes_new_from_message(message(codes, short_name=short_name)[0])
            codes.codes_grib_multi_append(handle, 4, multi)
            codes.codes_release(handle)
        with second.open("wb") as stream:
            codes.codes_grib_multi_write(multi, stream)
    finally:
        codes.codes_grib_multi_release(multi)
    with pytest.raises(ValueError, match="multi-field"):
        read_atmosphere(second)


@pytest.mark.parametrize("defect", ("duplicate", "missing_level", "missing_field", "wrong_units", "grid_wind", "mask",
                                   "regional", "grid_changes", "two_heights", "averaged", "hybrid", "truncated"))
def test_incomplete_or_ambiguous_sources_reject(tmp_path, codes, defect):
    path = tmp_path / "bad.grib2"
    write_source(codes, path, omit=("air_temperature", 50000.) if defect == "missing_level" else None)
    messages = []
    with path.open("rb") as stream:
        while (handle := codes.codes_grib_new_from_file(stream)) is not None:
            try:
                name = codes.codes_get(handle, "shortName")
                if defect == "missing_field" and name == "t":
                    continue
                if name == "t":
                    if defect == "wrong_units":
                        codes.codes_set(handle, "shortName", "r")
                    elif defect == "mask":
                        codes.codes_set(handle, "bitmapPresent", 1)
                        codes.codes_set(handle, "missingValue", 9999.)
                        values = codes.codes_get_values(handle)
                        values[0] = 9999.
                        codes.codes_set_values(handle, values)
                    elif defect == "regional":
                        codes.codes_set(handle, "longitudeOfLastGridPointInDegrees", 280.)
                        codes.codes_set(handle, "iDirectionIncrementInDegrees", 40.)
                    elif defect == "grid_changes" and codes.codes_get(handle, "level") == 500:
                        codes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", 1.)
                        codes.codes_set(handle, "longitudeOfLastGridPointInDegrees", 316.)
                    elif defect == "averaged":
                        codes.codes_set(handle, "stepType", "avg")
                        codes.codes_set(handle, "endStep", 6)
                    elif defect == "hybrid":
                        codes.codes_set(handle, "typeOfLevel", "hybrid")
                if defect == "grid_wind" and name in ("u", "v"):
                    codes.codes_set(handle, "uvRelativeToGrid", 1)
                messages.append(codes.codes_get_message(handle))
            finally:
                codes.codes_release(handle)
    if defect == "duplicate":
        messages.append(messages[0])
    if defect == "two_heights":
        messages.append(message(codes, short_name="z")[0])
    payload = b"".join(messages)
    path.write_bytes(payload[:-100] if defect == "truncated" else payload)
    mapping = {"air_temperature": "r"} if defect == "wrong_units" else None
    errors = {"duplicate": "duplicate GRIB", "missing_level": "complete pressure levels",
              "missing_field": "complete pressure levels", "wrong_units": "units",
              "grid_wind": "earth-relative", "mask": "missing GRIB", "regional": "longitude",
              "grid_changes": "changes between", "two_heights": "exactly one supplied height",
              "averaged": "instantaneous", "hybrid": "complete pressure levels"}
    if defect == "truncated":
        with pytest.raises(codes.CodesInternalError):
            read_atmosphere(path, mapping)
    else:
        with pytest.raises(ValueError, match=errors[defect]):
            read_atmosphere(path, mapping)
