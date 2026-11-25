"""Actual GRIB bitmaps share the qualified isobaric missing-value path."""


import numpy as np
import pytest
import torch

from mpas_init.horizontal import MISSING
from mpas_init.netcdf import read_atmosphere, read_pressure_levels
from test_grib import codes, message, rewrite
from grib_fixture import LIQUID_WATER, NAMES


def masked_messages(codes, path, *, edition=2, grid="regular_ll", geopotential=False, **scan):
    payloads, expected = [], {}
    levels = (100000., 50000., 100.)
    def stored(value, longitude):
        rows = np.arange(value.shape[0])[::-1] if scan.get("reverse_lat") else np.arange(value.shape[0])
        columns = np.argsort(np.remainder(longitude - scan.get("origin", 0.), 360.))
        if scan.get("reverse_lon"):
            columns = columns[::-1]
        ordered = value[np.ix_(rows, columns)]
        return (ordered.T if scan.get("column_scan") else ordered).reshape(-1)
    for name, short_name in NAMES.items():
        expected[name] = []
        for level in levels:
            payload, value, latitude, longitude = message(codes, edition=edition, grid=grid, pressure=level,
                                                         short_name="z" if geopotential and short_name == "gh" else short_name,
                                                         lead=21600, **scan)
            handle = codes.codes_new_from_message(payload)
            try:
                missing = np.zeros(value.shape, dtype=bool)
                missing[1, 2] = level == levels[0]
                raw = codes.codes_get_values(handle)
                codes.codes_set(handle, "bitmapPresent", 1)
                codes.codes_set(handle, "missingValue", 9999.)
                codes.codes_set_values(handle, np.where(stored(missing, longitude), 9999., raw))
                payloads.append(codes.codes_get_message(handle))
            finally:
                codes.codes_release(handle)
            value = value.copy()
            if level == levels[0]:
                value[1, 2] = MISSING
            if name == "geopotential_height" and geopotential:
                value = np.where(value == MISSING, MISSING, value / 9.80665)
            expected[name].append(value)
    payload, _, _, _ = message(codes, edition=edition, grid=grid, short_name="sp", pressure=100000.,
                                typeOfLevel="surface", level=0., lead=21600, **scan)
    handle = codes.codes_new_from_message(payload)
    ps = np.full((latitude.size, longitude.size), 105000.)
    ps[1, 2] = 90000.
    try:
        codes.codes_set_values(handle, stored(ps, longitude))
        payloads.append(codes.codes_get_message(handle))
    finally:
        codes.codes_release(handle)
    path.write_bytes(b"".join(payloads))
    return {name: np.stack(value) for name, value in expected.items()}, ps, payloads


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("grid", ("regular_ll", "regular_gg"))
@pytest.mark.parametrize("scan", ({}, {"reverse_lat": True, "reverse_lon": True, "column_scan": True}))
def test_isobaric_bitmaps_preserve_all_fields_and_support(tmp_path, codes, edition, grid, scan):
    path = tmp_path / "masked.grib"
    expected, ps, payloads = masked_messages(codes, path, edition=edition, grid=grid, **scan)
    source = read_atmosphere(path, LIQUID_WATER, include_relative_humidity=True)
    assert set(source.field_surface_pressure) == set(expected)
    for name, value in expected.items():
        np.testing.assert_array_equal(source.fields[name], value)
        np.testing.assert_array_equal(source.field_surface_pressure[name], ps)
    with pytest.raises(ValueError, match="missing atmospheric"):
        read_pressure_levels(path)
    # The same catalog must work when pressure lives in a separate file.
    path.write_bytes(b"".join(payloads[:-1]))
    surface = tmp_path / "pressure.grib"
    surface.write_bytes(payloads[-1])
    split = read_atmosphere([surface, path], LIQUID_WATER, include_relative_humidity=True)
    for name in expected:
        torch.testing.assert_close(split.fields[name], source.fields[name], rtol=0, atol=0)
        torch.testing.assert_close(split.field_surface_pressure[name], source.field_surface_pressure[name], rtol=0, atol=0)


def multigrid_messages(codes, tmp_path, *, edition=2, shifted=False):
    fields, pressures, payloads = {}, {}, []
    for wind in (False, True):
        grid = "regular_gg" if wind and not shifted else "regular_ll"
        selected = {"eastward_wind", "northward_wind"} if wind else set(NAMES) - {
            "eastward_wind", "northward_wind"}
        expected, ps, messages = masked_messages(codes, tmp_path / f"grid-{wind}.grib", edition=edition, grid=grid)
        fields.update({name: expected[name] for name in selected})
        pressures.update({name: ps for name in selected})
        for payload in messages:
            handle = codes.codes_new_from_message(payload)
            try:
                if codes.codes_get(handle, "shortName") in {NAMES[name] for name in selected} | {"sp"}:
                    if shifted and wind:
                        codes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", 1.)
                        codes.codes_set(handle, "longitudeOfLastGridPointInDegrees", 316.)
                    payloads.append(codes.codes_get_message(handle))
            finally:
                codes.codes_release(handle)
    return fields, pressures, payloads


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("split", (False, True))
@pytest.mark.parametrize("shifted", (False, True))
def test_masked_fields_select_their_collocated_pressure_grid(tmp_path, codes, edition, split, shifted):
    expected, pressures, payloads = multigrid_messages(codes, tmp_path, edition=edition, shifted=shifted)
    if split:
        paths = []
        for index, payload in enumerate(reversed(payloads)):
            path = tmp_path / f"message-{index}.grib"
            path.write_bytes(payload)
            paths.append(path)
    else:
        path = tmp_path / "multigrid.grib"
        path.write_bytes(b"".join(payloads))
        paths = path
    source = read_atmosphere(paths, LIQUID_WATER, include_relative_humidity=True)
    assert set(source.field_grids) == {"eastward_wind", "northward_wind"}
    assert set(source.field_surface_pressure) == set(expected)
    if shifted:
        np.testing.assert_array_equal(source.field_grids["eastward_wind"].longitude, np.arange(8) * 45. + 1.)
    for name, value in expected.items():
        np.testing.assert_array_equal(source.fields[name], value)
        np.testing.assert_array_equal(source.field_surface_pressure[name], pressures[name])
    with pytest.raises(ValueError, match="missing atmospheric"):
        read_pressure_levels(paths)


@pytest.mark.parametrize("grid", ("regular_ll", "regular_gg"))
@pytest.mark.parametrize("defect", ("absent", "duplicate", "rescanned_duplicate", "above_ground", "coverage", "masked"))
def test_multigrid_pressure_must_be_unique_complete_and_collocated(tmp_path, codes, grid, defect):
    _, _, payloads = multigrid_messages(codes, tmp_path)
    result = []
    for payload in payloads:
        handle = codes.codes_new_from_message(payload)
        try:
            selected = codes.codes_get(handle, "shortName") == "sp" and codes.codes_get(handle, "gridType") == grid
            if selected:
                if defect == "absent":
                    continue
                if defect == "duplicate":
                    result.append(payload)
                elif defect == "rescanned_duplicate":
                    _, _, rescanned = masked_messages(codes, tmp_path / "rescanned.grib", grid=grid,
                                                       reverse_lat=True, reverse_lon=True, column_scan=True)
                    result.append(rescanned[-1])
                else:
                    values = codes.codes_get_values(handle)
                    values[10] = {"above_ground": 105000., "coverage": 200., "masked": 9999.}[defect]
                    if defect == "masked":
                        codes.codes_set(handle, "bitmapPresent", 1)
                        codes.codes_set(handle, "missingValue", 9999.)
                    codes.codes_set_values(handle, values)
            result.append(codes.codes_get_message(handle))
        finally:
            codes.codes_release(handle)
    path = tmp_path / "invalid-multigrid.grib"
    path.write_bytes(b"".join(result))
    error = "duplicate GRIB" if "duplicate" in defect else "collocated|above-ground|missing GRIB"
    with pytest.raises(ValueError, match=error):
        read_atmosphere(path, LIQUID_WATER, include_relative_humidity=True)


def test_masked_geopotential_conversion_preserves_sentinel_and_rh(tmp_path, codes):
    path = tmp_path / "geopotential.grib"
    expected, _, _ = masked_messages(codes, path, geopotential=True)
    source = read_atmosphere(path, {**LIQUID_WATER, "geopotential": "z"}, include_relative_humidity=True)
    np.testing.assert_array_equal(source.fields["geopotential_height"], expected["geopotential_height"])
    assert "geopotential_height" in source.field_surface_pressure
    assert "geopotential" not in source.field_surface_pressure
    normalized = read_atmosphere(path, {"geopotential": "z", "relative_humidity.saturation_reference": "wps_ice_blend"},
                                 include_relative_humidity=True)
    assert torch.equal(source.fields["relative_humidity"] == MISSING, normalized.fields["relative_humidity"] == MISSING)


@pytest.mark.parametrize("defect", ("absent", "duplicate", "nonpositive", "masked", "above_ground", "coverage",
                                    "wrong_grid", "mean", "wrong_units", "wrong_time", "wrong_level"))
def test_invalid_grib_below_ground_support_rejects(tmp_path, codes, defect):
    path = tmp_path / "invalid.grib"
    _, _, payloads = masked_messages(codes, path)
    if defect == "absent":
        path.write_bytes(b"".join(payloads[:-1]))
    elif defect == "duplicate":
        path.write_bytes(b"".join(payloads + payloads[-1:]))
    else:
        def change(handle):
            if codes.codes_get(handle, "shortName") != "sp":
                return
            values = codes.codes_get_values(handle)
            if defect in ("nonpositive", "above_ground", "coverage"):
                values[10] = {"nonpositive": 0., "above_ground": 105000., "coverage": 200.}[defect]
                codes.codes_set_values(handle, values)
            elif defect == "masked":
                codes.codes_set(handle, "bitmapPresent", 1)
                codes.codes_set(handle, "missingValue", 9999.)
                values[10] = 9999.
                codes.codes_set_values(handle, values)
            elif defect == "wrong_grid":
                codes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", 1.)
                codes.codes_set(handle, "longitudeOfLastGridPointInDegrees", 316.)
            elif defect == "mean":
                codes.codes_set(handle, "stepType", "avg")
            elif defect == "wrong_units":
                codes.codes_set(handle, "shortName", "t")
            elif defect == "wrong_time":
                codes.codes_set(handle, "endStep", 7)
            else:
                codes.codes_set(handle, "typeOfLevel", "meanSea")
        rewrite(codes, path, change)
    mapping = {"surface_air_pressure": "t"} if defect == "wrong_units" else None
    with pytest.raises(ValueError):
        read_atmosphere(path, mapping)
