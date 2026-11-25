"""Reduced GRIB transport into the existing atmospheric/tracer tensor path."""

from datetime import timedelta

import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_atmosphere, read_ozone
from test_grib import codes, message as rectangular_message, rewrite
from grib_fixture import LIQUID_WATER, NAMES


def message(codes, values, counts, *, edition=2, short_name="t", pressure=100000., **settings):
    latitude = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(len(counts))[0]))[::-1]
    handle = codes.codes_grib_new_from_samples(f"reduced_gg_pl_32_grib{edition}")
    try:
        options = {"N": len(counts)//2, "Nj": len(counts),
                   "latitudeOfFirstGridPointInDegrees": float(latitude[0]),
                   "latitudeOfLastGridPointInDegrees": float(latitude[-1]),
                   "longitudeOfFirstGridPointInDegrees": 0.,
                   "longitudeOfLastGridPointInDegrees": 360. - 360./max(counts),
                   "dataDate": 20240521, "dataTime": 0, "shortName": short_name,
                   "typeOfLevel": "isobaricInhPa", "level": pressure/100.,
                   "stepUnits": "s", "endStep": 0}
        options.update({"packingType": "grid_ieee", "precision": 2} if edition == 2
                       else {"packingType": "grid_simple", "bitsPerValue": 24})
        options.update(settings)
        for key, value in options.items():
            codes.codes_set(handle, key, value)
        codes.codes_set_array(handle, "pl", counts)
        codes.codes_set_values(handle, np.asarray(values).reshape(-1))
        return codes.codes_get_message(handle)
    finally:
        codes.codes_release(handle)


def source(codes, path, *, edition=2, counts=(4, 8, 12, 12, 8, 4), lead=0, rectangular_wind=False):
    payloads, expected = [], {}
    for name, short_name in NAMES.items():
        expected[name] = []
        for pressure in (100000., 50000., 100.):
            values = np.arange(sum(counts))*.25 + pressure/1024.
            if short_name == "q":
                values /= 65536.
            if short_name == "gh":
                values += (100000.-pressure)/8.
            if rectangular_wind and short_name == "u":
                payload, values, _, _ = rectangular_message(codes, edition=edition, short_name=short_name,
                                                            pressure=pressure, lead=lead)
            else:
                payload = message(codes, values, counts, edition=edition, short_name=short_name,
                                  pressure=pressure, endStep=lead)
            expected[name].append(values)
            payloads.append(payload)
    path.write_bytes(b"".join(payloads))
    return {name: np.stack(values) for name, values in expected.items()}


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("rectangular_wind", (False, True))
def test_all_levels_values_coordinates_and_mixed_grids(tmp_path, codes, edition, rectangular_wind):
    path = tmp_path/"reduced.grib"
    expected = source(codes, path, edition=edition, lead=21600, rectangular_wind=rectangular_wind)
    actual = read_atmosphere(path, LIQUID_WATER, include_relative_humidity=True)
    counts = np.array([4, 8, 12, 12, 8, 4])
    latitude = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(6)[0]))[::-1]
    assert actual.grid_type == "reduced_gaussian" and actual.lead == timedelta(hours=6)
    np.testing.assert_allclose(actual.latitude, latitude, rtol=0, atol=3.e-14)
    np.testing.assert_array_equal(actual.row_counts, counts)
    np.testing.assert_array_equal(actual.longitude, np.concatenate([np.arange(n)*360./n for n in counts]))
    np.testing.assert_array_equal(actual.pressure, [100000., 50000., 100.])
    assert set(actual.field_grids or {}) == ({"eastward_wind"} if rectangular_wind else set())
    for name, values in expected.items():
        np.testing.assert_array_equal(actual.fields[name], values)
        assert actual.fields[name].dtype == torch.float64
    if rectangular_wind:
        assert actual.field_grids["eastward_wind"].row_counts is None
    first, second = tmp_path/"first.grib", tmp_path/"second.grib"
    data = path.read_bytes()
    with path.open("rb") as stream:
        h = codes.codes_grib_new_from_file(stream)
        length = codes.codes_get(h, "totalLength", int)
        codes.codes_release(h)
    first.write_bytes(data[:length])
    second.write_bytes(data[length:])
    split = read_atmosphere([first, second], LIQUID_WATER, include_relative_humidity=True)
    for name, values in actual.fields.items():
        torch.testing.assert_close(split.fields[name], values, rtol=0, atol=0)


@pytest.mark.parametrize("defect", ("regional", "short_row", "reverse_i", "reverse_j", "column_scan", "alternate_rows", "mask"))
def test_unsupported_or_incomplete_reduced_encoding_rejects(tmp_path, codes, defect):
    path = tmp_path/"invalid.grib2"
    counts = [3, 8, 12, 12, 8, 4] if defect == "short_row" else [4, 8, 12, 12, 8, 4]
    source(codes, path, counts=counts)
    def change(h):
        if codes.codes_get(h, "shortName") != "t":
            return
        keys = {"regional": ("longitudeOfLastGridPointInDegrees", 270.),
                "reverse_i": ("iScansNegatively", 1), "reverse_j": ("jScansPositively", 1),
                "column_scan": ("jPointsAreConsecutive", 1), "alternate_rows": ("alternativeRowScanning", 1)}
        if defect in keys:
            codes.codes_set(h, *keys[defect])
        elif defect == "mask":
            values = codes.codes_get_values(h)
            values[0] = 9999.
            codes.codes_set(h, "bitmapPresent", 1)
            codes.codes_set(h, "missingValue", 9999.)
            codes.codes_set_values(h, values)
    rewrite(codes, path, change)
    with pytest.raises(ValueError):
        read_atmosphere(path)


def test_changing_rows_and_moist_ozone_collocation_reject(tmp_path, codes):
    path = tmp_path/"ozone.grib2"
    counts = [4, 8, 12, 12, 8, 4]
    payloads = [message(codes, np.full(sum(counts), value), counts, short_name=name, pressure=pressure)
                for name, value in (("o3", 2.e-6), ("q", .01)) for pressure in (100000., 50000., 100.)]
    path.write_bytes(b"".join(payloads))
    mapping = {"mass_fraction_of_ozone_in_air.mass_basis": "moist_air"}
    actual = read_ozone(path, mapping)
    assert actual.ozone.shape == actual.specific_humidity.shape == (3, 48)
    np.testing.assert_array_equal(actual.row_counts, counts)
    for name, basis in (("ozone_mass_mixing_ratio", "dry_air"),):
        dry = read_ozone(path, {name: "o3", name+".mass_basis": basis})
        assert dry.specific_humidity is None and torch.equal(dry.ozone, actual.ozone)
    changed = [8, 4, 12, 12, 4, 8]
    for all_levels in (False, True):
        copies = list(payloads)
        for index, pressure in enumerate((100000., 50000., 100.)):
            if all_levels or index == 1:
                copies[3+index] = message(codes, np.full(sum(changed), .01), changed, short_name="q", pressure=pressure)
        path.write_bytes(b"".join(copies))
        with pytest.raises(ValueError, match="grid changes|collocated"):
            read_ozone(path, mapping)


@pytest.mark.parametrize("group", ("surface", "near"))
def test_reduced_surface_and_near_surface_reject(tmp_path, codes, group):
    from mpas_init.netcdf import read_near_surface, read_surface
    from test_grib_near_surface import write_near
    from test_grib_surface import write_surface
    path = tmp_path/"unsupported.grib2"
    if group == "surface":
        _, mapping = write_surface(codes, path)
    else:
        write_near(codes, path)
        mapping = LIQUID_WATER
    data = path.read_bytes()
    with path.open("rb") as stream:
        handle = codes.codes_grib_new_from_file(stream)
        length = codes.codes_get(handle, "totalLength", int)
        codes.codes_release(handle)
    settings = ({"short_name": "lsm", "typeOfLevel": "surface", "level": 0, "endStep": 21600} if group == "surface"
                else {"short_name": "t", "typeOfLevel": "heightAboveGround", "level": 2})
    path.write_bytes(message(codes, np.ones(48), [4, 8, 12, 12, 8, 4], **settings)+data[length:])
    with pytest.raises(ValueError, match="rectangular"):
        (read_surface if group == "surface" else read_near_surface)(path, mapping)
