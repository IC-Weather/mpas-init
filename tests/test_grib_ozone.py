"""Actual GRIB codecs, explicit mass bases and unchanged UFS tracer numerics."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_ozone
from test_grib import codes, message, rewrite


MOIST = {"mass_fraction_of_ozone_in_air": "o3", "mass_fraction_of_ozone_in_air.mass_basis": "moist_air"}
DRY = {"ozone_mass_mixing_ratio": "o3", "ozone_mass_mixing_ratio.mass_basis": "dry_air"}


def write_source(codes, path, *, dry=False, omit=None, **settings):
    expected, payloads = {}, []
    for name in (("o3",) if dry else ("o3", "q")):
        expected[name] = []
        for pressure in (100000., 50000., 100.):
            payload, values, latitude, longitude = message(codes, short_name=name, pressure=pressure, **settings)
            if name == "o3":
                values = values / 2.**24
                h = codes.codes_new_from_message(payload)
                try:
                    # Apply the known dyadic scale in the existing encoded scan order.
                    codes.codes_set_values(h, codes.codes_get_values(h) / 2.**24)
                    payload = codes.codes_get_message(h)
                finally:
                    codes.codes_release(h)
            expected[name].append(values)
            if omit != (name, pressure):
                payloads.append(payload)
    path.write_bytes(b"".join(payloads))
    return {name: np.stack(values) for name, values in expected.items()}, latitude, longitude


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("scan", ({}, {"reverse_lat": True, "reverse_lon": True, "column_scan": True}, {"grid": "regular_gg"}))
@pytest.mark.parametrize("dry", (False, True))
def test_exact_decoded_values_forecast_scans_and_basis(tmp_path, codes, edition, scan, dry):
    path = tmp_path / "source.nc"
    expected, lat, lon = write_source(codes, path, dry=dry, edition=edition, lead=21600, **scan)
    source = read_ozone(path, DRY if dry else MOIST)
    np.testing.assert_array_equal(source.ozone, expected["o3"])
    np.testing.assert_allclose(source.latitude, lat, rtol=0, atol=3.e-14)
    np.testing.assert_array_equal(source.longitude, lon)
    np.testing.assert_array_equal(source.pressure, [100000., 50000., 100.])
    assert source.ozone.dtype == source.pressure.dtype == torch.float64
    assert source.mass_basis == ("dry_air" if dry else "moist_air")
    assert source.reference_time == datetime(2024, 5, 21, tzinfo=timezone.utc)
    assert source.lead == timedelta(hours=6) and source.valid_time == source.reference_time + source.lead
    if dry:
        assert source.specific_humidity is None
    else:
        np.testing.assert_array_equal(source.specific_humidity, expected["q"])


@pytest.mark.parametrize("mapping", ({}, {"mass_fraction_of_ozone_in_air": "o3"},
                                     {"ozone_mass_mixing_ratio": "o3"},
                                     {**MOIST, "mass_fraction_of_ozone_in_air.mass_basis": "dry_air"},
                                     {**DRY, "ozone_mass_mixing_ratio.mass_basis": "moist_air"},
                                     {**MOIST, **DRY},
                                     {**MOIST, "ozone_mass_mixing_ratio.mass_basis": "dry_air"}))
def test_missing_conflicting_or_wrong_mass_basis_rejects(tmp_path, codes, mapping):
    path = tmp_path / "source.grib"
    write_source(codes, path)
    with pytest.raises(ValueError, match="mass_basis|conflicting ozone"):
        read_ozone(path, mapping)


@pytest.mark.parametrize("defect", ("humidity", "ozone", "duplicate", "grid", "level_grid", "units", "average", "bitmap"))
def test_incomplete_or_incompatible_fields_reject(tmp_path, codes, defect):
    path = tmp_path / "source.grib"
    omit = ("q" if defect == "humidity" else "o3", 50000.) if defect in ("humidity", "ozone") else None
    write_source(codes, path, omit=omit)
    if defect == "duplicate":
        with path.open("ab") as f:
            f.write(message(codes, short_name="o3", pressure=50000)[0])
    def change(h):
        name = codes.codes_get(h, "shortName")
        if defect == "grid" and name == "q" or defect == "level_grid" and name == "o3" and codes.codes_get(h, "level") == 500:
            for key in ("longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees"):
                codes.codes_set(h, key, (codes.codes_get(h, key) + .5) % 360.)
        if name == "o3":
            if defect == "units":
                codes.codes_set(h, "shortName", "tco3")
            elif defect == "average":
                codes.codes_set(h, "stepType", "avg")
            elif defect == "bitmap":
                values = codes.codes_get_values(h)
                values[0] = 9999.
                codes.codes_set(h, "missingValue", 9999.)
                codes.codes_set(h, "bitmapPresent", 1)
                codes.codes_set_values(h, values)
    if defect not in ("humidity", "ozone", "duplicate"):
        rewrite(codes, path, change)
    mapping = {**MOIST, "mass_fraction_of_ozone_in_air": "tco3"} if defect == "units" else MOIST
    with pytest.raises(ValueError, match="complete pressure|duplicate|collocated|grid changes|units|instantaneous|missing GRIB"):
        read_ozone(path, mapping)


def test_split_sources_cycles_members_and_pa_levels(tmp_path, codes):
    first, second = tmp_path / "first.grib", tmp_path / "second.grib"
    write_source(codes, first, lead=21600, member=2)
    write_source(codes, second, lead=10800, member=3, dataTime=300)
    valid = datetime(2024, 5, 21, 6, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="exactly one"):
        read_ozone([first, second], MOIST, valid_time=valid)
    source = read_ozone([first, second], MOIST, valid_time=valid, member=3)
    assert source.reference_time.hour == 3 and source.member == 3
    with pytest.raises(ValueError, match="exactly one"):
        read_ozone(first, MOIST, member=3)
    def pa(h):
        level = codes.codes_get(h, "level")
        codes.codes_set(h, "typeOfLevel", "isobaricInPa")
        codes.codes_set(h, "level", level * 100.)
    rewrite(codes, first, pa)
    assert torch.equal(read_ozone(first, MOIST).pressure, source.pressure)
    humidity, ozone = tmp_path / "q.grib", tmp_path / "o3.grib"
    with first.open("rb") as f, humidity.open("wb") as q, ozone.open("wb") as o:
        while (h := codes.codes_grib_new_from_file(f)) is not None:
            try:
                codes.codes_write(h, q if codes.codes_get(h, "shortName") == "q" else o)
            finally:
                codes.codes_release(h)
    selected = read_ozone([ozone, humidity], MOIST)
    assert torch.equal(selected.ozone, source.ozone) and torch.equal(selected.specific_humidity, source.specific_humidity)
