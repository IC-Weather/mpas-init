"""Explicit hybrid GRIB transport into the existing IFS tensor reconstruction."""

from datetime import timedelta

import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_atmosphere
from test_grib import codes, rewrite
from grib_fixture import LIQUID_WATER, NAMES


def encode(codes, path, values, *, edition=2, grid="regular_ll", log_pressure=False,
           auxiliary_level="surface", fields=None, lead=21600, reverse=False):
    ap, b, ps, temperature, humidity, phi = (np.asarray(v) for v in values)
    pv = np.concatenate((ap, b)).astype(np.float32).astype(np.float64)
    levels, rows, columns = temperature.shape
    counts = [4, 8, 12, 12, 8, 4]
    latitude = (np.linspace(90., -90., rows) if grid == "regular_ll" else
                np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(rows)[0]))[::-1])
    longitude = np.arange(columns)*360./columns
    fields = ({"air_temperature": temperature, "specific_humidity": humidity,
               "eastward_wind": np.full_like(temperature, 12.),
               "northward_wind": np.full_like(temperature, -4.),
               "relative_humidity": np.full_like(temperature, 60.)} if fields is None else fields)
    pressure_name = "logarithm_of_surface_air_pressure" if log_pressure else "surface_air_pressure"
    records = [(name, NAMES[name], level + 1, value[level]) for name, value in fields.items()
               for level in range(levels)]
    records += [("surface_geopotential", "z", 1 if auxiliary_level == "hybrid" else 0, phi),
                (pressure_name, "lnsp" if log_pressure else "sp", 1 if auxiliary_level == "hybrid" else 0,
                 np.log(ps) if log_pressure else ps)]
    payloads, consumed = [], {}
    for name, short_name, level, values in records:
        reduced = grid == "reduced_gg"
        sample = f"reduced_gg_pl_32_grib{edition}" if reduced else f"{grid}_pl_grib{edition}"
        handle = codes.codes_grib_new_from_samples(sample)
        try:
            options = {"Nj": rows, "latitudeOfFirstGridPointInDegrees": latitude[0],
                       "latitudeOfLastGridPointInDegrees": latitude[-1], "longitudeOfFirstGridPointInDegrees": 0.,
                       "longitudeOfLastGridPointInDegrees": longitude[-1], "dataDate": 20240521, "dataTime": 0,
                       "shortName": short_name, "typeOfLevel": auxiliary_level if name not in NAMES else "hybrid",
                       "level": level, "stepUnits": "s", "endStep": lead}
            if grid == "regular_ll":
                options.update(Ni=columns, iDirectionIncrementInDegrees=360./columns,
                               jDirectionIncrementInDegrees=180./(rows-1))
            else:
                options.update(N=rows//2)
                if not reduced:
                    options.update(Ni=columns, iDirectionIncrementInDegrees=360./columns)
            options.update({"packingType": "grid_ieee", "precision": 2} if edition == 2 else
                           {"packingType": "grid_simple", "bitsPerValue": 32})
            for key, value in options.items():
                codes.codes_set(handle, key, value)
            if reduced:
                codes.codes_set_array(handle, "pl", counts)
                values = np.concatenate([row[::columns//count] for row, count in zip(values, counts)])
            if name in NAMES:
                codes.codes_set(handle, "PVPresent", 1)
                codes.codes_set(handle, "NV", pv.size)
                codes.codes_set_array(handle, "pv", pv)
            if name in ("eastward_wind", "northward_wind"):
                codes.codes_set(handle, "uvRelativeToGrid", 0)
            codes.codes_set_values(handle, values.reshape(-1))
            consumed.setdefault(name, []).append(codes.codes_get_values(handle).reshape(values.shape))
            payloads.append(codes.codes_get_message(handle))
        finally:
            codes.codes_release(handle)
    path.write_bytes(b"".join(payloads[::-1] if reverse else payloads))
    consumed = {name: np.stack(arrays) if name in NAMES else arrays[0] for name, arrays in consumed.items()}
    consumed_ps = np.exp(consumed[pressure_name]) if log_pressure else consumed[pressure_name]
    inputs = (*np.split(pv, 2), consumed_ps, consumed["air_temperature"], consumed["specific_humidity"], consumed["surface_geopotential"])
    return inputs, consumed


def small_inputs(rows=5, columns=24):
    eta = np.r_[0., 2.**np.arange(-6, 1)]
    index = np.arange(rows*columns).reshape(rows, columns)
    shape = np.arange(7)[:, None, None] + index[None]
    return (32768.*eta*(1.-eta), eta**2, 98304.+64.*index,
            220.+shape/4., shape/65536., 128.*index)


@pytest.mark.parametrize("defect", ("implicit", "missing_pv", "different_pv", "full_pv", "top", "bottom", "gap",
                                   "duplicate", "surface_phi", "surface_pressure", "two_pressures", "unmapped_lnsp",
                                   "thermodynamic_grid", "surface_grid", "mask", "wrong_units", "supplied_height"))
def test_incomplete_or_ambiguous_hybrid_rejects(tmp_path, codes, defect):
    path = tmp_path/"wrong.grib2"
    values = list(small_inputs())
    if defect == "top":
        values[0][0] = 1.
    elif defect == "bottom":
        values[1][-1] = .9
    encode(codes, path, values, log_pressure=defect == "unmapped_lnsp")
    payloads = []
    with path.open("rb") as stream:
        while (handle := codes.codes_grib_new_from_file(stream)) is not None:
            try:
                name = codes.codes_get(handle, "shortName")
                level = codes.codes_get(handle, "level")
                if ((defect == "gap" and level == 3) or (defect == "surface_phi" and name == "z")
                        or (defect == "surface_pressure" and name == "sp")):
                    continue
                if name == "t":
                    if defect == "missing_pv": codes.codes_set(handle, "NV", 0)
                    elif defect in ("different_pv", "full_pv"):
                        pv = codes.codes_get_array(handle, "pv")
                        if defect == "different_pv" and level == 3: pv[2] += 1.
                        if defect == "full_pv":
                            pv = pv[1:-1]
                            codes.codes_set(handle, "NV", pv.size)
                        codes.codes_set_array(handle, "pv", pv)
                    elif defect == "thermodynamic_grid":
                        codes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", 1.)
                        codes.codes_set(handle, "longitudeOfLastGridPointInDegrees", 346.)
                    elif defect == "mask":
                        codes.codes_set(handle, "bitmapPresent", 1)
                        codes.codes_set(handle, "missingValue", 9999.)
                        data = codes.codes_get_values(handle)
                        data[0] = 9999.
                        codes.codes_set_values(handle, data)
                    elif defect == "wrong_units": codes.codes_set(handle, "shortName", "r")
                if name == "z" and defect == "surface_grid":
                    codes.codes_set(handle, "longitudeOfFirstGridPointInDegrees", 1.)
                    codes.codes_set(handle, "longitudeOfLastGridPointInDegrees", 346.)
                payload = codes.codes_get_message(handle)
                payloads.append(payload)
                if name == "t" and level == 3 and defect == "duplicate": payloads.append(payload)
                if name == "sp" and defect == "two_pressures":
                    codes.codes_set(handle, "shortName", "lnsp")
                    payloads.append(codes.codes_get_message(handle))
            finally:
                codes.codes_release(handle)
    path.write_bytes(b"".join(payloads))
    mapping = {**LIQUID_WATER, **({"logarithm_of_surface_air_pressure": "lnsp"} if defect == "two_pressures" else {})}
    if defect == "supplied_height": mapping["geopotential_height"] = "gh"
    with pytest.raises((ValueError, RuntimeError)):
        read_atmosphere(path, mapping, hybrid_convention=None if defect == "implicit" else "ifs-vapour-only")


def test_split_selection_and_unselected_invalid_coefficients(tmp_path, codes):
    first, second = tmp_path/"first.grib2", tmp_path/"second.grib2"
    encode(codes, first, small_inputs(), lead=0)
    encode(codes, second, small_inputs(), lead=21600)
    expected = read_atmosphere(second, hybrid_convention="ifs-vapour-only")
    def invalidate(handle):
        if codes.codes_get(handle, "shortName") == "t":
            codes.codes_set(handle, "NV", 0)
    rewrite(codes, first, invalidate)
    actual = read_atmosphere([first, second], valid_time=expected.valid_time, hybrid_convention="ifs-vapour-only")
    for name in actual.fields:
        torch.testing.assert_close(actual.fields[name], expected.fields[name], rtol=0, atol=0)
    torch.testing.assert_close(actual.pressure, expected.pressure, rtol=0, atol=0)
    assert actual.valid_time == expected.valid_time and actual.lead == timedelta(hours=6)
    with second.open("rb") as stream:
        handle = codes.codes_grib_new_from_file(stream)
        size = codes.codes_get(handle, "totalLength", int)
        codes.codes_release(handle)
    payload = second.read_bytes()
    first.write_bytes(payload[:size])
    second.write_bytes(payload[size:])
    split = read_atmosphere([second, first], hybrid_convention="ifs-vapour-only")
    torch.testing.assert_close(split.pressure, actual.pressure, rtol=0, atol=0)
    for name in split.fields:
        torch.testing.assert_close(split.fields[name], actual.fields[name], rtol=0, atol=0)
