"""Explicit ozone interface conventions; generated transport is not producer evidence."""

from dataclasses import replace
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_ozone
from mpas_init.output import write_dycore
from mpas_init.vertical import hybrid_pressure, ifs_full_level_pressure
from test_forecast_selection import MEMBER, expand_forecast
from test_grib import codes, rewrite
from test_grib_hybrid import encode, small_inputs
from test_grib_ozone import MOIST
from test_ozone_grids import generated_grid
from test_profile_ad import assert_derivative
from test_reduced import gather_cf
from test_source_collection import split_complete
from test_tracers import write_source
from test_vertical_grid import assert_stage


def pressure_reference(ap, b, ps):
    shape = (len(ap),) + (1,) * np.ndim(ps)
    half = np.asarray(ap).reshape(shape) + np.asarray(b).reshape(shape) * ps
    return (half[:-1] + half[1:]) / 2.


def write_interfaces(path, source, *, formula="ap", dry=False, reverse=False):
    half = [0.]
    for pressure in source.pressure.numpy():
        half.append(2.*pressure-half[-1])
    assert np.all(np.diff(half) > 0)
    ap, b = np.asarray(half), np.zeros(len(half))
    ps = np.full(source.ozone.shape[1:], ap[-1])
    ps += 1024.*np.cos(np.deg2rad(source.latitude.numpy()))[:, None]*np.cos(np.deg2rad(source.longitude.numpy()))
    ap[-1], b[-1] = 0., 1.
    expected = replace(source, pressure=torch.tensor(pressure_reference(ap, b, ps)),
                       mass_basis="dry_air" if dry else "moist_air", specific_humidity=None if dry else source.specific_humidity)
    write_source(path, source, dry=dry)
    with Dataset(path, "a") as ds:
        level = ds["pressure"]
        level.delncattr("standard_name")
        level.units, level[:] = "1", np.arange(1, len(half))
        ds.createDimension("interface", len(half))
        coordinate = ds.createVariable("interface", "f8", ("interface",))
        coordinate[:], coordinate.units = np.arange(len(half)), "1"
        coordinate.standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
        coordinate.formula_terms = "ap: aa b: bb ps: ps" if formula == "ap" else "a: aa b: bb ps: ps p0: p0"
        for name, values, units in (("aa", ap/100. if formula == "ap" else ap/131072., "hPa" if formula == "ap" else "1"),
                                     ("bb", b, "1")):
            variable = ds.createVariable(name, "f8", ("interface",))
            variable[:], variable.units = values, units
        surface = ds.createVariable("ps", "f8", ("time", "latitude", "longitude"))
        surface[0], surface.units, surface.standard_name = ps/100., "hPa", "surface_air_pressure"
        if formula == "a":
            p0 = ds.createVariable("p0", "f8")
            p0[:], p0.units = 131072., "Pa"
        if reverse:
            for variable in ds.variables.values():
                data = variable[:]
                for axis, dimension in enumerate(variable.dimensions):
                    if dimension in ("interface", "pressure", "latitude", "longitude"):
                        data = np.flip(data, axis=axis)
                variable[:] = data
    if reverse:
        expected = replace(expected, pressure=expected.pressure.flip((1, 2)), latitude=expected.latitude.flip(0),
                           longitude=expected.longitude.flip(0), ozone=expected.ozone.flip((1, 2)),
                           specific_humidity=None if dry else expected.specific_humidity.flip((1, 2)))
    return expected, (ap, b, ps)


@pytest.mark.parametrize("formula", ("a", "ap"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("reduced", (False, True))
@pytest.mark.parametrize("dry", (False, True))
def test_cf_interfaces_and_selected_split_sources(tmp_path, formula, reverse, reduced, dry):
    source, _, _ = generated_grid()
    path = tmp_path/"ozone.nc"
    expected, _ = write_interfaces(path, source, formula=formula, reverse=reverse, dry=dry)
    if reduced:
        gathered = tmp_path/"gathered.nc"
        counts = [4, 8, 12, 24, 24, 12, 8, 4]
        rows, columns = gather_cf(path, gathered, counts, shuffle=True)
        lat, lon = expected.latitude.numpy()[rows], expected.longitude.numpy()[columns]
        order = np.lexsort((np.remainder(lon, 360.), -lat))
        rows, columns = rows[order], columns[order]
        expected = replace(expected, pressure=expected.pressure[:, rows, columns],
                           latitude=torch.tensor(np.sort(expected.latitude.numpy())[::-1].copy()),
                           longitude=torch.tensor(lon[order]), ozone=expected.ozone[:, rows, columns],
                           specific_humidity=None if dry else expected.specific_humidity[:, rows, columns],
                           grid_type="reduced_gaussian", row_counts=torch.tensor(counts))
        path = gathered
    mapping = {"ozone_mass_mixing_ratio": "o3"} if dry else None
    with pytest.raises(ValueError, match="interfaces"):
        read_ozone(path, mapping)
    actual = read_ozone(path, mapping, hybrid_convention="ifs-arithmetic")
    assert actual.vertical_convention == "ifs-arithmetic" and actual.mass_basis == expected.mass_basis
    assert_stage(actual.pressure, expected.pressure, 16)
    for name in ("ozone", "latitude", "longitude"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    if not dry:
        torch.testing.assert_close(actual.specific_humidity, expected.specific_humidity, rtol=0, atol=0)
    paths = split_complete(path, tmp_path/"split")
    selected = tmp_path/"selected.nc"
    expand_forecast(path, selected, matrix=True, invalid_unselected=True)
    for decoded in (read_ozone(paths[::-1], mapping, hybrid_convention="ifs-arithmetic"),
                    read_ozone(selected, mapping, hybrid_convention="ifs-arithmetic", valid_time=source.valid_time,
                                reference_time=source.reference_time, member=MEMBER)):
        torch.testing.assert_close(decoded.pressure, actual.pressure, rtol=0, atol=0)
        torch.testing.assert_close(decoded.ozone, actual.ozone, rtol=0, atol=0)
    np.savez(tmp_path/"pressure.npz", expected=expected.pressure.numpy(), actual=actual.pressure.numpy())


@pytest.mark.parametrize("defect", ("top", "bottom", "order", "labels", "interface_ids", "units", "averaged",
                                    "pressure_grid", "missing_ps", "implicit", "wrong_convention", "explicit_pressure"))
def test_cf_incomplete_or_conflicting_interfaces_reject(tmp_path, defect):
    path = tmp_path/"wrong.nc"
    write_interfaces(path, generated_grid()[0])
    with Dataset(path, "a") as ds:
        if defect == "top": ds["aa"][0] = 1.
        elif defect == "bottom": ds["bb"][-1] = .9
        elif defect == "order": ds["aa"][2] = -1.
        elif defect == "labels": ds["pressure"][1] = 99
        elif defect == "interface_ids": ds["interface"][1] = 99
        elif defect == "units": ds["ps"].units = "1"
        elif defect == "averaged": ds["ps"].cell_methods = "time: mean"
        elif defect == "missing_ps": ds["interface"].formula_terms = "ap: aa b: bb ps: absent"
        elif defect == "explicit_pressure": ds["pressure"].standard_name, ds["pressure"].units = "air_pressure", "Pa"
        elif defect == "pressure_grid":
            lat = ds.createVariable("other_latitude", "f8", ("latitude",))
            lat[:], lat.units, lat.standard_name = ds["latitude"][:]+.1, "degrees_north", "latitude"
            ds["ps"].coordinates = "other_latitude longitude"
    with pytest.raises((ValueError, RuntimeError)):
        read_ozone(path, hybrid_convention=None if defect == "implicit" else
                   "ifs-vapour-only" if defect == "wrong_convention" else "ifs-arithmetic")


def encode_interfaces(codes, path, values, *, dry=False, **options):
    inputs, _ = encode(codes, path, values, **options)
    payloads, ozone = [], []
    with path.open("rb") as stream:
        while (handle := codes.codes_grib_new_from_file(stream)) is not None:
            try:
                name = codes.codes_get(handle, "shortName")
                if name not in (("t", "sp", "lnsp") if dry else ("t", "q", "sp", "lnsp")):
                    continue
                if name == "t":
                    raw = codes.codes_get_values(handle)/2.**30
                    codes.codes_set(handle, "shortName", "o3")
                    codes.codes_set_values(handle, raw)
                    ozone.append((codes.codes_get(handle, "level", int), codes.codes_get_values(handle)))
                payloads.append(codes.codes_get_message(handle))
            finally:
                codes.codes_release(handle)
    path.write_bytes(b"".join(payloads))
    return inputs, np.stack([data for _, data in sorted(ozone)])


@pytest.mark.parametrize("defect", ("pv", "different_pv", "top", "bottom", "ps_grid", "q_grid", "missing_ps", "two_pressures"))
def test_grib_incomplete_or_ambiguous_interfaces_reject(tmp_path, codes, defect):
    path = tmp_path/"wrong.grib2"
    values = list(small_inputs())
    if defect == "top": values[0][0] = 1.
    elif defect == "bottom": values[1][-1] = .9
    encode_interfaces(codes, path, values)
    def change(handle):
        name = codes.codes_get(handle, "shortName")
        if name == "o3" and defect == "pv": codes.codes_set(handle, "NV", 0)
        elif name == "o3" and defect == "different_pv" and codes.codes_get(handle, "level") == 3:
            pv = codes.codes_get_array(handle, "pv")
            pv[1] += 1.
            codes.codes_set_array(handle, "pv", pv)
        elif name == ("sp" if defect == "ps_grid" else "q") and defect in ("ps_grid", "q_grid"):
            for key in ("longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees"):
                codes.codes_set(handle, key, (codes.codes_get(handle, key)+.5) % 360.)
        elif name == "sp" and defect == "missing_ps": codes.codes_set(handle, "shortName", "z")
    rewrite(codes, path, change)
    if defect == "two_pressures":
        with path.open("rb") as stream:
            while (handle := codes.codes_grib_new_from_file(stream)) is not None:
                try:
                    if codes.codes_get(handle, "shortName") == "sp":
                        codes.codes_set(handle, "shortName", "lnsp")
                        duplicate = codes.codes_get_message(handle)
                        break
                finally:
                    codes.codes_release(handle)
        with path.open("ab") as stream: stream.write(duplicate)
    mapping = {**MOIST, **({"logarithm_of_surface_air_pressure": "lnsp"} if defect == "two_pressures" else {})}
    with pytest.raises((ValueError, RuntimeError)):
        read_ozone(path, mapping, hybrid_convention="ifs-arithmetic")


def test_shared_pressure_kernel_derivatives_and_controls(tmp_path):
    values = tuple(torch.tensor(v, dtype=torch.float64) for v in small_inputs()[:3])
    def calculate(ap, b, ps): return ifs_full_level_pressure(hybrid_pressure(ap, b, ps, interfaces=True))
    actual = calculate(*values)
    assert_stage(actual, torch.tensor(pressure_reference(*(v.numpy() for v in values))), 16)
    arrays = {"actual": actual.numpy()}
    for index in range(3):
        direction = tuple((v*.001 if i == 0 else v*(1.-v)*.001 if i == 1 else torch.full_like(v, 128.))
                          if i == index else torch.zeros_like(v) for i, v in enumerate(values))
        _, tangent = torch.func.jvp(calculate, values, direction)
        _, pullback = torch.func.vjp(calculate, *values)
        seed = torch.cos(torch.arange(actual.numel(), dtype=actual.dtype).reshape(actual.shape))
        reverse = sum((a*b).sum() for a, b in zip(pullback(seed), direction))
        for h in (1./16, 1./32, 1./64):
            plus = pressure_reference(*((v+h*d).numpy() for v, d in zip(values, direction)))
            minus = pressure_reference(*((v-h*d).numpy() for v, d in zip(values, direction)))
            finite = torch.tensor((plus-minus)/(2*h))
            assert_derivative(tangent, finite)
            assert_derivative(reverse, (finite*seed).sum())
            with pytest.raises(AssertionError): assert_derivative(torch.zeros_like(tangent), finite)
            arrays[f"fd_{index}_{h}"] = finite.numpy()
        arrays[f"jvp_{index}"] = tangent.numpy()
    np.savez(tmp_path/"derivatives.npz", **arrays)


def test_cli_convention_requires_source(tmp_path):
    result = subprocess.run([sys.executable, "-B", "-m", "mpas_init", "--source", "absent.nc", "--static", "absent.nc",
                             "--output", str(tmp_path/"unused.nc"), "--ozone-hybrid-convention", "ifs-arithmetic"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "--ozone-hybrid-convention requires --ozone-source" in result.stderr
    assert not (tmp_path/"unused.nc").exists()


@pytest.mark.parametrize("tracers,convention", ((None, "ifs-arithmetic"), ({}, "ifs-vapour-only")))
def test_writer_rejects_unsupported_or_orphan_convention(tmp_path, tracers, convention):
    with pytest.raises(ValueError, match="ozone vertical convention"):
        write_dycore(tmp_path/"unused.nc", {}, None, generated_grid()[1].valid_time,
                     tracers=tracers, ozone_vertical_convention=convention)
    assert not (tmp_path/"unused.nc").exists()
