from dataclasses import replace
from datetime import timedelta, timezone

from netCDF4 import Dataset, date2num, num2date
import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_atmosphere, read_ozone, read_ufs_surface
from test_atmosphere import generated_source
from test_source_collection import assert_source_equal, split_complete


MEMBER = 2**53 + 1


def expand_forecast(source, target, *, matrix=False, invalid_unselected=False, members=(MEMBER - 1, MEMBER)):
    """Re-encode real/generated singleton values; only [member=1, cycle=0, time=0] stays unchanged."""
    with Dataset(source) as src, Dataset(target, "w", clobber=False) as dst:
        clocks = {getattr(v, "standard_name", ""): v for v in src.variables.values()
                  if getattr(v, "standard_name", "") in ("time", "forecast_reference_time", "forecast_period")}
        time_dim = clocks["time"].dimensions[0]
        for name, dim in src.dimensions.items():
            dst.createDimension(name, 2 if name == time_dim else len(dim))
        dst.createDimension("ensemble", len(members))
        if matrix:
            dst.createDimension("cycle", 2)
        member = dst.createVariable("realization", "i8", ("ensemble",))
        member[:], member.standard_name, member.units = members, "realization", "1"
        units = clocks["time"].units
        base = num2date(np.asarray(clocks["time"][:]).reshape(-1)[0], units)
        reference = num2date(np.asarray(clocks["forecast_reference_time"][:]).reshape(-1)[0], clocks["forecast_reference_time"].units)
        period = float(np.asarray(clocks["forecast_period"][:]).reshape(-1)[0])
        period_unit = clocks["forecast_period"].units
        for name, var in src.variables.items():
            standard = getattr(var, "standard_name", "")
            dims = var.dimensions
            if standard == "time":
                dims = ("cycle", time_dim) if matrix else (time_dim,)
            elif standard == "forecast_reference_time" and matrix:
                dims = ("cycle",)
            elif standard == "forecast_period":
                dims = (time_dim,)
            elif time_dim in dims:
                dims = ("ensemble", *dims, "cycle") if matrix else ((*dims, "ensemble") if len(name) % 2 else ("ensemble", *dims))
            options = {"fill_value": var.getncattr("_FillValue")} if "_FillValue" in var.ncattrs() else {}
            out = dst.createVariable(name, var.dtype, dims, zlib=True, complevel=1, **options)
            out.setncatts({key: var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
            if standard == "time":
                values = [[date2num(base + timedelta(hours=6 * (t - c)), units) for t in range(2)] for c in range(2 if matrix else 1)]
                out[:] = values if matrix else values[0]
            elif standard == "forecast_reference_time":
                out[:] = [date2num(reference - timedelta(hours=6 * c), var.units) for c in range(2)] if matrix else var[:]
            elif standard == "forecast_period":
                out[:] = [period, period + (6 if period_unit == "hours" else 21600)]
            elif time_dim in var.dimensions:
                base_values = np.take(var[:], 0, axis=var.dimensions.index(time_dim))
                out.coordinates = (getattr(var, "coordinates", "") + " realization").strip()
                for m in range(len(members)):
                    for t in range(2):
                        for c in range(2 if matrix else 1):
                            index = tuple({"ensemble": m, time_dim: t, "cycle": c}.get(dim, slice(None)) for dim in dims)
                            selected = m == len(members) - 1 and t == 0 and c == 0
                            values = base_values
                            if not selected and standard in ("eastward_wind", "northward_wind"):
                                values = values + 10 * (1 + m + t + c)
                            out[index] = np.ma.masked if invalid_unselected and not selected else values
            else:
                out[:] = var[:]
    return target


@pytest.mark.parametrize("representation", ["a", "ap", "3d"])
@pytest.mark.parametrize("reverse", [False, True])
def test_time_member_selection_reads_only_requested_hyperslab(tmp_path, representation, reverse):
    base, forecast = tmp_path / "base.nc", tmp_path / "forecast.nc"
    generated_source(base, representation, reverse=reverse)
    expected = replace(read_atmosphere(base, include_relative_humidity=True), member=MEMBER)
    expand_forecast(base, forecast, invalid_unselected=True)
    actual = read_atmosphere(forecast, valid_time=expected.valid_time, member=MEMBER, include_relative_humidity=True)
    assert_source_equal(actual, expected)
    with Dataset(forecast, "a") as ds:
        assert np.ma.is_masked(ds["air_temperature"][:])
        ds.checked_after_close = 1


def test_two_dimensional_valid_time_requires_unique_cycle(tmp_path):
    base, forecast = tmp_path / "base.nc", tmp_path / "forecast.nc"
    generated_source(base, "ap")
    expected = replace(read_atmosphere(base), member=MEMBER)
    expand_forecast(base, forecast, matrix=True)
    with pytest.raises(ValueError, match="ambiguous"):
        read_atmosphere(forecast, valid_time=expected.valid_time, member=MEMBER)
    local = expected.valid_time.astimezone(timezone(timedelta(hours=-4)))
    actual = read_atmosphere(forecast, valid_time=local, reference_time=expected.reference_time, member=MEMBER)
    assert_source_equal(actual, expected)
    previous = read_atmosphere(forecast, valid_time=local, reference_time=expected.reference_time - timedelta(hours=6), member=MEMBER)
    assert previous.lead == expected.lead + timedelta(hours=6)
    assert torch.equal(previous.fields["eastward_wind"], expected.fields["eastward_wind"] + 40)


def test_noninitial_time_and_singleton_realization_axes(tmp_path):
    base, forecast = tmp_path / "base.nc", tmp_path / "forecast.nc"
    generated_source(base, "3d")
    expected = read_atmosphere(base, include_relative_humidity=True)
    expand_forecast(base, forecast)
    actual = read_atmosphere(forecast, valid_time=expected.valid_time + timedelta(hours=6), member=MEMBER, include_relative_humidity=True)
    assert actual.valid_time == expected.valid_time + timedelta(hours=6)
    assert actual.lead == expected.lead + timedelta(hours=6)
    for name, value in expected.fields.items():
        assert torch.equal(actual.fields[name], value + 30 if name in ("eastward_wind", "northward_wind") else value)
    single = tmp_path / "singleton.nc"
    expand_forecast(base, single, members=(MEMBER,))
    assert_source_equal(read_atmosphere(single, valid_time=expected.valid_time, include_relative_humidity=True), replace(expected, member=MEMBER))
    with pytest.raises(ValueError, match="requires a source realization"):
        read_atmosphere(base, valid_time=expected.valid_time, member=MEMBER)


@pytest.mark.parametrize("defect", ["missing_time", "missing_member", "absent_time", "absent_member", "cycle", "duplicate_time",
                                   "duplicate_member", "fractional_member", "member_units", "clock", "naive", "float_member", "bool_member"])
def test_invalid_selections_reject(tmp_path, defect):
    base, forecast = tmp_path / "base.nc", tmp_path / "forecast.nc"
    generated_source(base, "ap")
    expected = read_atmosphere(base)
    expand_forecast(base, forecast)
    options = {"valid_time": expected.valid_time, "member": MEMBER}
    if defect == "missing_time":
        options.pop("valid_time")
    elif defect == "missing_member":
        options.pop("member")
    elif defect == "absent_time":
        options["valid_time"] += timedelta(hours=1)
    elif defect == "absent_member":
        options["member"] += 1
    elif defect == "cycle":
        options["reference_time"] = expected.reference_time - timedelta(hours=1)
    elif defect == "naive":
        options["valid_time"] = expected.valid_time.replace(tzinfo=None)
    elif defect in ("float_member", "bool_member"):
        options["member"] = 1.0 if defect == "float_member" else True
    else:
        with Dataset(forecast, "a") as ds:
            if defect == "duplicate_time":
                ds["time"][1] = ds["time"][0]
            elif defect == "duplicate_member":
                ds["realization"][0] = ds["realization"][1]
            elif defect == "fractional_member":
                ds.renameVariable("realization", "old_member")
                ds["old_member"].delncattr("standard_name")
                var = ds.createVariable("realization", "f8", ("ensemble",))
                var.standard_name, var.units, var[:] = "realization", "1", [0., 1.5]
            elif defect == "member_units":
                ds["realization"].units = "kg"
            elif defect == "clock":
                ds["lead"][0] += 1
    with pytest.raises(ValueError):
        read_atmosphere(forecast, **options)
    with Dataset(forecast, "a") as ds:
        ds.checked_after_rejection = 1


def test_split_forecasts_select_before_collection_and_keep_integer_labels(tmp_path):
    base, forecast = tmp_path / "base.nc", tmp_path / "forecast.nc"
    generated_source(base, "a")
    expected = replace(read_atmosphere(base, include_relative_humidity=True), member=MEMBER)
    expand_forecast(base, forecast)
    paths = split_complete(forecast, tmp_path / "split")
    with Dataset(paths[1], "a") as ds:
        ds["time"][1] += 21600
        ds["lead"][1] += 6
    selected = {"valid_time": expected.valid_time, "member": MEMBER}
    for order in (paths, paths[::-1]):
        assert_source_equal(read_atmosphere(order, include_relative_humidity=True, **selected), expected)
    # Adjacent labels above 2**53 must not collapse through a floating comparison.
    other = read_atmosphere(forecast, valid_time=expected.valid_time, member=MEMBER - 1)
    assert other.member == MEMBER - 1
    assert torch.equal(other.fields["eastward_wind"], expected.fields["eastward_wind"] + 10)


@pytest.mark.parametrize("kind", ["ozone", "ufs_surface"])
def test_optional_forecast_codecs_and_explicit_clock_mappings(tmp_path, kind):
    if kind == "ozone":
        from test_tracers import generated_inputs, write_source
        read = read_ozone
    else:
        from test_ufs_surface import generated_inputs, write_source
        read = read_ufs_surface
    source, *_ = generated_inputs()
    base, forecast = tmp_path / "base.nc", tmp_path / "forecast.nc"
    write_source(base, source)
    expected = replace(read(base), member=MEMBER)
    expand_forecast(base, forecast, invalid_unselected=True)
    mapping = {}
    with Dataset(forecast, "a") as ds:
        for standard in ("time", "forecast_reference_time", "forecast_period", "realization"):
            variable = next(v for v in ds.variables.values() if getattr(v, "standard_name", "") == standard)
            renamed = "renamed_" + variable.name
            ds.renameVariable(variable.name, renamed)
            ds[renamed].delncattr("standard_name")
            mapping[standard] = renamed
    assert_source_equal(read(forecast, mapping, valid_time=expected.valid_time, member=MEMBER), expected)
