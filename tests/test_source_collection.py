from contextlib import nullcontext
from dataclasses import fields, is_dataclass
import re
import shutil

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import read_atmosphere, read_ozone, read_ufs_surface
from test_atmosphere import generated_source
from test_gaussian import make_cf_gaussian


def copy_variables(source, output, names, *, identity=True):
    source_context = nullcontext(source) if isinstance(source, Dataset) else Dataset(source)
    with source_context as src, Dataset(output, "w", clobber=False) as dst:
        names = set(names)
        if identity:
            names.update(name for name, var in src.variables.items() if getattr(var, "standard_name", "") in (
                "time", "forecast_reference_time", "forecast_period", "realization"))
        dimensions = {dim for name in names for dim in src[name].dimensions}
        dimensions.update(dim for name in names for dim in getattr(src[name], "compress", "").split())
        for name in dimensions:
            dst.createDimension(name, len(src.dimensions[name]))
        for name in src.variables:
            if name not in names:
                continue
            var = src[name]
            options = {"fill_value": var.getncattr("_FillValue")} if "_FillValue" in var.ncattrs() else {}
            out = dst.createVariable(name, var.dtype, var.dimensions, **options)
            out.setncatts({key: var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
            mask, scale = var.mask, var.scale
            try:
                var.set_auto_maskandscale(False)
                out.set_auto_maskandscale(False)
                out[:] = var[:]
            finally:
                var.set_auto_mask(mask)
                var.set_auto_scale(scale)
    return output


def test_copy_fixture_preserves_metadata_masks_and_caller_handle(tmp_path):
    source = tmp_path / "source.nc"
    with Dataset(source, "w", clobber=False) as ds:
        ds.createDimension("point", 3)
        field = ds.createVariable("temperature", "i2", ("point",), fill_value=-32768)
        field.setncatts({"units": "K", "scale_factor": .25, "add_offset": 250.})
        field[:] = np.ma.array([261., 272., 283.], mask=[False, True, False])
        lead = ds.createVariable("lead", "f8")
        lead.setncatts({"standard_name": "forecast_period", "units": "hours"})
        lead[:] = 6.
        ds.createVariable("omitted", "f8")[:] = 1.
    path_copy = copy_variables(source, tmp_path / "path.nc", {"temperature"})
    with Dataset(source) as ds:
        handle_copy = copy_variables(ds, tmp_path / "handle.nc", {"temperature"})
        assert ds.isopen()
        with pytest.raises(OSError):
            copy_variables(ds, handle_copy, {"temperature"})
        assert ds.isopen()
        for path in (path_copy, handle_copy):
            with Dataset(path) as copied:
                assert copied.data_model == ds.data_model
                assert set(copied.variables) == {"temperature", "lead"}
                assert {name: len(dim) for name, dim in copied.dimensions.items()} == {"point": 3}
                for name, actual in copied.variables.items():
                    expected = ds[name]
                    assert actual.dtype == expected.dtype and actual.dimensions == expected.dimensions
                    assert {key: actual.getncattr(key) for key in actual.ncattrs()} == {
                        key: expected.getncattr(key) for key in expected.ncattrs()}
                    np.testing.assert_array_equal(np.ma.getmaskarray(actual[:]), np.ma.getmaskarray(expected[:]))
                    np.testing.assert_array_equal(actual[:].data, expected[:].data)


@pytest.mark.parametrize("mask,scale", [(False, False), (False, True), (True, False), (True, True)])
def test_copy_fixture_preserves_raw_values_and_caller_read_modes(tmp_path, mask, scale):
    source, target = tmp_path / "source.nc", tmp_path / "copy.nc"
    with Dataset(source, "w", clobber=False) as ds:
        ds.createDimension("point", 3)
        packed = ds.createVariable("packed", "i2", ("point",), fill_value=-32768)
        packed.setncatts({"units": "K", "scale_factor": .25, "add_offset": 250.})
        packed[:] = np.ma.array([261., 272., 283.], mask=[False, True, False])
        plain = ds.createVariable("plain", "f4", ("point",), fill_value=-9999.)
        plain[:] = np.ma.array([-0., .1, -2.], mask=[False, True, False])
    with Dataset(source) as ds:
        for var in ds.variables.values():
            var.set_auto_mask(mask)
            var.set_auto_scale(scale)
        copy_variables(ds, target, {"packed", "plain"}, identity=False)
        assert ds.isopen()
        assert all(var.mask == mask and var.scale == scale for var in ds.variables.values())
    with Dataset(source) as expected, Dataset(target) as actual:
        for name in ("packed", "plain"):
            x, y = expected[name], actual[name]
            assert x.dtype == y.dtype and x.dimensions == y.dimensions
            assert {key: x.getncattr(key) for key in x.ncattrs()} == {
                key: y.getncattr(key) for key in y.ncattrs()}
            np.testing.assert_array_equal(np.ma.getmaskarray(x[:]), np.ma.getmaskarray(y[:]))
            np.testing.assert_array_equal(x[:].data, y[:].data)
            x.set_auto_maskandscale(False)
            y.set_auto_maskandscale(False)
            assert x[:].tobytes() == y[:].tobytes(), name
        np.testing.assert_array_equal(expected["packed"][:], np.array([44, -32768, 132], dtype="i2"))


def split_generated(source, output, *, representation="ap", gaussian=False):
    if gaussian:
        make_cf_gaussian(source, representation)
    else:
        generated_source(source, representation)
    output.mkdir(exist_ok=False)
    common = {"latitude", "longitude"}
    with Dataset(source) as ds:
        if "level" in ds.variables:
            common.add("level")
        paths = [copy_variables(ds, output / f"{name}.nc", common | {name}) for name in (
            "air_temperature", "eastward_wind", "northward_wind", "specific_humidity", "geopotential_height", "relative_humidity")]
        if representation == "3d":
            paths.append(copy_variables(ds, output / "pressure.nc", common | {"pressure"}))
        else:
            paths.append(copy_variables(ds, output / "surface-pressure.nc", common | {"ps"}))
            names = {"level", "aa", "bb"} | ({"p0"} if representation == "a" else set())
            paths.append(copy_variables(ds, output / "coefficients.nc", names, identity=False))
    return paths


def assert_source_equal(actual, expected):
    assert is_dataclass(actual) and type(actual) is type(expected)
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(b, dict):
            assert set(a) == set(b)
            for name in b:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        else:
            assert a == b


@pytest.mark.parametrize("representation", ["a", "ap", "3d"])
@pytest.mark.parametrize("gaussian", [False, True])
def test_split_full_levels_and_timeless_coefficients(tmp_path, representation, gaussian):
    source = tmp_path / "whole.nc"
    paths = split_generated(source, tmp_path / "split", representation=representation, gaussian=gaussian)
    expected = read_atmosphere(source, include_relative_humidity=True)
    for order in (paths, paths[::-1], paths[2:] + paths[:2]):
        assert_source_equal(read_atmosphere(order, include_relative_humidity=True), expected)
    # All handles close on success; no merged file is needed.
    for path in paths:
        with Dataset(path, "a") as ds:
            ds.checked_after_close = 1


def test_clock_encodings_normalize_but_cycles_do_not_mix(tmp_path):
    source = tmp_path / "whole.nc"
    paths = split_generated(source, tmp_path / "split")
    expected = read_atmosphere(source)
    with Dataset(paths[1], "a") as ds:
        for name in ("time", "reference"):
            ds[name][:] = ds[name][:] / 3600.
            ds[name].units = "hours since 2024-05-21 00:00:00"
        ds["lead"][:] = ds["lead"][:] * 3600.
        ds["lead"].units = "seconds"
    assert_source_equal(read_atmosphere(paths), expected)
    # Same valid time, different cycle and lead must not join.
    with Dataset(paths[1], "a") as ds:
        ds["reference"][:] = 1.
        ds["lead"][:] = 5. * 3600.
    with pytest.raises(ValueError, match="identities differ"):
        read_atmosphere(paths)


@pytest.mark.parametrize("defect", ["duplicate_file", "duplicate_field", "coordinate", "vertical", "coefficients",
                                    "missing_clock", "calendar", "valid_time", "missing_field", "timeless_state"])
def test_collection_conflicts_reject_and_close_handles(tmp_path, defect):
    source = tmp_path / "whole.nc"
    paths = split_generated(source, tmp_path / "split")
    error = "duplicate|conflict|expected one|calendar|identities|forecast"
    if defect == "duplicate_file":
        paths.append(paths[0])
    elif defect == "duplicate_field":
        new = tmp_path / "duplicate.nc"
        shutil.copyfile(paths[0], new)
        paths.append(new)
    elif defect == "missing_field":
        paths.pop(0)
    elif defect == "coefficients":
        new = tmp_path / "coefficients.nc"
        shutil.copyfile(paths[-1], new)
        with Dataset(new, "a") as ds:
            ds["aa"][0] += 1.
        paths.append(new)
    else:
        path = paths[1]
        with Dataset(path, "a") as ds:
            if defect == "coordinate":
                ds["latitude"][1] += .01
            elif defect == "vertical":
                ds["level"][1] += 1.
            elif defect == "missing_clock":
                ds["lead"].delncattr("standard_name")
            elif defect == "calendar":
                ds["time"].calendar = "360_day"
            elif defect == "valid_time":
                ds["time"][:] = 25200.
                ds["lead"][:] = 7.
            elif defect == "timeless_state":
                for name in ("time", "reference", "lead"):
                    ds[name].delncattr("standard_name")
    with pytest.raises(ValueError, match=error):
        read_atmosphere(paths)
    for path in paths:
        with Dataset(path, "a") as ds:
            ds.checked_after_rejection = 1


@pytest.mark.parametrize("defect", ["different", "missing", "fractional", "multiple", "large_adjacent"])
def test_member_identity_cannot_mix_across_files(tmp_path, defect):
    source = tmp_path / "whole.nc"
    paths = split_generated(source, tmp_path / "split")
    for path in paths[:-1]:
        with Dataset(path, "a") as ds:
            if defect == "multiple":
                ds.createDimension("realization", 2)
            var = ds.createVariable("member", "i8" if defect != "fractional" else "f8",
                                     ("realization",) if defect == "multiple" else ())
            var.standard_name, var.units = "realization", "1"
            var[:] = [1, 2] if defect == "multiple" else (2**53 if defect == "large_adjacent" else 1)
    with Dataset(paths[1], "a") as ds:
        if defect == "missing":
            ds["member"].delncattr("standard_name")
        elif defect == "different":
            ds["member"][:] = 2
        elif defect == "fractional":
            ds["member"][:] = 1.5
        elif defect == "large_adjacent":
            ds["member"][:] = 2**53 + 1
    with pytest.raises(ValueError, match="identities differ|one integer realization"):
        read_atmosphere(paths)


def test_member_mapping_and_single_file_selection(tmp_path):
    path = tmp_path / "source.nc"
    generated_source(path, "ap")
    with Dataset(path, "a") as ds:
        var = ds.createVariable("selected_member", "i8")
        var[:] = 2**53 + 1
        var.units = "1"
    assert read_atmosphere(path, {"realization": "selected_member"}).member == 2**53 + 1


def test_consistent_member_and_explicit_mapping_across_split_files(tmp_path):
    source = tmp_path / "whole.nc"
    paths = split_generated(source, tmp_path / "split")
    for path in [source, *paths[:-1]]:
        with Dataset(path, "a") as ds:
            var = ds.createVariable("selected_member", "i8")
            var[:], var.units = 2**53 + 1, "1"
    mapping = {"realization": "selected_member"}
    expected = read_atmosphere(source, mapping)
    for order in (paths, paths[::-1]):
        assert_source_equal(read_atmosphere(order, mapping), expected)


def test_split_formula_variable_names_preserve_single_file_contract(tmp_path):
    source = tmp_path / "whole.nc"
    paths = split_generated(source, tmp_path / "split")
    for path in [source, *paths]:
        with Dataset(path, "a") as ds:
            for old, new in (("aa", "hybrid-ap"), ("bb", "hybrid.b"), ("ps", "surface-pressure")):
                if old in ds.variables:
                    ds.renameVariable(old, new)
            if "level" in ds.variables:
                ds["level"].formula_terms = "ap: hybrid-ap b: hybrid.b ps: surface-pressure"
    expected = read_atmosphere(source)
    assert_source_equal(read_atmosphere(paths), expected)


def split_complete(source, directory):
    """Keep physical variables separate; retain exact shared coordinates and identity."""
    directory.mkdir(exist_ok=False)
    with Dataset(source) as ds:
        metadata = {name for name, var in ds.variables.items() if var.ndim == 1 or
                    getattr(var, "standard_name", "") in ("time", "forecast_reference_time", "forecast_period", "realization")}
        for var in ds.variables.values():
            for attribute in ("coordinates", "bounds"):
                metadata.update(getattr(var, attribute, "").split())
            metadata.update(re.findall(r"[A-Za-z_]\w*\s*:\s*([^\s:]+)",
                                       getattr(var, "formula_terms", "")))
        metadata = {name for name in metadata if name in ds.variables and ds[name].ndim <= 2}
        groups = [[name] for name in ds.variables if name not in metadata]
        return [copy_variables(ds, directory / f"field-{i:02}.nc", metadata | set(group))
                for i, group in enumerate(groups)]


@pytest.mark.parametrize("kind", ["ozone", "ufs_surface"])
def test_generated_split_optional_sources(tmp_path, kind):
    if kind == "ozone":
        from test_tracers import generated_inputs, write_source
        read = read_ozone
    else:
        from test_ufs_surface import generated_inputs, write_source
        read = read_ufs_surface
    path = tmp_path / "whole.nc"
    write_source(path, generated_inputs()[0])
    paths = split_complete(path, tmp_path / "split")
    for order in (paths, paths[::-1]):
        assert_source_equal(read(order), read(path))
