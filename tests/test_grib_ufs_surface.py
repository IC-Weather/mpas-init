"""Explicit GRIB transport of complete analyzed UFS quantities."""

from datetime import timedelta

import numpy as np
import pytest
import torch

from mpas_init.netcdf import UFS_SURFACE_VARIABLES, read_ufs_surface
from test_grib import codes, message, rewrite


PARAMETERS = {
    "aerodynamic_roughness_length": (2, 0, 1),
    "visible_strong_cosine_albedo": (0, 192, 192),
    "near_ir_strong_cosine_albedo": (0, 192, 193),
    "visible_weak_cosine_albedo": (0, 192, 194),
    "near_ir_weak_cosine_albedo": (0, 192, 195),
    "wind_speed_ratio_10m_to_lowest_model_level": (0, 192, 196),
    "surface_heat_transfer_function": (0, 192, 197),
    "surface_momentum_transfer_function": (0, 192, 198),
    "canopy_water_amount": (2, 0, 13), "friction_velocity": (0, 2, 197),
    "sea_ice_thickness": (10, 2, 1), "specific_humidity_2m": (0, 1, 0),
    "air_temperature_2m": (0, 0, 0), "deep_soil_temperature": (2, 0, 2),
    "sea_ice_surface_temperature": (10, 2, 8),
}


def write_ufs(codes, path, *, edition=2, overrides=None, **scan):
    """Use generated local parameters where codec tables lack the quantity.

    These are transport fixtures, not NCEP products or inferred definitions of
    strong/weak cosine albedo. Explicit mappings provide those physical roles.
    """
    payloads, expected, grids, mapping = [], {}, {}, {}
    for index, (name, units) in enumerate(UFS_SURFACE_VARIABLES.items()):
        options = {**scan, **(overrides or {}).get(name, {})}
        shift = options.pop("shift", 0.)
        base, _, latitude, longitude = message(codes, edition=edition, lead=21600, **options)
        h = codes.codes_new_from_message(base)
        try:
            codes.codes_set(h, "centre", 7)
            for key in ("longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees"):
                codes.codes_set(h, key, (codes.codes_get(h, key) + shift) % 360.)
            if edition == 2:
                codes.codes_set(h, "tablesVersion", 36)
                codes.codes_set(h, "localTablesVersion", 254)
                for key, number in zip(("discipline", "parameterCategory", "parameterNumber"), PARAMETERS[name]):
                    codes.codes_set(h, key, number)
                mapping[name] = "grib2:7:36:254:" + ":".join(map(str, PARAMETERS[name]))
            else:
                codes.codes_set(h, "table2Version", 250)
                codes.codes_set(h, "indicatorOfParameter", 200 + index)
                mapping[name] = f"grib1:7:250:{200+index}"
            kind, level = (("heightAboveGround", 2) if name in ("air_temperature_2m", "specific_humidity_2m")
                           else ("depthBelowLand", 3) if name == "deep_soil_temperature" else ("surface", 0))
            codes.codes_set(h, "typeOfLevel", kind)
            codes.codes_set(h, "level", level)
            pattern = np.arange(len(latitude) * len(longitude)).reshape(len(latitude), len(longitude))
            values = ((280 + pattern / 4) if units == "K" else
                      (.001953125 + pattern / 65536) if name == "specific_humidity_2m" else
                      (.125 + pattern / 1024) if units in ("1", "m") else .25 + pattern / 32)
            declared = "%" if name.endswith("albedo") else units
            encoded = values * 100 if declared == "%" else values.copy()
            if codes.codes_get(h, "units") in ("unknown", "~", ""):
                mapping[name + ".units"] = declared
            rows = np.arange(len(latitude))[::-1] if options.get("reverse_lat") else np.arange(len(latitude))
            columns = np.argsort(np.remainder(longitude - options.get("origin", 0.), 360.))
            if options.get("reverse_lon"):
                columns = columns[::-1]
            raw = encoded[np.ix_(rows, columns)]
            raw = (raw.T if options.get("column_scan") else raw).copy()
            if options.get("alternate_scan"):
                raw[1::2] = raw[1::2, ::-1]
            codes.codes_set_values(h, raw.reshape(-1))
            payloads.append(codes.codes_get_message(h))
            expected[name], grids[name] = values, (latitude, np.sort(np.remainder(longitude + shift, 360.)))
        finally:
            codes.codes_release(h)
    mapping.update({"deep_soil_temperature.type_of_level": "depthBelowLand", "deep_soil_temperature.level": "3"})
    path.write_bytes(b"".join(payloads))
    return expected, grids, mapping, payloads


@pytest.mark.parametrize("edition", (1, 2))
@pytest.mark.parametrize("scan", ({}, {"reverse_lat": True, "reverse_lon": True, "column_scan": True},
                                  {"grid": "regular_gg"}, {"origin": 180.}))
def test_complete_analytic_quantities_units_levels_and_scans(tmp_path, codes, edition, scan):
    path = tmp_path / "surface.nc"
    expected, grids, mapping, _ = write_ufs(codes, path, edition=edition, **scan)
    source = read_ufs_surface(path, mapping)
    assert source.lead == timedelta(hours=6) and source.field_grids is None
    assert set(source.fields) == set(expected) == set(UFS_SURFACE_VARIABLES)
    for name, values in expected.items():
        np.testing.assert_array_equal(source.fields[name], values)
        np.testing.assert_allclose(source.latitude, grids[name][0], rtol=0, atol=3.e-14)
        np.testing.assert_array_equal(source.longitude, grids[name][1])
        assert source.fields[name].dtype == torch.float64


def test_split_files_short_name_and_independent_grids(tmp_path, codes):
    path = tmp_path / "all.grib2"
    expected, grids, mapping, payloads = write_ufs(codes, path, overrides={
        "friction_velocity": {"shift": 9.}, "sea_ice_thickness": {"grid": "regular_gg"}})
    paths = [tmp_path / "odd", tmp_path / "even"]
    for offset, part in enumerate(paths):
        part.write_bytes(b"".join(payloads[offset::2]))
    h = codes.codes_new_from_message(payloads[0])
    try:
        mapping["aerodynamic_roughness_length"] = codes.codes_get(h, "shortName")
        assert mapping["aerodynamic_roughness_length"] != "unknown"
    finally:
        codes.codes_release(h)
    source = read_ufs_surface(paths, mapping)
    assert set(source.field_grids) == {"friction_velocity", "sea_ice_thickness"}
    for name, values in expected.items():
        np.testing.assert_array_equal(source.fields[name], values)
        if name in source.field_grids:
            np.testing.assert_allclose(source.field_grids[name].latitude, grids[name][0], rtol=0, atol=3.e-14)
            np.testing.assert_array_equal(source.field_grids[name].longitude, grids[name][1])


def test_member_selection_and_unselected_records(tmp_path, codes):
    first, second = tmp_path / "member3", tmp_path / "member4"
    expected, _, mapping, _ = write_ufs(codes, first, member=3)
    write_ufs(codes, second, member=4)
    with pytest.raises(ValueError, match="exactly one GRIB forecast"):
        read_ufs_surface([first, second], mapping)
    result = read_ufs_surface([first, second], mapping, member=3)
    assert result.member == 3 and result.lead == timedelta(hours=6)
    for name, values in expected.items():
        np.testing.assert_array_equal(result.fields[name], values)


def test_known_percent_albedo_and_conflicting_declaration(tmp_path, codes):
    path = tmp_path / "known-percent.grib2"
    expected, _, mapping, payloads = write_ufs(codes, path)
    name = "visible_strong_cosine_albedo"
    index = list(UFS_SURFACE_VARIABLES).index(name)
    h = codes.codes_new_from_message(payloads[index])
    try:
        codes.codes_set(h, "parameterCategory", 19)
        codes.codes_set(h, "parameterNumber", 19)
        assert codes.codes_get(h, "units") == "%"
        payloads[index] = codes.codes_get_message(h)
    finally:
        codes.codes_release(h)
    path.write_bytes(b"".join(payloads))
    mapping[name] = "grib2:7:36:254:0:19:19"
    mapping.pop(name + ".units")
    source = read_ufs_surface(path, mapping)
    np.testing.assert_array_equal(source.fields[name], expected[name])
    with pytest.raises(ValueError, match="cannot be overridden"):
        read_ufs_surface(path, {**mapping, name + ".units": "1"})


def test_reduced_analyzed_grid_rejects(tmp_path, codes):
    from test_grib_reduced import message as reduced_message
    path = tmp_path / "reduced.grib2"
    _, _, mapping, payloads = write_ufs(codes, path)
    payloads[0] = reduced_message(codes, np.ones(24), (4, 8, 8, 4), centre=7, tablesVersion=36,
        localTablesVersion=254, discipline=2, parameterCategory=0, parameterNumber=1,
        typeOfLevel="surface", level=0, endStep=21600)
    path.write_bytes(b"".join(payloads))
    with pytest.raises(ValueError, match="rectangular grids"):
        read_ufs_surface(path, mapping)


@pytest.mark.parametrize("defect,match", (
    ("mapping", "explicit mapping"), ("selector", "numeric parameter"), ("centre", "every complete"),
    ("table", "every complete"), ("units_missing", "unsupported GRIB units"),
    ("units_override", "cannot be overridden"), ("units_invalid", "units declaration"),
    ("deep_missing", "explicit point"), ("deep_nan", "positive depth"), ("deep_layer", "positive depth"),
    ("missing", "every complete"), ("duplicate", "duplicate"), ("wrong_height", "every complete"),
    ("bitmap", "missing GRIB pixels"), ("time", "exactly one GRIB forecast"), ("statistic", "instantaneous")))
def test_incomplete_or_ambiguous_grib_rejects(tmp_path, codes, defect, match):
    path = tmp_path / "bad.grib2"
    _, _, mapping, payloads = write_ufs(codes, path)
    rough, unknown = "aerodynamic_roughness_length", "visible_strong_cosine_albedo"
    if defect == "mapping": mapping.pop(rough)
    elif defect == "selector": mapping[rough] = "grib2:7:bad"
    elif defect == "centre": mapping[rough] = mapping[rough].replace(":7:", ":98:", 1)
    elif defect == "table": mapping[rough] = mapping[rough].replace(":254:", ":253:", 1)
    elif defect == "units_missing": mapping.pop(unknown + ".units")
    elif defect == "units_override":
        mapping[rough], mapping[rough + ".units"] = "grib2:7:36:254:2:0:0", "m"
        h = codes.codes_new_from_message(payloads[0])
        try:
            codes.codes_set(h, "parameterNumber", 0)
            payloads[0] = codes.codes_get_message(h)
        finally:
            codes.codes_release(h)
        path.write_bytes(b"".join(payloads))
    elif defect == "units_invalid": mapping[unknown + ".units"] = "Pa"
    elif defect == "deep_missing": mapping.pop("deep_soil_temperature.level")
    elif defect == "deep_nan": mapping["deep_soil_temperature.level"] = "nan"
    elif defect == "deep_layer": mapping["deep_soil_temperature.type_of_level"] = "depthBelowLandLayer"
    elif defect == "missing": path.write_bytes(b"".join(payloads[1:]))
    elif defect == "duplicate": path.write_bytes(b"".join(payloads + payloads[:1]))
    else:
        def change(h):
            if codes.codes_get(h, "parameterCategory") != 0 or codes.codes_get(h, "discipline") != 0:
                return
            if defect == "wrong_height": codes.codes_set(h, "level", 10)
            elif defect == "time": codes.codes_set(h, "dataTime", 1200)
            elif defect == "statistic": codes.codes_set(h, "stepType", "avg")
            elif defect == "bitmap":
                codes.codes_set(h, "bitmapPresent", 1)
                codes.codes_set(h, "missingValue", 9999.)
                values = codes.codes_get_values(h); values[0] = 9999.; codes.codes_set_values(h, values)
        rewrite(codes, path, change)
    with pytest.raises(ValueError, match=match):
        read_ufs_surface(path, mapping)
