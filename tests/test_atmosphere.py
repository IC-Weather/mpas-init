from datetime import timedelta

from netCDF4 import Dataset
import numpy as np
import pytest
import torch

from mpas_init.netcdf import REQUIRED, read_atmosphere, read_pressure_levels
from mpas_init.thermodynamics import log_pressure_columns
from mpas_init.vertical import hybrid_pressure
from test_vertical_grid import assert_stage


def generated_source(path, representation, *, reverse=False):
    """Independent moist-isothermal columns; not a real producer fixture."""
    lat, lon = np.linspace(-90., 90., 5), np.arange(8) * 45.
    ap = np.array([0., 800., 2000., 3000., 4000., 2000., 100.])
    b = np.array([1., .85, .65, .4, .2, .01, .001])
    ps = 96000. + 3000. * np.cos(np.deg2rad(lat[:, None])) ** 2 * np.cos(np.deg2rad(lon[None, :]))
    p = ap[:, None, None] + b[:, None, None] * ps
    height = 287. * 280. * (1. + (461.6 / 287. - 1.) * .004) / 9.80616 * np.log(ps / p)
    fields = {"air_temperature": np.full(p.shape, 280.), "geopotential_height": height,
              "specific_humidity": np.full(p.shape, .004), "eastward_wind": 10. + .0001 * height,
              "northward_wind": -3. + .00005 * height, "relative_humidity": np.full(p.shape, 45.)}
    if reverse:
        lat, lon, ap, b, ps, p = lat[::-1], lon[::-1] - 360., ap[::-1], b[::-1], ps[::-1, ::-1], p[::-1, ::-1, ::-1]
        fields = {name: value[::-1, ::-1, ::-1] for name, value in fields.items()}
    with Dataset(path, "w") as ds:
        for name, size in (("time", 1), ("level", 7), ("latitude", 5), ("longitude", 8)):
            ds.createDimension(name, size)

        def write(name, dims, value, units, standard=None):
            var = ds.createVariable(name, "f8", dims)
            var[:] = value
            var.units = units
            if standard:
                var.standard_name = standard
            return var

        write("latitude", ("latitude",), lat, "degrees_north", "latitude")
        write("longitude", ("longitude",), lon, "degrees_east", "longitude")
        write("time", ("time",), [21600.], "seconds since 2024-05-21 00:00:00", "time")
        write("reference", (), 0., ds["time"].units, "forecast_reference_time")
        write("lead", (), 6., "hours", "forecast_period")
        order = ("longitude", "time", "latitude", "level")
        for name, value in fields.items():
            var = write(name, order, value.transpose(2, 1, 0)[:, None], REQUIRED.get(name, "%"), name)
            if name == "relative_humidity":
                var.mpas_init_saturation_reference = "liquid_water"
        # A common file can contain two temperatures on different level types.
        write("temperature_2m", ("time", "latitude", "longitude"), np.full((1, 5, 8), 280.), "K", "air_temperature")
        if representation == "3d":
            write("pressure", order, p.transpose(2, 1, 0)[:, None], "Pa", "air_pressure")
        else:
            lev = write("level", ("level",), np.arange(7)[::-1] if reverse else np.arange(7), "1",
                        "atmosphere_hybrid_sigma_pressure_coordinate")
            write("bb", ("level",), b, "1")
            write("ps", ("longitude", "time", "latitude"), (ps / 100.).T[:, None], "hPa", "surface_air_pressure")
            if representation == "a":
                write("aa", ("level",), ap / 100000., "1")
                write("p0", (), 1000., "hPa", "reference_air_pressure_for_atmosphere_vertical_coordinate")
                lev.formula_terms = " b: bb\n ps: ps  p0: p0 a: aa "
            else:
                write("aa", ("level",), ap / 100., "hPa")
                lev.formula_terms = "ap: aa b: bb ps: ps"
    return p, fields


@pytest.mark.parametrize("representation", ["3d", "a", "ap"])
@pytest.mark.parametrize("reverse", [False, True])
def test_full_levels_cf_forms_units_axes_and_forecast(tmp_path, representation, reverse):
    path = tmp_path / "source.nc"
    p, fields = generated_source(path, representation, reverse=reverse)
    source = read_atmosphere(path, include_relative_humidity=True)
    if reverse:
        p, fields = p[..., ::-1], {name: value[..., ::-1] for name, value in fields.items()}
    else:
        p, fields = p[:, ::-1], {name: value[:, ::-1] for name, value in fields.items()}
    assert torch.all(torch.diff(source.longitude) > 0)
    assert torch.all(torch.diff(source.latitude) < 0)
    assert source.lead == timedelta(hours=6)
    assert source.reference_time + source.lead == source.valid_time
    assert_stage(source.pressure, torch.tensor(p.copy()), 16)
    for name, values in fields.items():
        assert torch.equal(source.fields[name], torch.tensor(values.copy()))
    if representation == "3d":
        with Dataset(path, "a") as ds:
            ds["pressure"][:] = ds["pressure"][:] / 100.
            ds["pressure"].units = "hPa"
            ds.renameVariable("pressure", "renamed_pressure")
            ds["renamed_pressure"].delncattr("standard_name")
        actual = read_atmosphere(path, {"air_pressure": "renamed_pressure"})
        assert_stage(actual.pressure, torch.tensor(p.copy()), 16)
    with pytest.raises(ValueError):
        read_pressure_levels(path)


@pytest.mark.parametrize("defect,error", [
    ("missing_p0", "formula_terms"), ("duplicate_term", "exactly once"), ("both_a_and_ap", "exactly once"),
    ("syntax", "formula_terms"), ("missing_variable", "missing hybrid"), ("a_units", "hybrid a"),
    ("b_units", "hybrid b"), ("p0_units", "hybrid p0"), ("p0_vector", "hybrid p0"),
    ("ps_units", "hybrid ps"), ("ps_mean", "point samples"), ("interfaces", "full-level"),
    ("missing_height", "geopotential_height"), ("missing_value", "missing or nonfinite"),
    ("height_order", "decrease with increasing height"), ("zero_ps", "surface pressure"),
    ("zero_pressure", "full-level pressure"), ("mean", "point samples"),
    ("wrong_time", "differs from valid time"),
])
def test_ambiguous_or_incomplete_hybrid_rejected(tmp_path, defect, error):
    path = tmp_path / "source.nc"
    generated_source(path, "a")
    with Dataset(path, "a") as ds:
        if defect == "missing_p0":
            ds["level"].formula_terms = "a: aa b: bb ps: ps"
        elif defect == "duplicate_term":
            ds["level"].formula_terms += " b: bb"
        elif defect == "both_a_and_ap":
            ds["level"].formula_terms += " ap: aa"
        elif defect == "syntax":
            ds["level"].formula_terms += " ; execute()"
        elif defect == "missing_variable":
            ds.renameVariable("aa", "other")
        elif defect in ("a_units", "b_units", "ps_units", "p0_units"):
            ds[{"a_units": "aa", "b_units": "bb", "ps_units": "ps", "p0_units": "p0"}[defect]].units = "unknown"
        elif defect == "p0_vector":
            ds["level"].formula_terms = "a: aa b: bb ps: ps p0: aa"
        elif defect == "ps_mean":
            ds["ps"].cell_methods = "time: mean"
        elif defect == "interfaces":
            ds.createDimension("interface", 8)
            ds.createVariable("interface_a", "f8", ("interface",)).units = "1"
            ds["level"].formula_terms = "a: interface_a b: bb ps: ps p0: p0"
        elif defect == "missing_height":
            ds["geopotential_height"].delncattr("standard_name")
        elif defect == "missing_value":
            ds["air_temperature"][2, 0, 3, 4] = np.ma.masked
        elif defect == "height_order":
            ds["geopotential_height"][:] = ds["geopotential_height"][:, :, :, ::-1].copy()
        elif defect == "zero_ps":
            ds["ps"][0, 0, 0] = 0.
        elif defect == "zero_pressure":
            ds["aa"][0], ds["bb"][0] = 0., 0.
        elif defect == "mean":
            ds["air_temperature"].cell_methods = "time: mean"
        elif defect == "wrong_time":
            ds["lead"].assignValue(5.)
    with pytest.raises((ValueError, RuntimeError), match=error):
        read_atmosphere(path)


def test_hybrid_analytic_forward_reverse_and_independent_fd():
    a = torch.tensor([.01, .03, .08, .1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([.9, .6, .3, .1], dtype=torch.float64, requires_grad=True)
    ps = torch.tensor([[98000., 101000.], [85000., 92000.]], dtype=torch.float64, requires_grad=True)
    p0 = torch.tensor(100000., dtype=torch.float64, requires_grad=True)
    inputs = (a, b, ps, p0)
    directions = (a.new_tensor([.002, -.001, .001, .003]), b.new_tensor([.001, -.002, .004, .003]),
                  ps.new_tensor([[70., -30.], [40., 90.]]), p0.new_tensor(80.))
    weights = torch.cos(torch.arange(16, dtype=torch.float64)).reshape(4, 2, 2)
    evaluate = lambda aa, bb, pp, pzero: hybrid_pressure(aa, bb, pp, pzero)
    primal = evaluate(*inputs)
    grads = torch.autograd.grad((primal * weights).sum(), inputs)
    expected_grads = ((weights * p0.detach()).sum((1, 2)), (weights * ps.detach()).sum((1, 2)),
                      (weights * b.detach()[:, None, None]).sum(0), (weights * a.detach()[:, None, None]).sum())
    for group in range(4):
        tangents = tuple(d if i == group else torch.zeros_like(d) for i, d in enumerate(directions))
        _, jvp = torch.func.jvp(evaluate, inputs, tangents)
        analytic = ((directions[0][:, None, None] * p0.detach(), directions[1][:, None, None] * ps.detach(),
                     b.detach()[:, None, None] * directions[2], a.detach()[:, None, None] * directions[3])[group]).expand_as(primal)
        assert_stage(jvp, analytic, 16)
        assert_stage(grads[group], expected_grads[group], 16)
        assert grads[group].abs().max() > 0
        numpy_inputs = [v.detach().numpy() for v in inputs]
        for step in (1.e-2, 1.e-3, 1.e-4):
            plus, minus = [v.copy() for v in numpy_inputs], [v.copy() for v in numpy_inputs]
            plus[group] += step * directions[group].numpy()
            minus[group] -= step * directions[group].numpy()
            def independent(v):
                return v[0][:, None, None] * v[3] + v[1][:, None, None] * v[2]
            fd = torch.tensor((independent(plus) - independent(minus)) / (2. * step))
            torch.testing.assert_close(jvp, fd, rtol=2.e-6, atol=2.e-8)
            torch.testing.assert_close((jvp * weights).sum(), (grads[group] * directions[group]).sum(), rtol=2.e-13, atol=1.e-12)
            with pytest.raises(AssertionError):
                torch.testing.assert_close(jvp * 0, fd, rtol=2.e-6, atol=2.e-8)
    # ap has pressure units, so its derivative is not multiplied by p0.
    ap = (a.detach() * p0.detach()).requires_grad_()
    output = hybrid_pressure(ap, b.detach(), ps.detach())
    derivative, = torch.autograd.grad((output * weights).sum(), (ap,))
    assert_stage(derivative, weights.sum((1, 2)), 16)
    _, tangent = torch.func.jvp(lambda aa: hybrid_pressure(aa, b.detach(), ps.detach()), (ap,), (directions[0],))
    assert_stage(tangent, directions[0][:, None, None].expand_as(primal), 16)


def pressure_remap_case():
    yy, xx = torch.meshgrid(torch.arange(9, dtype=torch.float64), torch.arange(12, dtype=torch.float64), indexing="ij")
    p = torch.tensor([90000., 60000., 30000.], dtype=torch.float64)[:, None, None] * (1. + .002 * (xx * xx + .5 * yy * yy + 2. * xx * yy))
    x, y = p.new_tensor([2.3, 4.7, 6.2]), p.new_tensor([2.4, 3.7, 5.1])
    expected_p = p.new_tensor([90000., 60000., 30000.])[None, :] * (1. + .002 * (x * x + .5 * y * y + 2. * x * y))[:, None]
    return p, x, y, expected_p


def test_pressure_remap_before_log_analytic_and_derivatives():
    p, x, y, expected_p = pressure_remap_case()
    actual = log_pressure_columns(p, x, y, 3)
    assert_stage(actual, torch.log(expected_p), 64)
    from mpas_init.horizontal import sixteen_point
    wrong = sixteen_point(torch.log(p), x, y).T
    with pytest.raises(AssertionError):
        assert_stage(wrong, torch.log(expected_p), 64)
    direction = p * .003
    _, jvp = torch.func.jvp(lambda value: log_pressure_columns(value, x, y, 3), (p,), (direction,))
    assert_stage(jvp, torch.full_like(jvp, .003), 64)
    pp = p.clone().requires_grad_()
    grad, = torch.autograd.grad(log_pressure_columns(pp, x, y, 3).sum(), pp)
    torch.testing.assert_close((grad * direction).sum(), jvp.sum(), rtol=2.e-13, atol=1.e-15)
    for step in (1.e-2, 1.e-3, 1.e-4):
        fd = torch.tensor((np.log(expected_p.numpy() * (1. + .003 * step))
                           - np.log(expected_p.numpy() * (1. - .003 * step))) / (2. * step))
        torch.testing.assert_close(jvp, fd, rtol=2.e-6, atol=2.e-8)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(jvp * 0, fd, rtol=2.e-6, atol=2.e-8)
    factor = 1. + .002 * (x * x + .5 * y * y + 2. * x * y)
    for axis, analytic in ((0, .002 * (2. * x + 2. * y) / factor), (1, .002 * (y + 2. * x) / factor)):
        tangents = (torch.ones_like(x), torch.zeros_like(y)) if axis == 0 else (torch.zeros_like(x), torch.ones_like(y))
        _, jvp = torch.func.jvp(lambda xx, yy: log_pressure_columns(p, xx, yy, 3), (x, y), tangents)
        assert_stage(jvp, analytic[:, None].expand_as(actual), 64)
        coordinates = (x.clone().requires_grad_(), y.clone().requires_grad_())
        gradient = torch.autograd.grad(log_pressure_columns(p, *coordinates, 3).sum(), coordinates)[axis]
        assert_stage(gradient, 3. * analytic, 64)
        for step in (1.e-3, 1.e-4, 1.e-5):
            def independent(sign):
                xx, yy = x.numpy().copy(), y.numpy().copy()
                if axis == 0:
                    xx += sign * step
                else:
                    yy += sign * step
                return np.log(1. + .002 * (xx * xx + .5 * yy * yy + 2. * xx * yy))
            fd = torch.tensor((independent(1) - independent(-1)) / (2. * step))[:, None].expand_as(jvp)
            torch.testing.assert_close(jvp, fd, rtol=2.e-6, atol=2.e-8)
            with pytest.raises(AssertionError):
                torch.testing.assert_close(jvp * 0, fd, rtol=2.e-6, atol=2.e-8)


@pytest.mark.parametrize("defect,error", [("zero", "positive"), ("negative", "positive"),
                                          ("missing", "missing or nonfinite"), ("order", "strictly monotonic"),
                                          ("mean", "point samples"), ("ambiguous", "one air_pressure")])
def test_invalid_explicit_pressure_rejected(tmp_path, defect, error):
    path = tmp_path / "source.nc"
    generated_source(path, "3d")
    with Dataset(path, "a") as ds:
        var = ds["pressure"]
        if defect in ("zero", "negative", "missing"):
            var[0, 0, 0, 0] = {"zero": 0., "negative": -10., "missing": np.ma.masked}[defect]
        elif defect == "order":
            var[0, 0, 0, 2] = var[0, 0, 0, 0]
        elif defect == "mean":
            var.cell_methods = "time: mean"
        else:
            duplicate = ds.createVariable("other_pressure", "f8", var.dimensions)
            duplicate[:] = var[:]
            duplicate.standard_name, duplicate.units = "air_pressure", "Pa"
    with pytest.raises(ValueError, match=error):
        read_atmosphere(path)
    if defect == "ambiguous":
        assert read_atmosphere(path, {"air_pressure": "pressure"}).pressure.shape == (7, 5, 8)
