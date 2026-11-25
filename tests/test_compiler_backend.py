"""Small execution and AD gates for the experimental compiler, not release qualification."""

import pytest
import torch

from mpas_init.vertical_grid import tc_interfaces


pytestmark = pytest.mark.compiler


def guarded(value):
    torch._assert_async((value > 0).all(), "positive source required")
    return value.square() + value.sin()


@pytest.fixture
def compiler():
    from mpas_init import _compiler
    return _compiler


@pytest.fixture
def observed_regions(monkeypatch, compiler):
    torch._dynamo.reset()
    original = compiler.compile_fx_inner
    regions = []

    def observe(graph, inputs, **kwargs):
        function = original(graph, inputs, **kwargs)
        entry = {"calls": 0}
        regions.append(entry)

        def execute(*values):
            entry["calls"] += 1
            return function(*values)

        return execute

    monkeypatch.setattr(compiler, "compile_fx_inner", observe)
    return regions


def values(fragmented):
    return (torch.linspace(.5, 2., 12, dtype=torch.float64).reshape(3, 4).T if fragmented
            else torch.tensor([.5, 1., 2.], dtype=torch.float64))


@pytest.mark.parametrize("fragmented", (False, True))
def test_checked_backend_executes_inductor_and_retains_native_guard(observed_regions, monkeypatch, compiler, fragmented):
    if fragmented:
        monkeypatch.setattr(compiler, "_MAX_NODES", 4)
    regions = observed_regions
    compiled = torch.compile(guarded, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    value = values(fragmented)
    actual = compiled(value)
    torch.testing.assert_close(actual, guarded(value), rtol=64 * torch.finfo(value.dtype).eps, atol=1.e-15, check_stride=True)
    assert regions and all(entry["calls"] > 0 for entry in regions)
    count = len(regions)
    with pytest.raises(RuntimeError, match="positive source required"):
        compiled(-value)
    assert len(regions) == count


@pytest.mark.parametrize("fragmented", (False, True))
def test_checked_backend_scalar_reverse_matches_analytic_and_finite_difference(observed_regions, monkeypatch, compiler, fragmented):
    if fragmented:
        monkeypatch.setattr(compiler, "_MAX_NODES", 4)
    compiled = torch.compile(guarded, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    value = values(fragmented).requires_grad_()
    gradient, = torch.autograd.grad(compiled(value).sum(), value)
    analytic = 2 * value + value.cos()
    torch.testing.assert_close(gradient, analytic, rtol=2.e-6, atol=2.e-8)
    direction = torch.cos(torch.arange(value.numel(), dtype=value.dtype)).reshape(value.shape) / 2
    for step in (.001, .0005, .00025):
        finite = (guarded(value + step * direction) - guarded(value - step * direction)) / (2 * step)
        torch.testing.assert_close(gradient * direction, finite, rtol=2.e-6, atol=2.e-8)
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_checked_backend_ordered_sum_values_and_ad(observed_regions, compiler, dtype):
    from mpas_init.gwd import _ordered_sum

    value = (torch.arange(48, dtype=dtype).reshape(2, 6, 4) * .37).sin().transpose(-2, -1).requires_grad_()
    direction = torch.cos(torch.arange(48, dtype=dtype)).reshape_as(value)
    compiled = torch.compile(_ordered_sum, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    expected = value.flatten(-2).cumsum(-1)[..., -1]
    actual = compiled(value)
    assert torch.equal(actual, expected)
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    weights = torch.tensor([.25, 1.25], dtype=dtype)
    reverse, = torch.autograd.grad((actual * weights).sum(), value)
    assert torch.equal(reverse, weights[:, None, None].expand_as(value))

    def transformed(x, dx):
        return torch.func.jvp(_ordered_sum, (x,), (dx,))

    compiled_jvp = torch.compile(transformed, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    primal, tangent = compiled_jvp(value, direction)
    assert torch.equal(primal, expected)
    assert torch.equal(tangent, direction.flatten(-2).cumsum(-1)[..., -1])
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_checked_backend_inventory_values_and_all_argument_ad(observed_regions, compiler, dtype, tmp_path):
    import math
    import numpy as np
    from test_inventory import case, evaluate, flatten, scalar_inventory, scalar_flatten, assert_inventory

    values = tuple(value.requires_grad_() for value in case(dtype))
    directions = tuple(torch.cos(torch.arange(value.numel(), dtype=dtype).reshape(value.shape) + index) * .01
                       for index, value in enumerate(values))
    compiled = torch.compile(evaluate, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    result = compiled(*values)
    assert_inventory(result, scalar_inventory(values), dtype)
    weights = torch.linspace(-.3, .7, flatten(result).numel(), dtype=dtype)
    reverse = torch.autograd.grad((flatten(result) * weights).sum(), values)
    payload = {"primal_actual": flatten(result).detach().numpy(),
               "primal_reference": np.asarray(scalar_flatten(scalar_inventory(values)))}
    for argument, value in enumerate(values):
        analytic = []
        for index in range(value.numel()):
            basis = [torch.zeros_like(x) for x in values]
            basis[argument].reshape(-1)[index] = 1.
            derivative = scalar_flatten(scalar_inventory(values, basis))
            analytic.append(math.fsum(x * y for x, y in zip(derivative, weights.tolist())))
        torch.testing.assert_close(reverse[argument], torch.tensor(analytic, dtype=dtype).reshape(value.shape), rtol=2.e-6, atol=2.e-8)
        payload["vjp_actual__" + str(argument)] = reverse[argument].detach().numpy()
        payload["vjp_reference__" + str(argument)] = np.asarray(analytic).reshape(value.shape)

    def transformed(*args):
        return torch.func.jvp(evaluate, args[:len(values)], args[len(values):])

    compiled_jvp = torch.compile(transformed, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    primal, tangent = compiled_jvp(*values, *directions)
    assert_inventory(primal, scalar_inventory(values), dtype)
    expected = torch.tensor(scalar_flatten(scalar_inventory(values, directions)), dtype=dtype)
    torch.testing.assert_close(flatten(tangent), expected, rtol=2.e-6, atol=2.e-8)
    payload["jvp_actual"], payload["jvp_reference"] = flatten(tangent).detach().numpy(), expected.numpy()
    for argument in range(len(values)):
        isolated = tuple(direction if i == argument else torch.zeros_like(direction) for i, direction in enumerate(directions))
        _, partial = compiled_jvp(*values, *isolated)
        reference = torch.tensor(scalar_flatten(scalar_inventory(values, isolated)), dtype=dtype)
        torch.testing.assert_close(flatten(partial), reference, rtol=2.e-6, atol=2.e-8)
        payload["partial_actual__" + str(argument)] = flatten(partial).detach().numpy()
        payload["partial_reference__" + str(argument)] = reference.numpy()
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)
    np.savez(tmp_path / "compiled-derivatives.npz", **payload)


@pytest.mark.parametrize("fragmented", (False, True))
def test_checked_backend_compiled_scalar_jvp(observed_regions, monkeypatch, compiler, fragmented):
    if fragmented:
        monkeypatch.setattr(compiler, "_MAX_NODES", 4)
    def transformed(value, direction):
        return torch.func.jvp(guarded, (value,), (direction,))

    compiled = torch.compile(transformed, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    value = values(fragmented)
    direction = torch.cos(torch.arange(value.numel(), dtype=value.dtype)).reshape(value.shape) / 2
    primal, tangent = compiled(value, direction)
    torch.testing.assert_close(primal, guarded(value), rtol=2.e-6, atol=2.e-8)
    torch.testing.assert_close(tangent, direction * (2 * value + value.cos()), rtol=2.e-6, atol=2.e-8)
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)
    with pytest.raises(RuntimeError, match="positive source required"):
        compiled(-value, direction)


@pytest.mark.parametrize("levels", (3, 55))
def test_checked_backend_preserves_reference_interface_arithmetic(observed_regions, compiler, levels):
    compiled = torch.compile(lambda top: tc_interfaces(levels, top), backend=compiler.checked_backend,
                             fullgraph=True, dynamic=False)
    for height in (30000., 27000.):
        top = torch.tensor(height, dtype=torch.float64)
        assert torch.equal(compiled(top), tc_interfaces(levels, top))
    constant_top = lambda scale: scale * tc_interfaces(levels, scale.new_tensor(30000.))
    compiled = torch.compile(constant_top, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    scale = torch.tensor(1., dtype=torch.float64)
    assert torch.equal(compiled(scale), constant_top(scale))
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


@pytest.mark.parametrize("levels", (3, 55))
def test_checked_backend_interface_forward_reverse_and_finite_difference(observed_regions, compiler, levels):
    function = lambda top: tc_interfaces(levels, top)
    transformed = lambda top, direction: torch.func.jvp(function, (top,), (direction,))
    compiled = torch.compile(transformed, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    top = torch.tensor(30000., dtype=torch.float64)
    direction = top.new_tensor(.7)
    primal, tangent = compiled(top, direction)
    assert torch.equal(primal, function(top))
    for step in (1., .5, .25):
        finite = (function(top + step * direction) - function(top - step * direction)) / (2 * step)
        torch.testing.assert_close(tangent, finite, rtol=2.e-6, atol=2.e-8)
    top.requires_grad_()
    compiled = torch.compile(function, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    weights = torch.cos(torch.arange(levels + 1, dtype=top.dtype))
    reverse, = torch.autograd.grad((compiled(top) * weights).sum(), top)
    torch.testing.assert_close(reverse * direction, (tangent * weights).sum(), rtol=2.e-6, atol=2.e-8)
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


def test_checked_backend_compiles_consumers_of_native_arithmetic(observed_regions, compiler):
    function = lambda value: (value / 3.).sin() + value
    compiled = torch.compile(function, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    value = torch.linspace(.1, 2., 7, dtype=torch.float64)
    actual = compiled(value)
    torch.testing.assert_close(actual, function(value), rtol=64 * torch.finfo(value.dtype).eps, atol=1.e-15)
    assert len(observed_regions) == 1 and observed_regions[0]["calls"] == 1


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("strided", (False, True))
def test_checked_backend_preserves_trigonometry_in_wind_projection(observed_regions, compiler, dtype, strided):
    from mpas_init.arithmetic import _sincos

    def project(angle, east, north):
        sine, cosine = _sincos(angle)
        return sine, cosine, cosine * east + sine * north

    # These two actual edge angles expose the ATen/scalar-libm difference.
    angle = torch.cat((torch.linspace(-2 * torch.pi, 2 * torch.pi, 257, dtype=dtype),
                       torch.tensor([-.6976799589230402, -.4619247131473842], dtype=dtype)))
    if strided:
        angle = angle[:, None].expand(-1, 3).clone().T
        assert not angle.is_contiguous()
    east = torch.cos(torch.arange(angle.numel(), dtype=dtype)).reshape(angle.shape) * 25
    north = torch.sin(torch.arange(angle.numel(), dtype=dtype)).reshape(angle.shape) * 17
    compiled = torch.compile(project, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    for offset in (0., .1):
        values = (angle, east + offset, north - offset)
        for actual, expected in zip(compiled(*values), project(*values)):
            assert torch.equal(actual, expected)
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


def test_checked_backend_wind_projection_forward_reverse_and_finite_difference(observed_regions, compiler):
    def project(angle, east, north):
        return torch.cos(angle) * east + torch.sin(angle) * north

    angle = torch.tensor([-.6976799589230402, -.4619247131473842, .37], dtype=torch.float64)
    values = (angle, angle.new_tensor([13., -21., 5.]), angle.new_tensor([-7., 19., 9.]))
    directions = tuple(torch.cos(torch.arange(3, dtype=angle.dtype) + i) * .2 for i in range(3))
    transformed = lambda *args: torch.func.jvp(project, args[:3], args[3:])
    compiled_jvp = torch.compile(transformed, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    primal, tangent = compiled_jvp(*values, *directions)
    assert torch.equal(primal, project(*values))
    analytic = (-angle.sin() * values[1] + angle.cos() * values[2]) * directions[0]
    analytic = analytic + angle.cos() * directions[1] + angle.sin() * directions[2]
    torch.testing.assert_close(tangent, analytic, rtol=2.e-6, atol=2.e-8)
    assert torch.count_nonzero(tangent) == tangent.numel()
    for step in (.001, .0005, .00025):
        plus = tuple(x + step * dx for x, dx in zip(values, directions))
        minus = tuple(x - step * dx for x, dx in zip(values, directions))
        finite = (project(*plus) - project(*minus)) / (2 * step)
        torch.testing.assert_close(tangent, finite, rtol=2.e-6, atol=2.e-8)
    differentiable = tuple(value.requires_grad_() for value in values)
    compiled = torch.compile(project, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    weights = angle.new_tensor([.3, -.7, 1.1])
    reverse = torch.autograd.grad((compiled(*differentiable) * weights).sum(), differentiable)
    expected = ((-angle.sin() * values[1] + angle.cos() * values[2]) * weights,
                angle.cos() * weights, angle.sin() * weights)
    for actual, reference in zip(reverse, expected):
        torch.testing.assert_close(actual, reference, rtol=2.e-6, atol=2.e-8)
        assert torch.count_nonzero(actual) == actual.numel()
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


def test_checked_backend_monthly_calendar_guards_date_metadata(observed_regions, compiler):
    from datetime import datetime
    from mpas_init.surface import monthly_interpolation

    compiled = torch.compile(monthly_interpolation, backend=compiler.checked_backend,
                             fullgraph=True, dynamic=False)
    monthly = torch.arange(36, dtype=torch.float64).reshape(3, 12).sin()
    for date in (datetime(2024, 2, 29), datetime(2023, 3, 1), datetime(2024, 1, 1),
                 datetime(2024, 12, 31), datetime(2024, 5, 15), datetime(2024, 5, 16)):
        assert torch.equal(compiled(monthly, date), monthly_interpolation(monthly, date)), date
        assert torch.equal(compiled(monthly + .3, date), monthly_interpolation(monthly + .3, date)), date
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


def test_checked_backend_captures_surface_jvp_with_distinct_times(observed_regions, compiler):
    from dataclasses import replace
    from mpas_init.ufs_surface import initialize_ufs_surface
    from test_ufs_surface import generated_inputs

    source, atmosphere, target, native, surface = generated_inputs()
    source = replace(source, valid_time=source.valid_time.replace(), reference_time=source.reference_time.replace())
    assert source.valid_time == atmosphere.valid_time and source.valid_time is not atmosphere.valid_time
    names = tuple(source.fields)
    values = tuple(source.fields[name] for name in names)
    directions = tuple(.01 * value * torch.cos(torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) + i)
                       for i, value in enumerate(values))

    def evaluate(*values):
        result = initialize_ufs_surface(replace(source, fields=dict(zip(names, values))),
                                        atmosphere, target, native, surface)
        return tuple(result[name] for name in sorted(result))

    transformed = lambda *args: torch.func.jvp(evaluate, args[:len(values)], args[len(values):])
    compiled = torch.compile(transformed, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    primal, tangent = compiled(*values, *directions)
    eager, expected = torch.func.jvp(evaluate, values, directions)
    for actual, reference in zip(primal, eager):
        assert torch.equal(actual, reference)
    for actual, reference in zip(tangent, expected):
        torch.testing.assert_close(actual, reference, rtol=2.e-6, atol=2.e-8)
        assert torch.count_nonzero(actual)
    for step in (.001, .0001, .00001):
        plus = evaluate(*(value + step * delta for value, delta in zip(values, directions)))
        minus = evaluate(*(value - step * delta for value, delta in zip(values, directions)))
        for actual, upper, lower in zip(tangent, plus, minus):
            torch.testing.assert_close(actual, (upper - lower) / (2 * step), rtol=2.e-6, atol=2.e-8)
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)


def test_checked_backend_guards_all_forecast_identity_parts(observed_regions, compiler):
    from dataclasses import replace
    from datetime import timedelta
    from mpas_init.netcdf import same_forecast_identity
    from test_ufs_surface import generated_inputs

    source, atmosphere, *_ = generated_inputs()
    source = replace(source, valid_time=source.valid_time.replace(), reference_time=source.reference_time.replace())

    def evaluate(value, selected):
        return value.square() if same_forecast_identity(selected, atmosphere) else value + 7

    compiled = torch.compile(evaluate, backend=compiler.checked_backend, fullgraph=True, dynamic=False)
    value = torch.tensor([.3, 1.7], dtype=torch.float64)
    assert torch.equal(compiled(value, source), value.square())
    for attribute, changed in (("valid_time", source.valid_time + timedelta(microseconds=1)),
                               ("reference_time", source.reference_time + timedelta(microseconds=1)),
                               ("lead", source.lead + timedelta(microseconds=1)), ("member", 1)):
        assert torch.equal(compiled(value, replace(source, **{attribute: changed})), value + 7)
    assert torch.equal(compiled(value, source), value.square())
    assert observed_regions and all(entry["calls"] > 0 for entry in observed_regions)
